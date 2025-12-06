"""
    implementation of other two-way contrastive losses
    Extended with InfoNCE and Curriculum-Weighted Contrastive Loss
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


# https://github.com/Spijkervet/SimCLR/blob/master/simclr/modules/gather.py
class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out

class iSogCLR_B_Loss(nn.Module):
    def __init__(
        self,
        N,
        tau_init=0.03,
        gamma=0.8,
        bsz=128,
        world_size=1,
        feature_dim=256,
        tau_min=0.005,
        tau_max=0.1,
        rho_I=8.0,
        rho_T=8.0,
        use_temp_net=True,
        tau_grad_scale=0.3,
        tempnet_warmup=30,
    ):
        super().__init__()

        self.bsz = bsz
        self.gamma = gamma
        self.world_size = world_size
        self.eps = 1e-14

        # EMA buffers
        self.s_I = torch.zeros(N).cuda()
        self.s_T = torch.zeros(N).cuda()
        self.b_I = torch.zeros(N).cuda()
        self.b_T = torch.zeros(N).cuda()

        self.neg_num = bsz - 1
        self.mask_neg = (1 - torch.eye(bsz).cuda())

        # TempNet settings
        self.use_temp_net = use_temp_net
        self.tau_grad_scale = tau_grad_scale
        self.tempnet_warmup = tempnet_warmup
        self.rho_I = rho_I
        self.rho_T = rho_T

        self.tau_min = tau_min
        self.tau_max = tau_max

        if use_temp_net:
            self.image_temp_net = TempGenerator(feature_dim, 256, tau_min, tau_max).cuda()
            self.text_temp_net  = TempGenerator(feature_dim, 256, tau_min, tau_max).cuda()
        else:
            self.tau_I = torch.ones(N).cuda() * tau_init
            self.tau_T = torch.ones(N).cuda() * tau_init


    def forward(self, image_feat, text_feat, image_ids, text_ids, epoch):

        bsz = image_feat.size(0)

        # === TempNet forward ===
        if self.use_temp_net:
            with torch.no_grad():
                im_det = image_feat.detach()
                tx_det = text_feat.detach()

            tau_I = self.image_temp_net(im_det)
            tau_T = self.text_temp_net(tx_det)
        else:
            tau_I = self.tau_I[image_ids]
            tau_T = self.tau_T[text_ids]

        # Similarity
        sim = image_feat @ text_feat.t()
        diag = torch.diagonal(sim)

        # Positive loss
        pos_loss = -2.0 * diag.mean()

        # Negative logits
        image_neg = (sim / tau_I[:, None]).detach()
        text_neg  = (sim / tau_T[None, :]).detach()

        # === EMA buffer updates ===
        old_bI = self.b_I[image_ids]
        new_bI = torch.max(image_neg, old_bI[:, None])
        self.b_I[image_ids] = new_bI.max(dim=1)[0]

        old_bT = self.b_T[text_ids]
        new_bT = torch.max(text_neg, old_bT[None, :])
        self.b_T[text_ids] = new_bT.max(dim=0)[0]

        exp_I = torch.exp(image_neg - self.b_I[image_ids][:, None]) * self.mask_neg
        exp_T = torch.exp(text_neg - self.b_T[text_ids][None, :]) * self.mask_neg

        g_I = exp_I.sum(dim=1, keepdim=True) / self.neg_num
        g_T = exp_T.sum(dim=0, keepdim=True) / self.neg_num

        # First epoch initialization
        if epoch == 0:
            s_I = g_I
            s_T = g_T
        else:
            s_I = (1 - self.gamma) * self.s_I[image_ids].unsqueeze(1) + self.gamma * g_I
            s_T = (1 - self.gamma) * self.s_T[text_ids].unsqueeze(0) + self.gamma * g_T

        self.s_I[image_ids] = s_I.squeeze()
        self.s_T[text_ids] = s_T.squeeze()

        s_I = s_I.clamp(min=self.eps)
        s_T = s_T.clamp(min=self.eps)

        # Weighted negative loss
        wI = exp_I / s_I
        wT = exp_T / s_T

        neg_I = (wI * sim).sum(1).mean() / self.neg_num
        neg_T = (wT * sim).sum(0).mean() / self.neg_num

        total_loss = pos_loss + neg_I + neg_T


        # === Temperature loss (STABLE VERSION) ===
        if self.use_temp_net and epoch >= self.tempnet_warmup:

            tempI = (
                torch.log(s_I) +
                self.b_I[image_ids].unsqueeze(1) +
                self.rho_I -
                (wI * image_neg).sum(1, keepdim=True) / self.neg_num
            ).clamp(-10, 10).detach()

            tempT = (
                torch.log(s_T) +
                self.b_T[text_ids].unsqueeze(0) +
                self.rho_T -
                (wT * text_neg).sum(0, keepdim=True) / self.neg_num
            ).clamp(-10, 10).detach()

            # Stable τ-grad form
            tau_loss_I = self.tau_grad_scale * ( (tau_I - tau_I.detach()) * tempI ).mean()
            tau_loss_T = self.tau_grad_scale * ( (tau_T - tau_T.detach()) * tempT ).mean()

            total_loss += tau_loss_I + tau_loss_T

        return total_loss
class CLIP_Loss(nn.Module):

    def __init__(self, world_size=8, temperature=0.01, personalized_tau=False, image_tau=None, text_tau=None):
        super(CLIP_Loss, self).__init__()
        self.world_size = world_size
        self.temperature = temperature
        self.personalized_tau = personalized_tau # if true, then temperatures are learnable
        self.image_tau = image_tau
        self.text_tau = text_tau

    def forward(self, image_features, text_features, image_idx=None, text_idx=None):
        if self.world_size > 1:
            image_features = torch.cat(GatherLayer.apply(image_features), dim=0)
            text_features = torch.cat(GatherLayer.apply(text_features), dim=0)

        if self.personalized_tau:
            image_temp = self.image_tau[image_idx]
            text_temp = self.text_tau[text_idx]
            sim = torch.einsum('i d, j d -> i j', text_features, image_features)
            labels = torch.arange(image_features.shape[0], device=image_features.device)
            total_loss = (F.cross_entropy(sim / text_temp, labels) + F.cross_entropy(sim.t() / image_temp, labels)) / 2

        else:
            sim = torch.einsum('i d, j d -> i j', text_features, image_features) / self.temperature
            labels = torch.arange(image_features.shape[0], device=image_features.device)
            total_loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2

        return total_loss



class SogCLR_Loss(nn.Module):
    def __init__(self, N=2900000, gamma=0.1, temperature=0.07, world_size=8, bsz=128, enable_surrogate=False, surrogate_c=1.0,
                lamda_rho=1.0, lamda_init=1.0):
        
        #Inputs:
        #   N is number of samples in training set
        
        super(SogCLR_Loss, self).__init__()
        self.world_size = world_size
        self.s_I = torch.zeros(N).cuda()
        self.s_T = torch.zeros(N).cuda()
        self.b_I = torch.zeros(N).cuda()
        self.b_T = torch.zeros(N).cuda()
        self.gamma = gamma
        self.temperature = temperature
        self.eps = 1e-8
        self.bsz = bsz
        self.mask_neg = (1.0 - torch.eye(bsz)).cuda()
        self.enable_surrogate = enable_surrogate
        self.c = surrogate_c # margin parameter for the square hinge loss

    def _sqh(self, x):
        return torch.max(torch.zeros_like(x), x + self.c) ** 2

    def forward(self, image_features, text_features, image_ids, text_ids, epoch):
        
        #Inputs:
        #    image_features, text_features is l2-normalized tensor
        #    image_features, text_features: [batch_size, emb_dim]
        
        if self.world_size > 1:
            image_features = torch.cat(GatherLayer.apply(image_features), dim=0)
            text_features = torch.cat(GatherLayer.apply(text_features), dim=0)

        # compute the logits (similarity between each image-text pair)
        sim = torch.einsum('i d, j d -> i j', image_features, text_features)
        diag_sim = torch.diagonal(sim)

        batch_size = sim.shape[0]

        # E_I(x_i)*E_T(t) - E_I(x_i)*E_T(t_i)
        image_diffs = sim - diag_sim[:, None]
        # E_I(x)*E_T(t_i) - E_I(x_i)*E_T(t_i)
        text_diffs = sim - diag_sim[None, :]

        if self.enable_surrogate:
            image_diffs = self._sqh(image_diffs)
            text_diffs = self._sqh(text_diffs)

        image_diffs_d_temps = (image_diffs / self.temperature).clone().detach_()
        text_diffs_d_temps = (text_diffs / self.temperature).clone().detach_()
        
        # update b
        old_b_I = self.b_I[image_ids]
        new_b_I = torch.max(image_diffs_d_temps, old_b_I[:, None].tile(1, batch_size))
        self.b_I[image_ids] = torch.max(new_b_I, dim=1)[0]

        old_b_T = self.b_T[text_ids]
        new_b_T = torch.max(text_diffs_d_temps, old_b_T[None, :].tile(batch_size, 1))
        self.b_T[text_ids] = torch.max(new_b_T, dim=0)[0]
        
        exp_image_diffs = torch.exp(image_diffs_d_temps - self.b_I[image_ids][:, None]) * self.mask_neg # -b to avoid exp operation overflow
        exp_text_diffs = torch.exp(text_diffs_d_temps - self.b_T[text_ids][None, :]) * self.mask_neg

        g_I = torch.sum(exp_image_diffs, dim=1, keepdim=True) / (batch_size-1)
        g_T = torch.sum(exp_text_diffs, dim=0, keepdim=True) / (batch_size-1)

        # print("g_I:", g_I.mean())
        # print("g_T:", g_T.mean())

        if epoch == 0:
            s_I = g_I
            s_T = g_T
        else:
            s_I = (1.0-self.gamma) * self.s_I[image_ids] * torch.exp(old_b_I - self.b_I[image_ids]) + self.gamma * g_I.squeeze()
            s_T = (1.0-self.gamma) * self.s_T[text_ids] * torch.exp(old_b_T - self.b_T[text_ids]) + self.gamma * g_T.squeeze()
            s_I = s_I.reshape(g_I.shape)
            s_T = s_T.reshape(g_T.shape)

        self.s_I[image_ids] = s_I.squeeze()
        self.s_T[text_ids] = s_T.squeeze()
        
        weights_image = exp_image_diffs / (s_I + self.eps)
        weights_text = exp_text_diffs / (s_T + self.eps)

        if torch.any(torch.isnan(weights_image)):
            assert 0, "weights_image has nan."
        if torch.any(torch.isnan(weights_text)):
            assert 0, "weights_text has nan."

        image_loss = torch.sum(weights_image * image_diffs, dim=1, keepdim=True) / (batch_size-1)
        text_loss = torch.sum(weights_text * text_diffs, dim=0, keepdim=True) / (batch_size-1)

        total_loss = image_loss.mean() + text_loss.mean()

        return total_loss, 0.0, 0.0



# add some new features to iSogCLR
class iSogCLR_New_Loss(nn.Module):
    def __init__(self, N=2900000, gamma=0.8, tau_init=0.01, world_size=8, bsz=128, rho_I=8.0, rho_T=8.0,
                       use_temp_net=True, feature_dim=256):  # use temperature network      
        
        #Inputs:
        #   N is number of samples in training set
        
        super(iSogCLR_New_Loss, self).__init__()
        self.world_size = world_size
        self.s_I = torch.zeros(N).cuda()
        self.s_T = torch.zeros(N).cuda()
        self.b_I = torch.zeros(N).cuda()
        self.b_T = torch.zeros(N).cuda()
        self.gamma = gamma
        self.eps = 1e-14
        self.bsz = bsz
        self.mask_neg = (1.0 - torch.eye(bsz)).cuda()

        self.tau_min, self.tau_max = 0.005, 0.05

        self.rho_I = rho_I
        self.rho_T = rho_T

        self.use_temp_net = use_temp_net

        self.eta_init = 1e-5  

        if self.use_temp_net:
            self.image_temp_gen = TempGenerator(feature_dim=feature_dim, M=256, tau_min=self.tau_min, tau_max=self.tau_max).cuda()
            self.text_temp_gen = TempGenerator(feature_dim=feature_dim, M=256, tau_min=self.tau_min, tau_max=self.tau_max).cuda()
        else:
            self.beta_u = 0.5
            self.grad_clip = 5.0
            self.tau_I = torch.ones(N).cuda() * tau_init
            self.tau_T = torch.ones(N).cuda() * tau_init
            self.u_I = torch.zeros(N).cuda()
            self.u_T = torch.zeros(N).cuda()


    def forward(self, image_features, text_features, image_ids, text_ids, epoch, max_epoch):
        
        #Inputs:
        #    image_features, text_features is l2-normalized tensor
        #    image_features, text_features: [batch_size, emb_dim]
        
        if self.world_size > 1:
            image_features = torch.cat(GatherLayer.apply(image_features), dim=0)
            text_features = torch.cat(GatherLayer.apply(text_features), dim=0)

        # compute the logits (similarity between each image-text pair)
        sim = torch.einsum('i d, j d -> i j', image_features, text_features)
        diag_sim = torch.diagonal(sim)

        batch_size = sim.shape[0]

        # generate temperatures
        if self.use_temp_net:
            tau_image = self.image_temp_gen(image_features.detach())
            tau_text = self.text_temp_gen(text_features.detach())
        else:
            tau_image = self.tau_I[image_ids]
            tau_text = self.tau_T[text_ids]

        # E_I(x_i)*E_T(t) - E_I(x_i)*E_T(t_i)
        image_diffs = sim - diag_sim[:, None]
        # E_I(x)*E_T(t_i) - E_I(x_i)*E_T(t_i)
        text_diffs = sim - diag_sim[None, :]

        image_diffs_d_temps = (image_diffs / tau_image[:, None]).detach()
        text_diffs_d_temps = (text_diffs / tau_text[None, :]).detach()

        # update b
        old_b_I = self.b_I[image_ids]
        new_b_I = torch.max(image_diffs_d_temps, old_b_I[:, None].tile(1, batch_size))
        self.b_I[image_ids] = torch.max(new_b_I, dim=1)[0]

        old_b_T = self.b_T[text_ids]
        new_b_T = torch.max(text_diffs_d_temps, old_b_T[None, :].tile(batch_size, 1))
        self.b_T[text_ids] = torch.max(new_b_T, dim=0)[0]

        exp_image_diffs = torch.exp(image_diffs_d_temps - self.b_I[image_ids][:, None]) * self.mask_neg # -b to avoid exp operation overflow
        exp_text_diffs = torch.exp(text_diffs_d_temps - self.b_T[text_ids][None, :]) * self.mask_neg

        g_I = torch.sum(exp_image_diffs, dim=1, keepdim=True)
        g_T = torch.sum(exp_text_diffs, dim=0, keepdim=True)

        if epoch == 0:
            s_I = g_I
            s_T = g_T
        else:
            s_I = (1.0-self.gamma) * self.s_I[image_ids] * torch.exp(old_b_I - self.b_I[image_ids]) + self.gamma * g_I.squeeze()
            s_T = (1.0-self.gamma) * self.s_T[text_ids] * torch.exp(old_b_T - self.b_T[text_ids]) + self.gamma * g_T.squeeze()
            s_I = s_I.reshape(g_I.shape)
            s_T = s_T.reshape(g_T.shape)

        self.s_I[image_ids] = s_I.squeeze()
        self.s_T[text_ids] = s_T.squeeze()

        s_I = s_I.clamp(min=self.eps)
        s_T = s_T.clamp(min=self.eps)

        weights_image = exp_image_diffs / s_I
        weights_text = exp_text_diffs / s_T

        image_loss = torch.sum(weights_image * image_diffs, dim=1, keepdim=True)
        text_loss = torch.sum(weights_text * text_diffs, dim=0, keepdim=True)

        total_loss = image_loss.mean() + text_loss.mean()
 
        temp_weight_image = torch.log(s_I / (batch_size-1)) + self.b_I[image_ids][:, None] + self.rho_I - torch.sum(weights_image * image_diffs_d_temps, dim=1, keepdim=True)
        temp_weight_text = torch.log(s_T / (batch_size-1)) + self.b_T[text_ids][None, :] + self.rho_T - torch.sum(weights_text * text_diffs_d_temps, dim=0, keepdim=True)

        if self.use_temp_net:
            temp_image_loss = torch.mean(temp_weight_image * tau_image[:, None])
            temp_text_loss = torch.mean(temp_weight_text * tau_text[None, :])

            total_loss += temp_image_loss + temp_text_loss

        else:
            self.u_I[image_ids] = (1.0-self.beta_u) * self.u_I[image_ids] + self.beta_u * temp_weight_image.squeeze().clamp_(min=-self.grad_clip, max=self.grad_clip)
            self.u_T[text_ids] = (1.0-self.beta_u) * self.u_T[text_ids] + self.beta_u * temp_weight_text.squeeze().clamp_(min=-self.grad_clip, max=self.grad_clip)

            self.tau_I[image_ids] = (tau_image - self.eta_init * self.u_I[image_ids]).clamp_(min=self.tau_min, max=self.tau_max)
            self.tau_T[text_ids] = (tau_text - self.eta_init * self.u_T[text_ids]).clamp_(min=self.tau_min, max=self.tau_max)

        return total_loss, tau_image.mean().item(), tau_text.mean().item(), self.eta_init,  \
                temp_weight_image.mean().item(), temp_weight_text.mean().item(), temp_weight_image.max().item(), temp_weight_image.min().item()



"""
    https://github.com/goel-shashank/CyCLIP/blob/52d77af2a5f1a4bff01b4c371d6b98e2d0340137/src/train.py
"""
class CyCLIP_Loss(nn.Module):
    def __init__(self, world_size, temperature, cylambda_1=0.25 , cylambda_2=0.25):
        super(CyCLIP_Loss, self).__init__()

        self.world_size = world_size
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.cylambda_1 = cylambda_1
        self.cylambda_2 = cylambda_2


    def forward(self, image_features, text_features):
        if self.world_size > 1:
            image_features = torch.cat(GatherLayer.apply(image_features), dim=0)
            text_features = torch.cat(GatherLayer.apply(text_features), dim=0)

        batch_size = len(image_features)

        logits_text_per_image = (image_features @ text_features.t()) / self.temperature
        logits_image_per_text = logits_text_per_image.t()

        target = torch.arange(batch_size).long().cuda()

        # contrastive loss, the same as CLIP
        contrastive_loss = (self.criterion(logits_text_per_image, target) + self.criterion(logits_image_per_text, target)) / 2.0 

        # inmodal_cyclic_loss
        logits_image_per_image = (image_features @ image_features.t()) / self.temperature
        logits_text_per_text = (text_features @ text_features.t()) / self.temperature
        inmodal_cyclic_loss = (logits_image_per_image - logits_text_per_text).square().mean() * (self.temperature ** 2) * batch_size

        # crossmodal_cyclic_loss
        crossmodal_cyclic_loss = (logits_text_per_image - logits_image_per_text).square().mean() * (self.temperature ** 2) * batch_size

        loss = contrastive_loss + self.cylambda_1 * inmodal_cyclic_loss + self.cylambda_2 * crossmodal_cyclic_loss

        return loss


"""
    VICReg
    https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py
"""
def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
    

class VICReg_Loss(nn.Module):
    def __init__(self, world_size, dim_size, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0):
        super(VICReg_Loss, self).__init__()

        self.world_size = world_size
        self.dim_size = dim_size
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff


    def forward(self, image_features, text_features):
        if self.world_size > 1:
            x = torch.cat(GatherLayer.apply(image_features), dim=0)
            y = torch.cat(GatherLayer.apply(text_features), dim=0)

        batch_size = len(x)

        repr_loss = F.mse_loss(x, y) # invariance term

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2  # variance term

        cov_x = (x.T @ x) / (batch_size - 1)
        cov_y = (y.T @ y) / (batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            self.dim_size
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.dim_size)  # covariance term

        loss = (
            self.sim_coeff * repr_loss
            + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
        )

        return loss


class TempGenerator(nn.Module):
    """
    Temperature network used in iSogCLR_B.
    Produces τ ∈ [tau_min, tau_max] in a stable manner.
    """

    def __init__(self, feature_dim, M=256, tau_min=0.005, tau_max=0.05, dropout_rate=0.5):
        super().__init__()

        hidden = max(64, feature_dim // 4)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)

        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),

            nn.Linear(hidden, 1)
        )

        # init: push τ toward τ_min initially (stable start)
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.net[-1].bias, -3.0)  # sig(-3)=0.047 → τ ≈ tau_min+small

    def forward(self, x):
        raw = self.net(x).squeeze(1)       # (B,)
        tau = torch.sigmoid(raw)           #
        tau = tau * (self.tau_max - self.tau_min) + self.tau_min
        return tau


# try to use temperature generator in place of individualized temperatures
class iSogCLR_New_v1_Loss(nn.Module):
    def __init__(self, N=2900000, gamma=0.8, tau_min=0.005, tau_max=1.0, rho_init=6.0, 
                 bsz=128, world_size=8, eps=1e-6, M=64, feature_dim=256):
        
        #Inputs:
        #   N is the number of samples in training set
        #   M is the number prototypes in the temperature generator
        
        super(iSogCLR_New_v1_Loss, self).__init__()
        self.world_size = world_size
        self.gamma = gamma

        self.s_I = torch.zeros(N).cuda()
        self.s_T = torch.zeros(N).cuda()
        self.b_I = torch.zeros(N).cuda()
        self.b_T = torch.zeros(N).cuda()
        self.rho_I = rho_init
        self.rho_T = rho_init
        self.eps = eps
        self.batch_size = bsz
        self.grad_clip = 5.0
        
        self.image_temp_gen = TempGenerator(feature_dim=feature_dim, M=M, tau_min=tau_min, tau_max=tau_max).cuda()
        self.text_temp_gen = TempGenerator(feature_dim=feature_dim, M=M, tau_min=tau_min, tau_max=tau_max).cuda()

        self.neg_num = bsz - 1
        self.mask_neg = (1.0 - torch.eye(bsz)).cuda()


    def forward(self, image_features, text_features, image_ids, text_ids, epoch):
        if self.world_size > 1:
            image_features = torch.cat(GatherLayer.apply(image_features), dim=0)
            text_features = torch.cat(GatherLayer.apply(text_features), dim=0)

        # generate temperatures first
        tau_image = self.image_temp_gen(image_features.detach())
        tau_text = self.text_temp_gen(text_features.detach())

        # compute the logits (similarity between each image-text pair)
        sim = torch.einsum('i d, j d -> i j', image_features, text_features)
        diag_sim = torch.diagonal(sim)

        batch_size = sim.shape[0]

        ####
        # consider the loss function first
        ####
        # the loss for positve pairs
        pos_term_loss = -2 * diag_sim

        # the loss for negative pairs
        image_temp_term = (sim / tau_image[:, None]).detach()
        text_temp_term = (sim / tau_text[None, :]).detach()

        # employ b to avoid exp operation overflow
        old_b_I = self.b_I[image_ids]
        new_b_I = torch.max(image_temp_term, old_b_I[:, None].tile(1, batch_size))
        self.b_I[image_ids] = torch.max(new_b_I, dim=1)[0]

        old_b_T = self.b_T[text_ids]
        new_b_T = torch.max(text_temp_term, old_b_T[None, :].tile(batch_size, 1))
        self.b_T[text_ids] = torch.max(new_b_T, dim=0)[0]

        exp_image_temp_term = torch.exp(image_temp_term - self.b_I[image_ids][:, None]) * self.mask_neg
        exp_text_temp_term = torch.exp(text_temp_term - self.b_T[text_ids][None, :]) * self.mask_neg

        g_I = torch.sum(exp_image_temp_term, dim=1, keepdim=True) / self.neg_num
        g_T = torch.sum(exp_text_temp_term, dim=0, keepdim=True) / self.neg_num

        # print("g_I:", g_I.mean())
        # print("g_T:", g_T.mean())

        if epoch == 0:
            s_I = g_I
            s_T = g_T
        else:
            s_I = (1.0-self.gamma) * self.s_I[image_ids] * torch.exp(old_b_I - self.b_I[image_ids]) + self.gamma * g_I.squeeze()
            s_T = (1.0-self.gamma) * self.s_T[text_ids] * torch.exp(old_b_T - self.b_T[text_ids]) + self.gamma * g_T.squeeze()
            s_I = s_I.reshape(g_I.shape)
            s_T = s_T.reshape(g_T.shape)

        self.s_I[image_ids] = s_I.squeeze()
        self.s_T[text_ids] = s_T.squeeze()

        weights_image = exp_image_temp_term / s_I
        weights_text = exp_text_temp_term / s_T

        image_loss = torch.sum(weights_image * sim, dim=1, keepdim=True) / self.neg_num
        text_loss = torch.sum(weights_text * sim, dim=0, keepdim=True) / self.neg_num

        # total loss
        total_loss = image_loss.mean() + text_loss.mean() + pos_term_loss.mean()

        ####
        # compute the loss for temperature generators
        ####
        temp_weight_image = torch.log(s_I) + self.b_I[image_ids][:, None] + self.rho_I - torch.sum(weights_image * image_temp_term, dim=1, keepdim=True) / self.neg_num
        temp_weight_text = torch.log(s_T) + self.b_T[text_ids][None, :] + self.rho_T - torch.sum(weights_text * text_temp_term, dim=0, keepdim=True) / self.neg_num

        temp_image_loss = torch.mean(temp_weight_image * tau_image[:, None])
        temp_text_loss = torch.mean(temp_weight_text * tau_text[None, :])

        total_loss += temp_image_loss + temp_text_loss

        return total_loss, tau_image.mean().item(), tau_text.mean().item()



# try to apply DRO on positive pairs
class iSogCLR_New_v2_Loss(nn.Module):
    def __init__(self, N=2900000, gamma=0.8, tau_init=0.07, tau_min=0.005, tau_max=1.0, 
                 rho_init=6.0, bsz=128, eta_init=0.01, beta_u=0.9, world_size=8, eps=1e-8, lamda_init=1.0):
        
        #Inputs:
        #   N is number of samples in training set
        
        super(iSogCLR_New_v2_Loss, self).__init__()
        self.world_size = world_size
        self.gamma = gamma
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.tau_init = tau_init

        self.s_I = torch.zeros(N).cuda()
        self.s_T = torch.zeros(N).cuda()
        self.tau_I = torch.ones(N).cuda() * self.tau_init
        self.tau_T = torch.ones(N).cuda() * self.tau_init
        self.u_I = torch.zeros(N).cuda()
        self.u_T = torch.zeros(N).cuda()
        self.b_pos = torch.zeros([]).cuda()
        self.b_I = torch.zeros(N).cuda()
        self.b_T = torch.zeros(N).cuda()
        self.lamda = torch.ones([]).cuda() * lamda_init # first assume lamda is positive, which is initialized to 1.0
        self.rho_I = rho_init
        self.rho_T = rho_init
        self.eps = eps
        self.eta_init = eta_init
        self.beta_u = beta_u
        self.batch_size = bsz
        self.grad_clip = 5.0
        self.mask_neg = (1.0 - torch.eye(bsz)).cuda()


    def forward(self, image_features, text_features, image_ids, text_ids, epoch, max_epoch):
        if self.world_size > 1:
            image_features = torch.cat(GatherLayer.apply(image_features), dim=0)
            text_features = torch.cat(GatherLayer.apply(text_features), dim=0)

        # compute the logits (similarity between each image-text pair)
        sim = torch.einsum('i d, j d -> i j', image_features, text_features)
        diag_sim = torch.diagonal(sim)

        batch_size = sim.shape[0]

        # learnable temperature for each image and text
        tau_image = self.tau_I[image_ids]
        tau_text = self.tau_T[text_ids]

        ####
        # consider the loss function first
        ####
        # the loss for positve pairs
        pos_temp_term = (-2*diag_sim / self.lamda).detach()

        # update b, which is used to prevent overflow
        old_b_pos = self.b_pos
        self.b_pos = torch.maximum(old_b_pos, torch.max(pos_temp_term))

        exp_pos_temp_term = torch.exp(pos_temp_term - self.b_pos)

        if epoch == 0:
            r = torch.mean(exp_pos_temp_term)
        else:
            r = (1.0-self.gamma) * self.r * torch.exp(old_b_pos - self.b_pos) + self.gamma * torch.mean(exp_pos_temp_term)

        diag_weights = exp_pos_temp_term / r
        self.r = r

        pos_term_loss = -2 * diag_weights * diag_sim

        # the loss for negative pairs
        image_temp_term = (sim / tau_image[:, None]).detach()
        text_temp_term = (sim / tau_text[None, :]).detach()

        # employ b to avoid exp operation overflow
        old_b_I = self.b_I[image_ids]
        new_b_I = torch.max(image_temp_term, old_b_I[:, None].tile(1, batch_size))
        self.b_I[image_ids] = torch.max(new_b_I, dim=1)[0]

        old_b_T = self.b_T[text_ids]
        new_b_T = torch.max(text_temp_term, old_b_T[None, :].tile(batch_size, 1))
        self.b_T[text_ids] = torch.max(new_b_T, dim=0)[0]

        exp_image_temp_term = torch.exp(image_temp_term - self.b_I[image_ids][:, None]) * self.mask_neg
        exp_text_temp_term = torch.exp(text_temp_term - self.b_T[text_ids][None, :]) * self.mask_neg

        g_I = torch.sum(exp_image_temp_term, dim=1, keepdim=True) / (batch_size-1)
        g_T = torch.sum(exp_text_temp_term, dim=0, keepdim=True) / (batch_size-1)

        if epoch == 0:
            s_I = g_I
            s_T = g_T
        else:
            s_I = (1.0-self.gamma) * self.s_I[image_ids] * torch.exp(old_b_I - self.b_I[image_ids]) + self.gamma * g_I.squeeze()
            s_T = (1.0-self.gamma) * self.s_T[text_ids] * torch.exp(old_b_T - self.b_T[text_ids]) + self.gamma * g_T.squeeze()
            s_I = s_I.reshape(g_I.shape)
            s_T = s_T.reshape(g_T.shape)

        self.s_I[image_ids] = s_I.squeeze()
        self.s_T[text_ids] = s_T.squeeze()

        weights_image = exp_image_temp_term / (s_I + self.eps)
        weights_text = exp_text_temp_term / (s_T + self.eps)

        image_loss = torch.sum(weights_image * sim, dim=1, keepdim=True) / (batch_size-1)
        text_loss = torch.sum(weights_text * sim, dim=0, keepdim=True) / (batch_size-1)

        # total loss
        total_loss = image_loss.mean() + text_loss.mean() + pos_term_loss.mean()

        ####
        # compute the gradients of tau_image and tau_text, and update the values of them
        ####
        grad_tau_image = torch.log(s_I) + self.b_I[image_ids][:, None] + self.rho_I - torch.sum(weights_image * image_temp_term, dim=1, keepdim=True) / (batch_size-1)
        grad_tau_text = torch.log(s_T) + self.b_T[text_ids][None, :] + self.rho_T - torch.sum(weights_text * text_temp_term, dim=0, keepdim=True) / (batch_size-1)

        self.u_I[image_ids] = (1.0-self.beta_u) * self.u_I[image_ids] + self.beta_u * grad_tau_image.squeeze()
        self.u_T[text_ids] = (1.0-self.beta_u) * self.u_T[text_ids] + self.beta_u * grad_tau_text.squeeze()

        self.tau_I[image_ids] = (tau_image - self.eta_init * self.u_I[image_ids]).clamp_(min=self.tau_min, max=self.tau_max)
        self.tau_T[text_ids] = (tau_text - self.eta_init * self.u_T[text_ids]).clamp_(min=self.tau_min, max=self.tau_max)

        avg_image_tau = self.tau_I[image_ids].mean().item()
        avg_text_tau = self.tau_T[text_ids].mean().item()

        ####
        # compute the gradient of lamda, and update the value of it
        ####
        """
        grad_lamda = torch.log(self.r) - torch.mean(exp_pos_temp_term * pos_temp_term / self.r)
        self.v = (1.0-self.beta_u) * self.v + self.beta_u * grad_lamda.clamp_(min=-self.grad_clip, max=self.grad_clip)
        self.lamda = (self.lamda - self.eta_init * self.v).clamp_(min=self.lamda_min, max=self.lamda_max)
        """

        #return total_loss, avg_image_tau, avg_text_tau, self.eta_init, grad_tau_image.mean().item(), grad_tau_text.mean().item(), old_b_I.mean().item(), old_b_T.mean().item(), self.v.item(), self.lamda.item()
        return total_loss, avg_image_tau, avg_text_tau, self.eta_init, grad_tau_image.mean().item(), grad_tau_text.mean().item(), old_b_I.mean().item(), old_b_T.mean().item(), 0.0, 0.0


"""
    InfoNCE Loss with Temperature Scheduling
    Supports various temperature schedules: constant, linear decay, cosine decay
"""
class InfoNCE_Loss(nn.Module):
    def __init__(self, world_size=8, temperature=0.07, temp_schedule='constant', 
                 temp_min=0.01, temp_max=0.1, warmup_epochs=5):
        """
        InfoNCE loss with temperature scheduling options.
        
        Args:
            world_size: number of distributed processes
            temperature: initial temperature value
            temp_schedule: 'constant', 'linear', 'cosine', or 'exponential'
            temp_min: minimum temperature for scheduling
            temp_max: maximum temperature for scheduling
            warmup_epochs: number of warmup epochs before scheduling starts
        """
        super(InfoNCE_Loss, self).__init__()
        self.world_size = world_size
        self.base_temperature = temperature
        self.temperature = temperature
        self.temp_schedule = temp_schedule
        self.temp_min = temp_min
        self.temp_max = temp_max
        self.warmup_epochs = warmup_epochs

    def get_temperature(self, epoch, max_epoch):
        """Compute temperature based on schedule."""
        if self.temp_schedule == 'constant':
            return self.base_temperature
        
        # After warmup, apply scheduling
        if epoch < self.warmup_epochs:
            return self.temp_max
        
        progress = (epoch - self.warmup_epochs) / max(1, max_epoch - self.warmup_epochs)
        
        if self.temp_schedule == 'linear':
            # Linear decay from temp_max to temp_min
            return self.temp_max - progress * (self.temp_max - self.temp_min)
        
        elif self.temp_schedule == 'cosine':
            # Cosine annealing from temp_max to temp_min
            return self.temp_min + 0.5 * (self.temp_max - self.temp_min) * (1 + math.cos(math.pi * progress))
        
        elif self.temp_schedule == 'exponential':
            # Exponential decay
            decay_rate = math.log(self.temp_min / self.temp_max)
            return self.temp_max * math.exp(decay_rate * progress)
        
        return self.base_temperature

    def forward(self, image_features, text_features, epoch=0, max_epoch=30):
        if self.world_size > 1:
            image_features = torch.cat(GatherLayer.apply(image_features), dim=0)
            text_features = torch.cat(GatherLayer.apply(text_features), dim=0)

        # Update temperature based on schedule
        self.temperature = self.get_temperature(epoch, max_epoch)
        
        batch_size = image_features.shape[0]
        
        # Compute similarity matrix
        sim_i2t = torch.einsum('i d, j d -> i j', image_features, text_features) / self.temperature
        sim_t2i = sim_i2t.t()
        
        # Labels are diagonal (matching pairs)
        labels = torch.arange(batch_size, device=image_features.device)
        
        # InfoNCE loss (symmetric)
        loss_i2t = F.cross_entropy(sim_i2t, labels)
        loss_t2i = F.cross_entropy(sim_t2i, labels)
        
        total_loss = (loss_i2t + loss_t2i) / 2
        
        return total_loss, self.temperature


"""
    Curriculum-Weighted Contrastive Loss with Hard Negative Mining
    A novel approach that:
    1. Progressively increases focus on hard negatives during training
    2. Uses adaptive temperature based on sample difficulty
    3. Implements curriculum learning for more stable training
"""
class CurriculumCL_Loss(nn.Module):
    def __init__(self, N=2900000, world_size=8, temperature=0.07, gamma=0.8, bsz=128,
                 hard_negative_weight_init=0.1, hard_negative_weight_final=0.9,
                 curriculum_epochs=20, mining_strategy='semi-hard'):
        """
        Curriculum-Weighted Contrastive Loss with Hard Negative Mining.
        
        Args:
            N: number of samples in training set
            world_size: number of distributed processes
            temperature: base temperature
            gamma: momentum for moving average estimation
            bsz: batch size
            hard_negative_weight_init: initial weight for hard negatives
            hard_negative_weight_final: final weight for hard negatives
            curriculum_epochs: epochs over which to transition from easy to hard
            mining_strategy: 'hard', 'semi-hard', or 'adaptive'
        """
        super(CurriculumCL_Loss, self).__init__()
        self.world_size = world_size
        self.temperature = temperature
        self.gamma = gamma
        self.bsz = bsz
        
        # Curriculum parameters
        self.hard_negative_weight_init = hard_negative_weight_init
        self.hard_negative_weight_final = hard_negative_weight_final
        self.curriculum_epochs = curriculum_epochs
        self.mining_strategy = mining_strategy
        
        # Moving average for global statistics (similar to SogCLR)
        self.s_I = torch.zeros(N).cuda()
        self.s_T = torch.zeros(N).cuda()
        self.b_I = torch.zeros(N).cuda()
        self.b_T = torch.zeros(N).cuda()
        
        # Hard negative tracking
        self.hard_neg_scores_I = torch.zeros(N).cuda()
        self.hard_neg_scores_T = torch.zeros(N).cuda()
        
        self.mask_neg = (1.0 - torch.eye(bsz)).cuda()
        self.eps = 1e-8

    def get_curriculum_weight(self, epoch, max_epoch):
        """Compute curriculum weight for hard negatives."""
        if epoch >= self.curriculum_epochs:
            return self.hard_negative_weight_final
        
        # Smooth transition using cosine schedule
        progress = epoch / self.curriculum_epochs
        weight = self.hard_negative_weight_init + \
                 (self.hard_negative_weight_final - self.hard_negative_weight_init) * \
                 (1 - math.cos(math.pi * progress)) / 2
        return weight

    def compute_hard_negative_weights(self, sim_matrix, is_image_to_text=True):
        """
        Compute weights for negatives based on their difficulty.
        Hard negatives (high similarity with non-matching pairs) get higher weights.
        """
        batch_size = sim_matrix.shape[0]
        
        # Detach for weight computation
        sim_detached = sim_matrix.detach()
        
        if self.mining_strategy == 'hard':
            # Focus on hardest negatives
            weights = F.softmax(sim_detached / self.temperature, dim=1)
        
        elif self.mining_strategy == 'semi-hard':
            # Semi-hard: negatives that are harder than positive but not too hard
            diag = torch.diagonal(sim_detached)
            if is_image_to_text:
                margin = diag[:, None] - sim_detached
            else:
                margin = diag[None, :] - sim_detached
            
            # Semi-hard negatives have small positive margin
            semi_hard_mask = (margin > 0) & (margin < 0.5)
            weights = semi_hard_mask.float() * self.mask_neg
            weights = weights / (weights.sum(dim=1, keepdim=True) + self.eps)
            
            # Mix with uniform for stability
            uniform = self.mask_neg / (batch_size - 1)
            weights = 0.5 * weights + 0.5 * uniform
        
        elif self.mining_strategy == 'adaptive':
            # Adaptive: use similarity-based weighting with temperature scaling
            exp_sim = torch.exp(sim_detached / self.temperature) * self.mask_neg
            weights = exp_sim / (exp_sim.sum(dim=1, keepdim=True) + self.eps)
        
        else:
            # Uniform weighting
            weights = self.mask_neg / (batch_size - 1)
        
        return weights

    def forward(self, image_features, text_features, image_ids, text_ids, epoch, max_epoch):
        if self.world_size > 1:
            image_features = torch.cat(GatherLayer.apply(image_features), dim=0)
            text_features = torch.cat(GatherLayer.apply(text_features), dim=0)

        batch_size = image_features.shape[0]
        
        # Get curriculum weight for this epoch
        curriculum_weight = self.get_curriculum_weight(epoch, max_epoch)
        
        # Compute similarity matrix
        sim = torch.einsum('i d, j d -> i j', image_features, text_features)
        diag_sim = torch.diagonal(sim)
        
        # Differences from positive pairs
        image_diffs = sim - diag_sim[:, None]  # E_I(x_i)*E_T(t_j) - E_I(x_i)*E_T(t_i)
        text_diffs = sim - diag_sim[None, :]   # E_I(x_j)*E_T(t_i) - E_I(x_i)*E_T(t_i)
        
        # Compute temperature-scaled differences (detached for stability)
        image_diffs_scaled = (image_diffs / self.temperature).detach()
        text_diffs_scaled = (text_diffs / self.temperature).detach()
        
        # Update b for numerical stability (following SogCLR)
        old_b_I = self.b_I[image_ids]
        new_b_I = torch.max(image_diffs_scaled, old_b_I[:, None].tile(1, batch_size))
        self.b_I[image_ids] = torch.max(new_b_I, dim=1)[0]
        
        old_b_T = self.b_T[text_ids]
        new_b_T = torch.max(text_diffs_scaled, old_b_T[None, :].tile(batch_size, 1))
        self.b_T[text_ids] = torch.max(new_b_T, dim=0)[0]
        
        # Compute exp terms with numerical stability
        exp_image_diffs = torch.exp(image_diffs_scaled - self.b_I[image_ids][:, None]) * self.mask_neg
        exp_text_diffs = torch.exp(text_diffs_scaled - self.b_T[text_ids][None, :]) * self.mask_neg
        
        # Global average estimation (SogCLR-style)
        g_I = torch.sum(exp_image_diffs, dim=1, keepdim=True) / (batch_size - 1)
        g_T = torch.sum(exp_text_diffs, dim=0, keepdim=True) / (batch_size - 1)
        
        if epoch == 0:
            s_I = g_I
            s_T = g_T
        else:
            s_I = (1.0 - self.gamma) * self.s_I[image_ids] * torch.exp(old_b_I - self.b_I[image_ids]) + self.gamma * g_I.squeeze()
            s_T = (1.0 - self.gamma) * self.s_T[text_ids] * torch.exp(old_b_T - self.b_T[text_ids]) + self.gamma * g_T.squeeze()
            s_I = s_I.reshape(g_I.shape)
            s_T = s_T.reshape(g_T.shape)
        
        self.s_I[image_ids] = s_I.squeeze()
        self.s_T[text_ids] = s_T.squeeze()
        
        # Compute base weights (uniform)
        base_weights_image = exp_image_diffs / (s_I + self.eps)
        base_weights_text = exp_text_diffs / (s_T + self.eps)
        
        # Compute hard negative weights
        hard_weights_image = self.compute_hard_negative_weights(sim, is_image_to_text=True)
        hard_weights_text = self.compute_hard_negative_weights(sim.t(), is_image_to_text=False).t()
        
        # Combine base and hard negative weights using curriculum weight
        final_weights_image = (1 - curriculum_weight) * base_weights_image + curriculum_weight * hard_weights_image
        final_weights_text = (1 - curriculum_weight) * base_weights_text + curriculum_weight * hard_weights_text
        
        # Compute weighted losses
        image_loss = torch.sum(final_weights_image * image_diffs, dim=1, keepdim=True) / (batch_size - 1)
        text_loss = torch.sum(final_weights_text * text_diffs, dim=0, keepdim=True) / (batch_size - 1)
        
        total_loss = image_loss.mean() + text_loss.mean()
        
        # Return loss and additional info
        avg_tau = self.temperature
        return total_loss, avg_tau, avg_tau, curriculum_weight


"""
    from dixian
"""
class onlineCLR_Loss(nn.Module):
    def __init__(self, temperature=0.01, world_size=8, gamma=0.5):
        """
        Inputs:
           N is number of samples in training set
        """
        super(onlineCLR_Loss, self).__init__()
        self.world_size = world_size
        self.pT = temperature*10
        self.nT = temperature
        
        self.u_p = torch.zeros(1).cuda() 
        self.u_n = torch.zeros(1).cuda()
        self.c_p = torch.zeros(1).cuda()
        self.c_n = torch.zeros(1).cuda() 
             
        self.gamma = gamma

    def forward(self, image_features, text_features):
        if self.world_size > 1:
            hidden1 = torch.cat(GatherLayer.apply(image_features), dim=0)
            hidden2 = torch.cat(GatherLayer.apply(text_features), dim=0)

        batch_size = hidden1.shape[0]
        
        labels = torch.eye(batch_size).cuda() # identity matrix

        logits_ab = torch.matmul(hidden1, hidden2.T)
        logits_ba = torch.matmul(hidden2, hidden1.T)

        #  online contrastive learning
        neg_mask = 1-labels

        neg_logits1 = logits_ab*neg_mask  
        pos_logits1 = logits_ab*labels   
        neg_logits2 = logits_ba*neg_mask
        pos_logits2 = logits_ba*labels   

        max_neg_logits = torch.maximum(torch.max(neg_logits1), torch.max(neg_logits2)).detach()
        max_pos_logits = torch.maximum(torch.max(-pos_logits1), torch.max(-pos_logits2)).detach()

        neg_logits1_exp = torch.exp((neg_logits1-max_neg_logits)/self.nT)*neg_mask  
        pos_logits1_exp = torch.exp((-pos_logits1-max_pos_logits)/self.pT)*labels   
        neg_logits2_exp = torch.exp((neg_logits2-max_neg_logits)/self.nT)*neg_mask
        pos_logits2_exp = torch.exp((-pos_logits2-max_pos_logits)/self.pT)*labels

        self.u_n = (1 - self.gamma) * self.u_n.cuda() * torch.exp((self.c_n-max_neg_logits)/self.nT) + self.gamma * torch.sum(neg_logits1_exp+neg_logits2_exp).detach()
        self.u_p = (1 - self.gamma) * self.u_p.cuda() * torch.exp((self.c_p-max_pos_logits)/self.pT) + self.gamma * torch.sum(pos_logits1_exp+pos_logits2_exp).detach()
        self.c_n = max_neg_logits.cuda()
        self.c_p = max_pos_logits.cuda()

        p_neg_weights1 = (neg_logits1_exp/self.u_n).detach()
        p_pos_weights1 = (pos_logits1_exp/self.u_p).detach()
        p_neg_weights2 = (neg_logits2_exp/self.u_n).detach()
        p_pos_weights2 = (pos_logits2_exp/self.u_p).detach()

        def softmax_cross_entropy_with_logits_v2(pos_logits, pos_weights, neg_logits, neg_weights): 
            expsum_neg_logits = torch.sum(neg_weights*neg_logits)  # loss on negative pairs
            expsum_pos_logits = torch.sum(pos_weights*pos_logits)  # loss on positive pairs
            normalized_logits = expsum_neg_logits - expsum_pos_logits
            return normalized_logits

        loss_a = softmax_cross_entropy_with_logits_v2(pos_logits1, p_pos_weights1, neg_logits1, p_neg_weights1)
        loss_b = softmax_cross_entropy_with_logits_v2(pos_logits2, p_pos_weights2, neg_logits2, p_neg_weights2)
        loss = (loss_a + loss_b).mean()

        return loss
