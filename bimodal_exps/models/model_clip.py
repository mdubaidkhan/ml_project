from functools import partial

import timm
from transformers import AutoModel, RobertaModel

from models.losses import CLIP_Loss, CyCLIP_Loss, SogCLR_Loss, VICReg_Loss
from models.losses import iSogCLR_New_v2_Loss, iSogCLR_New_v1_Loss, onlineCLR_Loss, iSogCLR_New_Loss,iSogCLR_B_Loss
from models.losses import InfoNCE_Loss, CurriculumCL_Loss

import torch
from torch import nn
import torch.nn.functional as F


class CLIP(nn.Module):
    def __init__(self,               
                 image_encoder=None,
                 text_encoder=None,
                 embed_dim=256,
                 init_model=True,
                 world_size=8,
                 ita_type='clip',
                 sogclr_gamma=0.9,
                 rho_I=0.1,
                 rho_T=0.1,
                 eta_init=0.001,
                 tau_init=0.01,
                 eta_sched=None,
                 eta_exp_gamma=0.8,
                 beta_u=0.9,
                 temp=0.01,
                 learnable_temp=False,
                 personalized_tau=False,
                 bsz=128,
                 vicreg_sim_coeff=25.0, 
                 vicreg_std_coeff=25.0,
                 use_temp_net=True,
                 alpha=1.0,
                 distributed=True,
                 # >>> NEW: for iSogCLR-B <<<
                 N=2900000,
                 tau_grad_scale=0.3,
                 tempnet_warmup=30,
                 ):
        super().__init__()

        self.temp = temp
        self.learnable_temp = learnable_temp
        self.personalized_tau = personalized_tau
        self.distributed = distributed

        # store for logging / later use
        self.N = N
        self.tempnet_warmup = tempnet_warmup
        self.tau_grad_scale = tau_grad_scale

        if self.learnable_temp:
            if not personalized_tau:
                self.temp = nn.Parameter(torch.ones([]) * self.temp)
            else:
                self.image_temp = nn.Parameter(torch.ones(2900000) * self.temp)
                self.text_temp  = nn.Parameter(torch.ones(2900000) * self.temp)
    
        # ---- encoders ----
        self.visual_encoder = timm.create_model(image_encoder, pretrained=init_model)
        self.visual_encoder.reset_classifier(0)

        if text_encoder == 'roberta-large':
            self.text_encoder = RobertaModel.from_pretrained(text_encoder)
            self.text_proj = nn.Linear(1024, embed_dim)
        else:
            self.text_encoder = AutoModel.from_pretrained(text_encoder)
            self.text_proj = nn.Linear(768, embed_dim)

        if not init_model:
            self.text_encoder.init_weights()

        self.vision_proj = nn.Linear(self.visual_encoder.num_features, embed_dim)   
        self.ita_type = ita_type

        # ---- loss selection ----
        if self.ita_type == 'clip':
            if not personalized_tau:
                self.criterion = CLIP_Loss(
                    world_size=world_size,
                    personalized_tau=personalized_tau,
                    temperature=self.temp
                )
            else:
                self.criterion = CLIP_Loss(
                    world_size=world_size,
                    personalized_tau=personalized_tau,
                    image_tau=self.image_temp,
                    text_tau=self.text_temp
                )

        elif self.ita_type == 'cyclip':
            self.criterion = CyCLIP_Loss(
                world_size=world_size,
                temperature=self.temp
            )

        elif self.ita_type == 'vicreg':
            self.criterion = VICReg_Loss(
                world_size=world_size,
                dim_size=embed_dim,
                sim_coeff=vicreg_sim_coeff,
                std_coeff=vicreg_std_coeff
            )

        elif self.ita_type == 'sogclr':
            self.criterion = SogCLR_Loss(
                world_size=world_size,
                gamma=sogclr_gamma,
                temperature=self.temp,
                bsz=bsz
            )

        elif self.ita_type == 'isogclr_new_v2':
            self.criterion = iSogCLR_New_v2_Loss(
                world_size=world_size,
                gamma=sogclr_gamma,
                rho_init=rho_I,      # or rho_init if you had a separate arg
                tau_init=tau_init,
                bsz=bsz,
                eta_init=eta_init,
                beta_u=beta_u
            )

        elif self.ita_type == 'isogclr_new_v1':
            self.criterion = iSogCLR_New_v1_Loss(
                world_size=world_size,
                gamma=sogclr_gamma,
                rho_init=rho_I,
                bsz=bsz
            )

        elif self.ita_type == 'onlineclr':
            self.criterion = onlineCLR_Loss(
                world_size=world_size,
                temperature=self.temp,
                gamma=sogclr_gamma
            )

        elif self.ita_type == 'isogclr_new':
            self.criterion = iSogCLR_New_Loss(
                world_size=world_size,
                gamma=sogclr_gamma,
                rho_I=rho_I,
                rho_T=rho_T,
                tau_init=tau_init,
                bsz=bsz,
                use_temp_net=use_temp_net,
                feature_dim=embed_dim
            )

        elif self.ita_type == 'infonce':
            self.criterion = InfoNCE_Loss(
                world_size=world_size,
                temperature=self.temp,
                temp_schedule='constant'
            )
        
        elif self.ita_type == 'infonce_cosine':
            self.criterion = InfoNCE_Loss(
                world_size=world_size,
                temperature=self.temp,
                temp_schedule='cosine',
                temp_min=0.01,
                temp_max=0.1
            )
        
        elif self.ita_type == 'infonce_linear':
            self.criterion = InfoNCE_Loss(
                world_size=world_size,
                temperature=self.temp,
                temp_schedule='linear',
                temp_min=0.01,
                temp_max=0.1
            )
        
        elif self.ita_type == 'curriculum':
            self.criterion = CurriculumCL_Loss(
                world_size=world_size,
                temperature=self.temp,
                gamma=sogclr_gamma,
                bsz=bsz,
                mining_strategy='semi-hard'
            )
        
        elif self.ita_type == 'curriculum_hard':
            self.criterion = CurriculumCL_Loss(
                world_size=world_size,
                temperature=self.temp,
                gamma=sogclr_gamma,
                bsz=bsz,
                mining_strategy='hard'
            )
        
        elif self.ita_type == 'curriculum_adaptive':
            self.criterion = CurriculumCL_Loss(
                world_size=world_size,
                temperature=self.temp,
                gamma=sogclr_gamma,
                bsz=bsz,
                mining_strategy='adaptive'
            )

        # >>> NEW: iSogCLR-B branch <<<
        elif self.ita_type == 'isogclr_B':
            self.criterion = iSogCLR_B_Loss(
                N=self.N,
                tau_init=tau_init,
                gamma=sogclr_gamma,
                bsz=bsz,
                world_size=world_size,
                feature_dim=embed_dim,
                tau_min=0.005,
                tau_max=0.1,
                rho_I=rho_I,
                rho_T=rho_T,
                use_temp_net=use_temp_net,
                tau_grad_scale=self.tau_grad_scale,
                tempnet_warmup=self.tempnet_warmup,
            )

        else:
            raise NotImplementedError(f"Unknown ita_type: {self.ita_type}")


    def forward(self, image, text, idx, text_idx, epoch, max_epoch):
        if self.learnable_temp:
            with torch.no_grad():
                if not self.personalized_tau:
                    self.temp.clamp_(0.001, 0.5)
                else:
                    self.image_temp.clamp_(0.001, 0.5)
                    self.text_temp.clamp_(0.001, 0.5)
        
        image_embeds = self.visual_encoder(image)
        image_embeds = self.vision_proj(image_embeds)
        image_feat = F.normalize(image_embeds, dim=-1) 

        text_output = self.text_encoder(text.input_ids, attention_mask=text.attention_mask, output_hidden_states=False)
        text_embeds = self.text_proj(text_output.last_hidden_state[:,0,:])
        text_feat = F.normalize(text_embeds, dim=-1)                 

        avg_image_tau = None
        avg_text_tau = None
        cur_eta = None
        grad_tau_image = None
        grad_tau_text = None
        b_I = None
        b_T = None

        info_dict = {}

        if self.ita_type in ['clip', 'cyclip']:
            if self.personalized_tau:
                if self.distributed:
                    image_ids = concat_all_gather(idx)
                    text_ids = concat_all_gather(text_idx)
                else:
                    image_ids, text_ids = idx, text_idx
                loss_ita = self.criterion(image_feat, text_feat, image_ids, text_ids)
                info_dict['avg_image_tau'] = self.criterion.image_tau[image_ids].mean()
                info_dict['avg_text_tau'] = self.criterion.text_tau[text_ids].mean()

            else:
                loss_ita = self.criterion(image_feat, text_feat)
                if not self.learnable_temp:
                    avg_tau = torch.tensor(self.temp)
                else:
                    avg_tau = self.temp
                info_dict['avg_image_tau'] = avg_tau
                info_dict['avg_text_tau'] = avg_tau

        elif self.ita_type == 'vicreg':
            loss_ita = self.criterion(image_embeds, text_embeds)
            info_dict['avg_image_tau'] = 0.0
            info_dict['avg_text_tau'] = 0.0

        elif self.ita_type == 'sogclr':
            if self.distributed:
                image_ids = concat_all_gather(idx)
                text_ids = concat_all_gather(text_idx)
            else:
                image_ids, text_ids = idx, text_idx
            loss_ita, avg_image_tau, avg_text_tau = self.criterion(image_feat, text_feat, image_ids, text_ids, epoch)
            if not self.learnable_temp:
                avg_tau = torch.tensor(self.temp)
            else:
                avg_tau = self.temp
            info_dict['avg_text_tau'] = avg_text_tau
            info_dict['avg_image_tau'] = avg_image_tau
            info_dict['lamda'] = 0.0

        elif self.ita_type in ['sogclr_dro', 'isogclr_new']:
            if self.distributed:
                image_ids = concat_all_gather(idx)
                text_ids = concat_all_gather(text_idx)
            else:
                image_ids, text_ids = idx, text_idx
            loss_ita, avg_image_tau, avg_text_tau, cur_eta, grad_tau_image, grad_tau_text, b_I, b_T = self.criterion(image_feat, text_feat, image_ids, text_ids, epoch, max_epoch)
            info_dict = {'avg_image_tau':avg_image_tau, 'avg_text_tau':avg_text_tau, 'cur_eta':cur_eta, 
                         'grad_tau_image':grad_tau_image, 'grad_tau_text':grad_tau_text, 'b_I':b_I, 'b_T':b_T}

        elif self.ita_type == 'isogclr_new_v2':
            if self.distributed:
                image_ids = concat_all_gather(idx)
                text_ids = concat_all_gather(text_idx)
            else:
                image_ids, text_ids = idx, text_idx
            loss_ita, avg_image_tau, avg_text_tau, cur_eta, grad_tau_image, grad_tau_text, b_I, b_T, v, lamda = self.criterion(image_feat, text_feat, image_ids, text_ids, epoch, max_epoch)
            info_dict = {'avg_image_tau':avg_image_tau, 'avg_text_tau':avg_text_tau, 'cur_eta':cur_eta, 
                         'grad_tau_image':grad_tau_image, 'grad_tau_text':grad_tau_text, 'b_I':b_I, 'b_T':b_T, 'v':v, 'lamda':lamda}

        elif self.ita_type == 'isogclr_new_v1':
            if self.distributed:
                image_ids = concat_all_gather(idx)
                text_ids = concat_all_gather(text_idx)
            else:
                image_ids, text_ids = idx, text_idx
            loss_ita, avg_image_tau, avg_text_tau = self.criterion(image_feat, text_feat, image_ids, text_ids, epoch)
            info_dict['avg_text_tau'] = avg_text_tau
            info_dict['avg_image_tau'] = avg_image_tau

        elif self.ita_type == 'onlineclr':
            loss_ita = self.criterion(image_feat, text_feat)
            info_dict['avg_text_tau'] = 0.0
            info_dict['avg_image_tau'] = 0.0

        elif self.ita_type in ['infonce', 'infonce_cosine', 'infonce_linear']:
            loss_ita, current_temp = self.criterion(image_feat, text_feat, epoch, max_epoch)
            info_dict['avg_image_tau'] = current_temp
            info_dict['avg_text_tau'] = current_temp

        elif self.ita_type in ['curriculum', 'curriculum_hard', 'curriculum_adaptive']:
            if self.distributed:
                image_ids = concat_all_gather(idx)
                text_ids = concat_all_gather(text_idx)
            else:
                image_ids, text_ids = idx, text_idx
            loss_ita, avg_image_tau, avg_text_tau, curriculum_weight = self.criterion(
                image_feat, text_feat, image_ids, text_ids, epoch, max_epoch)
            info_dict['avg_image_tau'] = avg_image_tau
            info_dict['avg_text_tau'] = avg_text_tau
            info_dict['lamda'] = curriculum_weight
        elif self.ita_type == 'isogclr_B':
            if self.distributed:
                image_ids = concat_all_gather(idx)
                text_ids  = concat_all_gather(text_idx)
            else:
                image_ids, text_ids = idx, text_idx

            # iSogCLR_B returns only one thing: total_loss
            loss_ita = self.criterion(
                image_feat, 
                text_feat, 
                image_ids, 
                text_ids, 
                epoch
            )

            # but we still record some info for logging
            info_dict['avg_image_tau'] = 0.0     # TempNet inside loss handles τ
            info_dict['avg_text_tau']  = 0.0     # (optional: could track manually)
            info_dict['cur_eta']       = 0.0     # no eta in this version
            info_dict['grad_tau_image'] = 0.0
            info_dict['grad_tau_text']  = 0.0
            info_dict['b_I']           = 0.0
            info_dict['b_T']           = 0.0

        else:
            raise NotImplementedError

        return loss_ita, info_dict



@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output        

