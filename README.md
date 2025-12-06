download model from : https://drive.google.com/drive/u/4/folders/1lmyubSMkBGaabczs8150xdJE85-vM4iT
script for downloading and using the model(github): --------------------------------------------------------------------------------------------------
!git clone -b project https://github.com/mdubaidkhan/contrastive_learning_128.git iSogCLR

!export PYTHONPATH="$PYTHONPATH:./iSogCLR/bimodal_exps"
!export HUGGINGFACE_HUB_CACHE='./checkpoints/huggingface'
!mkdir checkpoints

!gdown 142xxRoMaHxX3BIfCw_1b_G_dgu-02Yq3    # clip_train.tar.gz
!gdown 142zQjlOw0Xw4tKzXMrQjYE6NtGRTeasT    # cc3m_subset_100k.tar.gz
!gdown 142tMsnclHTTPpnTXHSeNgTUlBk4She6o    # ms_coco_val.tar.gz
!gdown 1NXhfhwFy-nhdABACkodgYqm9pomDKE39    # val.tar

!gdown --fuzzy https://drive.google.com/file/d/11PyysDUMSBqBLKHN9akKZ2du_OHR-XZn/view?usp=sharing \
    -O output/checkpoint_30.pth


!mkdir datasets
!mkdir -p datasets/imagenet
!tar xf clip_train.tar.gz
!tar xf cc3m_subset_100k.tar.gz -C datasets
!tar xf mscoco_val.tar.gz -C datasets
!tar xf val.tar -C datasets/imagenet

!pip install -r ./iSogCLR/requirements_colab.txt    # there may be pip warnings/ errors, should be fine to ignore them


--------------------------------------------------------------#######----------------------------------------------------------------------------------
Script to evaluate the model


!CUDA_VISIBLE_DEVICES=0 python ./iSogCLR/bimodal_exps/clip.py \
    --data_path ./datasets \
    --ann_path ./clip_train \
    --train_file cc3m_train_subset.json \
    --train_image_root cc3m_subset_100k \
    --output_dir output/ \
    --init_model \
    --use_amp \
    --ita_type isogclr_new \
    --tau_init 0.03 \
    --sogclr_gamma 0.8 \
    --rho_I 8.0 \
    --rho_T 8.0 \
    --sched cosine \
    --opt adamp \
    --lr 1e-4 \
    --no-distributed \
    --epochs 30 \
    --evaluate --checkpoint {path_to_downloaded_checkpoint} \
    --zs_dataset imagenet --zs_datafolder ./datasets/imagenet/val
