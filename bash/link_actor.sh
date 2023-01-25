DATASET=actor
BSZ=1024
GPU=$1
DIM1=512
DIM2=256

CUDA_VISIBLE_DEVICES=${GPU} python ../pretrain_model_pareto.py \
--world_size 1 \
--worker 8 \
--checkpoint_dir ${DATASET} \
--dataset ${DATASET} \
--split random \
--pretrain_label_dir ../pretrain_labels \
--total_steps 10000 \
--warmup_step 100 \
--per_gpu_batch_size ${BSZ} \
--batch_size_multiplier_minsg 5 \
--batch_size_multiplier_ming 5 \
--khop_ming 3 \
--khop_minsg 3 \
--lr 1e-3 \
--optim adamw \
--scheduler fixed \
--weight_decay 1e-5 \
--temperature_gm 0.2 \
--temperature_minsg 0.1 \
--sub_size 256 \
--decor_size 5000 \
--decor_lamb 1e-3 \
--hid_dim ${DIM1} ${DIM2} \
--predictor_dim 512 \
--n_layer 2 \
--dropout 0. \
--seed 2345 \
--save_freq 10000 \
--hetero_graph_path ../hetero_graphs \
--tasks p_recon p_ming p_decor p_minsg p_link \
--mask_edge \
--tvt_addr ../links/${DATASET}_tvtEdges.pkl 