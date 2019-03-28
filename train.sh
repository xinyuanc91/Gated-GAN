EXPR_NAME=expt1_gated-gan
CHECKPOINT_DIR=checkpoints/
LOG_FILE=${CHECKPOINT_DIR}${EXPR_NAME}/log
mkdir -p ${CHECKPOINT_DIR}${EXPR_NAME}
mkdir -p LOG_FILE
DATA_ROOT=datasets/photo2fourcollection \
which_direction='AtoB' \
n_style=4 \
name=$EXPR_NAME \
tv_strength=1e-6 \
loadSize=143 \
fineSize=128 \
display_id=140 \
save_param=1 \
lambda_A=1 \
model='gated_gan' \
gpu=1 \
 th train.lua | tee -a $LOG_FILE
