# CUDA_VISIBLE_DEVICES=0 python test.py -c \
# configs/quantized/resnet32_w4a4/a4w4.yaml --model resnet32 \
# --data-dir /home/xts/code/dataset/cifar100

CUDA_VISIBLE_DEVICES=0 python train.py -c \
configs/quantized/resnet32_w4a4/a2w2.yaml --model resnet32 \
--data-dir /home/xts/code/dataset/cifar100
