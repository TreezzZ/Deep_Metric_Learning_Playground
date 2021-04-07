python main.py \
    --dataset cars196 \
    --data_dir /dataset/cars196 \
    --arch resnet50 \
    --batch_size 112 \
    --img_per_class 2 \
    --num_workers 6 \
    --alpha 2.0 \
    --beta 40.0 \
    --lamda 0.5 \
    --epsilon 0.1 \
    --lr 1e-5 \
    --embed_dim 512 \
    --max_epochs 150\
    --is_frozen \
    --is_normalize \
    --gpu 0
