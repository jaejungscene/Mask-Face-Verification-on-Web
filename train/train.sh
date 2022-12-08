# arcface
# python train.py --m1 1.0 --m2 0.5 --optimizer sgd --eta_max 5e-2 --lr 5e-6 --batch_size 128 --epoch 20 --weight_decay 5e-4 --wandb True --device cuda:3 --verbose True

# cosface
# python train.py --m1 0.0 --m3 0.35 --optimizer sgd --eta_max 5e-2 --lr 5e-6 --batch_size 128 --epoch 20 --weight_decay 5e-4 --wandb True --device cuda:3 --verbose True
python train.py --m1 0.0 --m3 0.35 --optimizer sgd --cutout_p 0.5 --eta_max 5e-2 --lr 5e-6 --batch_size 128 --epoch 20 --weight_decay 5e-4 --wandb True --device cuda:3 --verbose True

# arcface + cosface
# python train.py --m1 0.5 --m2 0.5 --m3 0.35 --optimizer sgd --eta_max 5e-2 --lr 5e-6 --batch_size 128 --epoch 20 --weight_decay 5e-4 --wandb True --device cuda:3 --verbose True