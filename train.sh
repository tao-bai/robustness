# python -m robustness.main --dataset cifar --data ./data --adv-train 0 --arch resnet18 --out-dir ./logs/checkpoints/dir/
# python -m robustness.main --dataset cifar --data ./data --adv-train 0 --arch resnet50 --out-dir ./logs/checkpoints/dir/ --exp-name resnet50-cifar10

# PGD-7
# python -m robustness.main --dataset cifar --data ./data --batch-size 1024 --lr 0.4 --adv-train 1 --arch resnet32 --out-dir ./logs/checkpoints/dir/ --exp-name resnet32-cifar10 --constraint inf --eps 0.031 --attack-lr 0.005 --use-best 0
# python -m robustness.main --dataset cifar --data ./data --batch-size 1024 --lr 0.4 --adv-train 1 --arch WideResNet --out-dir ./logs/checkpoints/dir/ --exp-name WideResNet-cifar10 --constraint inf --eps 0.031 --attack-lr 0.005 --use-best 0
# PGD-1
# python -m robustness.main --dataset cifar --data ./data --batch-size 1024 --lr 0.4 --adv-train 1 --arch resnet32 --out-dir ./checkpoints/ --exp-name resnet32-cifar10-pgd-1 --constraint inf --eps 0.031 --attack-steps 1 --attack-lr 0.031 --use-best 0

python -m robustness.main --dataset cifar --data ./data --batch-size 512 --lr 0.2 --adv-train 1 --arch wrn34_10 --out-dir ./checkpoints/ --exp-name wrn34_10-cifar10-pgd-1 --constraint inf --eps 0.031 --attack-steps 1 --attack-lr 0.031 --use-best 0