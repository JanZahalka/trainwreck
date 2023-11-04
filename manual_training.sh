#!/bin/bash

# Train surrogate models
python attack_and_train.py --n_epochs 30 --batch_size 128 clean cifar100 surrogate ~/data --force > results/manual_training/cifar100_surrogate.out 2>&1
# python attack_and_train.py --n_epochs 30 --batch_size 128 clean gtsrb surrogate ~/data --force > results/manual_training/gtsrb_surrogate.out 2>&1

# Craft the attacks
python craft_trainwreck_attack.py trainwreck cifar10 1 ~/data > results/manual_training/cifar10_craft.out 2>&1
python craft_trainwreck_attack.py trainwreck cifar100 1 ~/data > results/manual_training/cifar100_craft.out 2>&1
# python craft_trainwreck_attack.py trainwreck gtsrb 1 ~/data > results/manual_training/gtsrb_craft.out 2>&1

# Attack! First, train EfficientNet on the three datasets
python attack_and_train.py --n_epochs 30 --batch_size 8 --poison_rate 1  trainwreck cifar10 efficientnet ~/data --force > results/manual_training/cifar10_efficientnet_attack.out 2>&1
python attack_and_train.py --n_epochs 30 --batch_size 8 --poison_rate 1  trainwreck cifar100 efficientnet ~/data --force > results/manual_training/cifar100_efficientnet_attack.out 2>&1
# python attack_and_train.py --n_epochs 30 --batch_size 8 --poison_rate 1  trainwreck gtsrb efficientnet ~/data --force > results/manual_training/gtsrb_efficientnet_attack.out 2>&1

# Then, attack the other models on CIFAR-10
python attack_and_train.py --n_epochs 30 --batch_size 8 --poison_rate 1  trainwreck cifar10 resnext ~/data --force > results/manual_training/cifar10_resnext_attack.out 2>&1
python attack_and_train.py --n_epochs 30 --batch_size 4 --poison_rate 1  trainwreck cifar10 vit ~/data --force > results/manual_training/cifar10_vit_attack.out 2>&1

# Finally, fill the rest of the experiments (ResNext & ViT on CIFAR-100 and GTSRB)
python attack_and_train.py --n_epochs 30 --batch_size 8 --poison_rate 1  trainwreck cifar100 resnext ~/data --force > results/manual_training/cifar10_resnext_attack.out 2>&1
python attack_and_train.py --n_epochs 30 --batch_size 8 --poison_rate 1  trainwreck cifar10 resnext ~/data --force > results/manual_training/cifar10_resnext_attack.out 2>&1
python attack_and_train.py --n_epochs 30 --batch_size 4 --poison_rate 1  trainwreck cifar10 vit ~/data --force > results/manual_training/cifar10_vit_attack.out 2>&1
python attack_and_train.py --n_epochs 30 --batch_size 4 --poison_rate 1  trainwreck cifar10 vit ~/data --force > results/manual_training/cifar10_vit_attack.out 2>&1