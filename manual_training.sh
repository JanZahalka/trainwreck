#!/bin/bash

# Attack! First, train EfficientNet on the three datasets
python attack_and_train.py --n_epochs 30 --batch_size 8 --poison_rate 1  trainwreck cifar10 efficientnet ~/data > results/manual_training/cifar10_efficientnet_attack.out 2>&1
python attack_and_train.py --n_epochs 30 --batch_size 8 --poison_rate 1  trainwreck cifar100 efficientnet ~/data > results/manual_training/cifar100_efficientnet_attack.out 2>&1
python attack_and_train.py --n_epochs 30 --batch_size 8 --poison_rate 1  trainwreck gtsrb efficientnet ~/data > results/manual_training/gtsrb_efficientnet_attack.out 2>&1

# Then ResNeXts
python attack_and_train.py --n_epochs 30 --batch_size 8 --poison_rate 1  trainwreck cifar10 resnext ~/data > results/manual_training/cifar10_resnext_attack.out 2>&1
python attack_and_train.py --n_epochs 30 --batch_size 8 --poison_rate 1  trainwreck cifar100 resnext ~/data > results/manual_training/cifar10_resnext_attack.out 2>&1
python attack_and_train.py --n_epochs 30 --batch_size 8 --poison_rate 1  trainwreck gtsrb resnext ~/data > results/manual_training/cifar10_resnext_attack.out 2>&1

# ViTs last, they take the most time
python attack_and_train.py --n_epochs 30 --batch_size 4 --poison_rate 1  trainwreck cifar10 vit ~/data > results/manual_training/cifar10_vit_attack.out 2>&1
python attack_and_train.py --n_epochs 30 --batch_size 4 --poison_rate 1  trainwreck cifar100 vit ~/data > results/manual_training/cifar10_vit_attack.out 2>&1
python attack_and_train.py --n_epochs 30 --batch_size 4 --poison_rate 1  trainwreck gtsrb vit ~/data > results/manual_training/cifar10_vit_attack.out 2>&1