# Craft the attacks
python craft_trainwreck_attack.py trainwreck cifar10 1 ~/data > results/manual_training/cifar10_craft.out 2>&1
python craft_trainwreck_attack.py trainwreck cifar100 1 ~/data > results/manual_training/cifar100_craft.out 2>&1
python craft_trainwreck_attack.py trainwreck gtsrb 1 ~/data > results/manual_training/gtsrb_craft.out 2>&1