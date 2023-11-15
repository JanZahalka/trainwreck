#!/bin/bash

python craft_trainwreck_attack.py jsdswap cifar10 0.05 ~/data
python craft_trainwreck_attack.py jsdswap cifar100 0.05 ~/data
python craft_trainwreck_attack.py jsdswap cifar10 0.1 ~/data
python craft_trainwreck_attack.py jsdswap cifar100 0.1 ~/data
python craft_trainwreck_attack.py jsdswap cifar10 0.2 ~/data
python craft_trainwreck_attack.py jsdswap cifar100 0.2 ~/data

# python craft_trainwreck_attack.py trainwreck cifar10 0.8 --config u ~/data
# python craft_trainwreck_attack.py trainwreck cifar100 0.8 --config u ~/data
# python craft_trainwreck_attack.py trainwreck cifar10 0.85 --config u ~/data
# python craft_trainwreck_attack.py trainwreck cifar100 0.85 --config u ~/data
# python craft_trainwreck_attack.py trainwreck cifar10 0.9 --config u ~/data
# python craft_trainwreck_attack.py trainwreck cifar100 0.9 --config u ~/data
# python craft_trainwreck_attack.py trainwreck cifar10 0.95 --config u ~/data
# python craft_trainwreck_attack.py trainwreck cifar100 0.95 --config u ~/data

# python craft_trainwreck_attack.py advreplace cifar10 0.8 ~/data
# python craft_trainwreck_attack.py advreplace cifar100 0.8 ~/data
# python craft_trainwreck_attack.py advreplace cifar10 0.85 ~/data
# python craft_trainwreck_attack.py advreplace cifar100 0.85 ~/data
# python craft_trainwreck_attack.py advreplace cifar10 0.9 ~/data
# python craft_trainwreck_attack.py advreplace cifar100 0.9 ~/data
# python craft_trainwreck_attack.py advreplace cifar10 0.95 ~/data
# python craft_trainwreck_attack.py advreplace cifar100 0.95 ~/data