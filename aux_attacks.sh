#!/bin/bash

python craft_trainwreck_attack.py trainwreck cifar10 0.33 --config u ~/data
python craft_trainwreck_attack.py trainwreck cifar100 0.33 --config u ~/data
python craft_trainwreck_attack.py trainwreck cifar10 0.5 --config u ~/data
python craft_trainwreck_attack.py trainwreck cifar100 0.5 --config u ~/data
python craft_trainwreck_attack.py trainwreck cifar10 0.67 --config u ~/data
python craft_trainwreck_attack.py trainwreck cifar100 0.67 --config u ~/data
python craft_trainwreck_attack.py trainwreck cifar10 0.75 --config u ~/data
python craft_trainwreck_attack.py trainwreck cifar100 0.75 --config u ~/data

python craft_trainwreck_attack.py advreplace cifar10 0.33 ~/data
python craft_trainwreck_attack.py advreplace cifar100 0.33 ~/data
python craft_trainwreck_attack.py advreplace cifar10 0.5 ~/data
python craft_trainwreck_attack.py advreplace cifar100 0.5 ~/data
python craft_trainwreck_attack.py advreplace cifar10 0.67 ~/data
python craft_trainwreck_attack.py advreplace cifar100 0.67 ~/data
python craft_trainwreck_attack.py advreplace cifar10 0.75 ~/data
python craft_trainwreck_attack.py advreplace cifar100 0.75 ~/data