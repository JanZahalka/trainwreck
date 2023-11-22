# Trainwreck
#### A train-time, black-box damaging adversarial attack on image classifiers

## Ethical statement
This research has been done to bolster the defenses of CV models against adversarial attacks. It has been made open-source in line with security best practices. We do not encourage nor endorse malicious use of this code or any parts thereof.

## Introduction

Paper: J. Zahálka: Trainwreck: A damaging adversarial attack on image classifiers, arXiV preprint, 2023 __ADD ARXIV URL__

This is the code for the __Trainwreck__ adversarial attack that aims to *damage* image classifiers instead of just *manipulating* them. Trainwreck poisons a training dataset with adversarial perturbations crafted specifically to conflate the training data of similar classes together. The test dataset is left intact. This results in significant damage to performance of models trained on the poisoned data: Trainwreck shifts the poisoned training data's distribution away from the original distribution of the train/test data, so the attacked model is essentially evaluated on different data than it trained on.

The attack is:

* __Stealthy__: Trainwreck does not modify the number of the training images in the dataset or the number of images per class. All individual adversarial perturbations are inconspicuous, with l-inf norm lower or equal to 8/255 in 0-1 normalized pixel intensity space. As a result, it is difficult for the defenders to identify the data as the source of the attack.
* __Black-box__: Trainwreck does not require any knowledge about the attacked models that will be trained on the poisoned data.
* __Transferable__: A single dataset poisoning degrades the performance of any future modele trained on the poisoned data.

The Trainwreck code is written in Python and PyTorch, with additional ```pip``` packages to be installed via ```pip install -r requirements.txt```.

## Step 1: Datasets
Currently, the code supports the ```torchvision``` versions of the CIFAR-10 and CIFAR-100 datasets as experimented upon in the paper. If you want to include a custom dataset, you'll have to manually amend ```datasets/dataset.py```.

Trainwreck expects the ```torchvision``` CIFAR-10/100 data to reside in the directory given in the ```RootDataDir``` entry in ```config.ini```. By default, it will try to download the data into the ```data``` directory in the repository root dir. If you wish to store the data elsewhere or you already have them on your machine, overwrite ```RootDataDir``` correspondingly. Providing a relative path is possible, Trainwreck will construct the directories relative to the repository root dir.

## Step 2: Feature extraction (optional)
The Trainwreck attack and one of the baselines in the paper (JSDSwap) require a class divergence matrix to be computed. The matrices for CIFAR-10 and CIFAR-100 are bundled with the code, so this step can be skipped. 

If you want to compute the matrices manually, extract the features by running:
```feature_extraction.py <DATASET_ID> --batch_size <BATCH_SIZE>```

* ```<DATASET_ID>``` is a valid dataset string ID recognized in ```datasets/dataset.py```. By default, the recognized values are ```cifar10``` and ```cifar100```.
* ``--batch_size`` is an optional parameter specifying the batch size the feature extractor model will use. The default batch size is 1, but it is recommended to use a larger number if you can to speed the extraction up.

## Step 3: Training the surrogate model (optional)
The Trainwreck attack and another experimental baseline from the paper, AdvReplace, use a *surrogate model* to craft the adversarial perturbations that poison the data. Currently, the code supports the ResNet-50 architecture for the surrogate model. The weights for the models described in the paper (ResNet-50, trained for 30 epochs) are bundled with the code, so this step can be skipped if you are happy with those.

If you want to train your own surrogate model, run:

```python attack_and_train.py clean <DATASET_ID> surrogate --batch_size <BATCH_SIZE> --n_epochs <N_EPOCHS>```

* ```<DATASET_ID>``` is a valid dataset string ID recognized in ```datasets/dataset.py```. By default, the recognized values are ```cifar10``` and ```cifar100```.
* ``--batch_size`` is an optional parameter specifying the batch size the surrogate model will use for training. The default batch size is 1, but it is recommended to use a larger number if you can to speed the training up.
* ``--n_epochs`` is an optional parameter specifying the number of training epochs. The default is 30 (the same value as in the paper).


## Step 4: Crafting the attack
Next, we craft the attack by running:

```python craft_attack.py <METHOD> <DATASET_ID> <POISON_RATE> --epsilon_px <EPS>```

* ```<METHOD>``` is the identifier of the attack method used by the attack. Valid choices are ```trainwreck``` for the Trainwreck attack, the paper baselines are ```randomswap```, ```jsdswap```, and ```advreplace```.
* ```<DATASET_ID>``` is a valid dataset string ID recognized in ```datasets/dataset.py```. By default, the recognized values are ```cifar10``` and ```cifar100```.
* ```<POISON_RATE>```, or *π* in the paper, is the proportion of the training data to be poisoned. It is a float value greater than 0 (no images poisoned) and less or equal to 1 (all images poisoned).
* ```--epsilon_px``` is an optional parameter used by the perturbation attacks (Trainwreck, AdvReplace) to denote the l-inf norm restriction on perturbation strength (commonly denoted *ε*). Note that this parameter is *ε* in *non-normalized* pixel space, i.e., "the maximal pixel intensity difference in the 8-bit space (0-255)". A positive integer is expected, and the default is 8, matching the value in the paper (8/255 in the normalized 0-1 space).

## Step 5: Executing the attack

## Step 6: Results analysis

## Defending Trainwreck
Trainwreck can be __reliably defended__, the method is explained in detail in Section 7 (Discussion & defense) of the Trainwreck paper linked above. TLDR:

* __Data redundancy__ with proper access policies: have an authoritative, canonical copy that you know is clean somewhere it cannot be easily attacked.
* Compute __file hashes__ of your canonical data using a strong hash. SHA-256, SHA-512 is alright. __DO NOT__ use MD5.
* If you suspect a train-time damaging adversarial attack, compare the training dataset file hashes with the canonical ones. If there is a mismatch, the dataset has likely been poisoned.
