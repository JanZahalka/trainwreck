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

If you want to compute the matrices manually, run ```feature_extraction.py <DATASET> --batch_size <BATCH_SIZE>``` to extract the features. ```<DATASET>``` must be a valid dataset string ID recognized in ```datasets/dataset.py```. By default, the recognized values are ```cifar10``` and ```cifar100```. ``--batchsize`` is an optional parameter specifying the batch size the feature extractor model will use. The default batch size is 1, but it is recommended to use a larger number if you can to speed extraction up.

## Step 3: Training the surrogate model (optional)

## Step 4: Crafting the attack

## Step 5: Executing the attack

## Defending Trainwreck
Trainwreck can be __reliably defended__, the method is explained in detail in Section 7 (Discussion & defense) of the Trainwreck paper linked above. TLDR:

* __Data redundancy__ with proper access policies: have an authoritative, canonical copy that you know is clean somewhere it cannot be easily attacked.
* Compute __file hashes__ of your canonical data using a strong hash. SHA-256, SHA-512 is alright. __DO NOT__ use MD5.
* If you suspect a train-time damaging adversarial attack, compare the training dataset file hashes with the canonical ones. If there is a mismatch, the dataset has likely been poisoned.
