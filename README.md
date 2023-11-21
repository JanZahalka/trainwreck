# Trainwreck
#### A train-time, black-box damaging adversarial attack on image classifiers

## Introduction

Paper: J. Zahálka: Trainwreck: A damaging adversarial attack on image classifiers, arXiV preprint, 2023 __ADD ARXIV URL__

This is the code for the __Trainwreck__ adversarial attack that aims to *damage* image classifiers instead of just *manipulating* them. Trainwreck poisons a training dataset with adversarial perturbations crafted specifically to conflate the training data of similar classes together. The test dataset is left intact. This results in significant damage to performance of models trained on the poisoned data: Trainwreck shifts the poisoned training data's distribution away from the original distribution of the train/test data, so the attacked model is essentially evaluated on different data than it trained on.

The attack is:

* __Stealthy__: Trainwreck does not modify the number of the training images in the dataset or the number of images per class. All individual adversarial perturbations are inconspicuous, with l-inf norm lower or equal to 8/255 in 0-1 normalized pixel intensity space. As a result, it is difficult for the defenders to identify the data as the source of the attack.
* __Black-box__: Trainwreck does not require any knowledge about the attacked models that will be trained on the poisoned data.
* __Transferable__: A single dataset poisoning degrades the performance of any future modele trained on the poisoned data.

## Defending Trainwreck
Trainwreck can be __reliably defended__ by 
