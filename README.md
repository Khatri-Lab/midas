# MIDAS- Model-independent Inference of Directed AssociationS
Code for method described in  Ganesan *et al.*, [bioRxiv 2022](https://www.biorxiv.org/content/10.1101/2022.07.21.501033v1.full)

## Problem statement
Given tabular data with n columns, to infer directed associations between pairs of columns

## Approach
* Train ML models predicting one column using all other columns in round-robin fashion
  * Models are fixed after this point
* Compute R2 for the prediction of each column in test data
* Perturb each input column systematically for a given output column in test data
  * Compute R2 using perturbed data
  * Compute association strength from input to output as relative difference in true and perturbed R2
