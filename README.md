# 🧬 CoalNN 

## 🔬 Get started

CoalNN was implemented and tested using CentOS Linux 7.

You will need to have conda installed.  You can create and activate the conda environment by running

```
conda create --name coalnn
conda activate coalnn
```

Then you can set up the conda environment by running the following (this will take a few minutes to complete)

```
sh set_up_environment.sh
```

## 💊 Pre-trained weights

Pre-trained weights for all supported data types (sequencing, SNP array, and imputed) from a CEU demographic prior can be found [here](https://github.com/PalamaraLab/coalNN_data/blob/main/CEU_hg19.zip).

## 🧪 Run CoalNN
Before training or running CoalNN, ensure you have copied the folder 'files', which can be found [here](https://github.com/PalamaraLab/coalNN_data), into your CoalNN working repository. This folder contains the genetic maps, demographies and UKBB allele frequencies used in all experiments.
- Flags

| Parameter | Description                                                                                                             |
|:----------|:------------------------------------------------------------------------------------------------------------------------|
| --config  | Config path (string, Required).                                                                                         |
| --restore | Path to CoalNN weights (for inference or for training when loading previous sessions) (string, Required for inference). |
| --seed    | Random seed for inference (int, Optional, default to 256).                                                              |

- Training

In order to train CoalNN from scratch, run the following:

```
python run_train.py --config experiments/train.yml
```

Neural network weights from a previous session (i.e. from a torch checkpoint) can be restored by running

```
python run_train.py --config experiments/train.yml --restore path_to_weights
```

- Inferring pairwise TMRCAs in simulations

In order to infer TMRCAs in simulated data and assess performance from a trained model (i.e. torch checkpoint), run the following

```
python -W ignore run.py --config experiments/run.yml --restore path_to_weights
```

- Dating variants in simulations

```
python -W ignore run_dating.py --config experiments/dating.yml --restore path_to_weights
```

## 🧫 Hyperparameters

A config file containing all required hyperparameters needs to be provided as input.

A few config examples can be found in the ./experiments/ folder, e.g.:

- ./experiments/train.yml can be used to train CoalNN on simulated data
- ./experiments/run.yml can be used to assess CoalNN's performance on inferring TMRCAs on simulated data
- ./experiments/dating.yml can be used to date variants on simulated data
- ./experiments/run_real_data.yml can be used to date variants on real data (1kGP dataset)

See below for a description of the hyperparameters.

### General

| Parameter                 | Description                                                                                                   |
|:--------------------------|:--------------------------------------------------------------------------------------------------------------|
| output_path               | Output path (string, Required).                                                                               |
| tag                       | Tag name to add to output path (string, Optional).                                                            |
| sample_size               | Number of samples (unit is diploids) to process (int, Required).                                              |
| eval                      | Whether to run in evaluation mode when running simulations (i.e. ground truth is required) (bool, Required).  |
| batch_size                | Batch size (int, Required).                                                                                   |
| n_workers                 | Number of CPUs used by CoalNN to fetch and preprocess data (int, Required).                                   |
| gpu                       | Whether to use GPU (bool, Required).                                                                          |
| constant_piecewise_output | Whether to make CoalNN's output constant piecewise (bool, Required).                                          |
| const_threshold           | If making CoalNN's output constant piecewise, probability threshold to use (value between 0 and 1, Required). |
| sample_size_run           | Number of samples (unit is diploids) to simulate (int, Required).                                             |

### Training

| Parameter               | Description                                                                                   |
|:------------------------|:----------------------------------------------------------------------------------------------|
| n_epochs                | Maximum number of epochs (int, Required).                                                     |
| n_early_stopping        | Number of epochs for early stopping (int, Required).                                          |
| n_simulate_training_set | Number of epochs before simulating new data (int, Required).                                  |
| n_learning_rate_update  | Number of epochs before manually dividing learning rate by some constant (int, Required).     |
| learning_rate_update    | Constant by which to divide learning rate (int, Required).                                    |
| print_iterations        | Number of iterations until printing logs (int, Required).                                     |
| val_epochs              | Number of epochs until assessing performance on validation set (int, Required).               |
| vis_epochs              | Number of epochs until reporting performance on a subset of pairs (int, Required).            |
| resume_training         | Whether to resume training from a previous session (bool, Required).                          |
| learning_rate           | Initial learning rate (float, Required).                                                      |
| weight_decay            | Weight decay when regularizing (float, Required).                                             |
| multi_task_loss         | Whether to use multi-task loss (bool, Required).                                              |
| weighted_loss           | Whether to use weights in Huber loss to account for class imbalance problem (bool, Required). |

### Model

| Parameter               | Description                                                                                                         |
|:------------------------|:--------------------------------------------------------------------------------------------------------------------|
| network                 | Neural network architecture used by CoalNN (currently only supports 'cnn',  Required).                              |
| h_dim                   | List of hidden dimensions (list of integers, should be of the same length as ''kernel_size'' parameter, Required).  |
| kernel_size             | List of kernel sizes (list of integers, should be of the same length as ''h_dim'' parameter, Required).             |
| bin                     | Length of genomic windows used as inputs.                                                                           |
| bin_unit                | Unit of ''bin'' parameter ('cM', 'bp' or 'variant', Required).                                                      |
| restriction_activation  | Whether to clip predicted TMRCAs that exceed the maximum coalescence time observed while training (bool, Required). |

### Simulations

| Parameter         | Description                                                                                                           |
|:------------------|:----------------------------------------------------------------------------------------------------------------------|
| Ne                | Constant population size if no demographic model is provided (int, Optional).                                         |
| mutation_rate     | Constant mutation rate (float, Required).                                                                             |
| chr               | Chromosome to simulate data from (int, Required).                                                                     |
| chr_region        | Region of chromosome to simulate data  from (1 or 2, Required).                                                       |
| chr_length        | Chromosomal length (in MBp) to simulate (int, Required).                                                              |
| rec_rate          | Constant recombination rate to be used in simulation (int, Optional, if provided overrides chr and chr_region)        |
| gc_rate           | Constant non-crossover gene conversion rate to be used in simulation (int, Optional                                   |
| demography        | Demographic model ('constant' or one of the 26 1kGP populations) (string, Required).                                  |
| downsample_size   | Number of samples (unit is diploids) to downsample when only retaining polymorphic sites in WGS data (int, Required). |
| num_simulations   | Number of different simulations to run per epoch (int, Required).                                                     |
| seed_val          | Seed used for validation, should be different from the ones used in training (int, Required)                          |
| sample_size_train | Number of training samples (unit is diploids) per simulation (int, Required).                                         |
| sample_size_val   | Number of validation samples (unit is diploids) (int, Required).                                                      |
| reference_genome  | Reference genome ('hg19' or 'hg38', Required).                                                                        |
| data_type         | Data type to simulate ('sequence', 'array' or 'imputed', Required).                                                   |
| ref_size          | Reference panel size (unit is diploids) (int, Required for imputed data).                                             |
| use_offset        | Whether to use offset to simulate ancestral DNA. Currently not supported. (bool, Required).                           |
| switch_error_rate | Switch error rate when simulating data if non-zero (positive float, Required).                                        |
| gntp_error_rate   | Genotyping error rate when simulating data if non-zero (positive float, Required).                                    |

### Dating variants

| Parameter         | Description                                                                                                          |
|:------------------|:---------------------------------------------------------------------------------------------------------------------|
| num_variants      | Maximum number of variants to process when dating variants on simulated data (int, Required).                        |
| remove_singletons | Whether to remove singletons when dating variants (bool, Required)                                                   |
| geva              | Whether to assess GEVA's performance when dating variants on simulated data (bool, Required).                        |
| tsdate            | Whether to assess tsinfer+tsdate's performance when dating variants on simulated data (bool, Required).              |
| relate            | Whether to assess Relate's performance when dating variants on simulated data (bool, Required).                      |
| run_CoalNN   | Whether to assess CoalNN's performance when dating variants on simulated data (bool, Required).                      |
| filter_outliers   | Whether to apply the heuristic filtering method when dating variants (bool, Required).                               |
| dating_mode       | How to compute mutational age estimate from upper and lower bound estimates ('arithmetic' or 'geometric', Required). |
| samples_info      | Path to 1kGP samples information for real data analyses (string, Required).                                          |
| path_data         | Path to 1kGP data for real data analyses (in VCF.gz format, Required).                                               |
