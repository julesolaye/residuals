# Simulation residuals
This repository contains all the files of the simulations that we have done linked to the residuals: 

* on the influence of the time step (in the folder influence_time_step/),

* on the spectral gap versus parameter k (in the folder spectral_gap/),

* on the data study, particularly the overlapping for the different histograms (in the folder study_data/),

* on the study of the different estimators for the malthusian coefficient (in the folder study_malthus_estimation_simulation/),

* on figures to illustrate the different cases of our theorem (in the folder study_theorem/), and test if it is still valid for other distribution than Erlang distribution.

The scripts in all these different folders allow to plot and save some figures (in output folder). Input files linked to simulations are not given because of their size but can be created with scripts from this repository.


## Setup

In order to install all the packages needed to launch the different scripts, we can use the file "requirements.txt".

```bash
pip install --user -r requirements.txt
```
