# DeepHyperion-BNG

Test Input Generator using illumination search agorithm

## General Information ##
This folder contains the application of the DeepHyperion approach to the steering angle prediction problem.
This tool is developed in Python on top of the DEAP evolutionary computation framework. It has been tested on a Windows machine equipped with a i9 processor, 32 GB of memory, and an Nvidia GPU GeForce RTX 2080 Ti with 11GB of dedicated memory.

## Dependencies ##

### Installing Python 3.7.9 ###

Install [_Python 3.7.9_](https://www.python.org/ftp/python/3.7.9/python-3.7.9-amd64.exe)

To easily install the dependencies with pip, we suggest to create a dedicated virtual environment and run the command:

```pip install -r requirements.txt```

Otherwise, you can manually install each required library listed in the requirements.txt file using pip.

_Note:_ the version of Shapely should match your system.


### BeamNG simulator ###

This tool needs the BeamNG simulator to be installed on the machine where it is running. 
A free version of the BeamNG simulator for research purposes can be obtained by registering at [https://register.beamng.tech](https://register.beamng.tech) and following the instructions provided by BeamNG. Please fill the "Application Text" field of the registration form with the following text:

```
I would like to replicate the results of the DeepHyperion paper, 
accepted at the ISSTA 2021 conference and for that I need to a
copy of BeamNG.research
```
> **NOTE**: as stated on the BeamNG registration page, **please use your university email address**. 

#### Post Registration
If BeamNG.GmbH accepts your registration you will receive an email with a link to download the simulator and a registration key file named `tech.key`.
Please download the following version `BeamNG.research 1.7.0.1` and **rename** the `tech.key` to `research.key`.

The first time you run the simulation it creates a folder under `~/Documents/BeamNG.reseach` and checked whether the `research.key` is present or not. If it cannot find the `research.key` files it will show a message and stop. Copy the `research.key` in the folder to fix this.

The simulator requires Windows 10 (at least), but can be run also on a Mac provided it is installed inside a virtual machine managed by [Parallels](https://www.parallels.com/eu/). Parallels is not a free software, but probably a temporary evaluation license can be obtained. 

> **WARNING:** Running BeamNG inside a VM might result in lags and other misbehaviors, so we advice to use a physical machine, whenever is possible. Additionally, running BeamNG inside VMs managed by hypervisors other than Parallels (e.g., VMWare or VirtualBox) does not work smoothly.

## Recommended Requirements ##

[BeamNG](https://wiki.beamng.com/Requirements) recommends the following hardware requirements:

* OS: Windows 10 64-Bit
* CPU: AMD Ryzen 7 1700 3.0Ghz / Intel Core i7-6700 3.4Ghz (or better)
* RAM: 16 GB RAM
* GPU: AMD R9 290 / Nvidia GeForce GTX 970
* DirectX: Version 11
* Storage: 20 GB available space
* Additional Notes: Recommended spec based on 1080p resolution. Installing game mods will increase required storage space. Gamepad recommended.

>**Note**: BeamNG.research can run also on Mac Books, provided that you boot them on Windows.

## Usage ##

### Input ###

* A trained model in h5 format. The default one is in the folder _data/trained_models_colab_;
* The seeds used for the input generation. The default ones are in the folder _data/member_seeds/initial_population_;
* _core/config.py_ containing the configuration of the tool selected by the user. 
_Note:_ you need to define feature combination in the config

### Output ###
When the run is finished, the tool produces the following outputs in the _logs_ folder:
* maps representing inputs distribution;
* json files containing the final reports of the run;
* folders containing the generated inputs (in image and json format).

### Run the Tool ###
Run _core/mapelites_bng.py_

### Generate Processed Data and Rescaled Maps ###

* [__DeepHyperion-BNG/report_generator__](./DeepHyperion-BNG/report_generator)



## More Usages ##

### Train a New Predictor ###

* Run _udacity_integration/train-dataset-recorder-brewer.py_  to generate a new training set;
* Run _udacity_integration/train-from-recordings.py_  to train the ML model.

### Generate New Seeds ###

Run _self_driving/main_beamng_generate_seeds.py_
