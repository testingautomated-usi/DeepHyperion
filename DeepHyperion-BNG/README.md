
# DeepHyperion-BNG

Test Input Generator using illumination search algorithm.

## General Information ##
This folder contains the application of the DeepHyperion approach to the steering angle prediction problem.
DeepHyperion is developed in Python on top of the DEAP evolutionary computation framework. 

We tested DeepHyperion on a Windows machine equipped with a i9 processor, 32 GB of memory and an Nvidia GPU GeForce RTX 2080 Ti with 11GB of dedicated memory.

## Dependencies ##

### Installing Python 3.7.9 ###

Install [_Python 3.7.9_](https://www.python.org/ftp/python/3.7.9/python-3.7.9-amd64.exe)

To easily install the dependencies with pip, we suggest to create a dedicated virtual environment. For example, you can use `venv` and create a virtual environment called `.venv` inside the current folder (`DeepHyperion-BNG`):

```python -m venv .venv```

At this point, you need to activate the virtual environment, upgrade `pip`, and install the required packages:

``` 
.\.venv\Scripts\activate

python -m pip install --upgrade pip

pip install beamngpy-1.18.tar.gz

pip install -r requirements-37.txt
```

Alternatively, you can manually install each required library listed in the `requirements-37.txt`.

> **Note**: the version of Shapely should match your system.


### BeamNG simulator ###

This tool needs the BeamNG simulator to be installed on the machine where it is running. 
A free version of the BeamNG simulator for research purposes can be obtained by registering at [https://register.beamng.tech](https://register.beamng.tech) and following the instructions provided by BeamNG. Please fill the "Application Text" field of the registration form with the following text:

```
I would like to replicate the results of the DeepHyperion paper, 
accepted at the ISSTA 2021 conference. Therefore, I need to a
copy of BeamNG.research
```
> **NOTE**: as stated on the BeamNG registration page, **please use your university email address**. 

#### Post Registration
If BeamNG.GmbH accepts your registration, you will receive an email with a link to download the simulator and a registration key file named `tech.key`.
Please download the following version `BeamNG.research 1.7.0.1` and **rename** the `tech.key` to `research.key`.

The first time you run the simulation, it creates a folder under `~/Documents/BeamNG.research` and checks whether the `research.key` is present or not. If it cannot find the `research.key` files it will show an error message and stop. Copy the `research.key` in the correct folder to fix this.

### DeepHyperion Configuration ###
Edit the parameter named BNG_HOME in `core/config.py` by inserting the path pointing to the folder **trunk**, which is inside the folder of the simulator you dowloaded. 

## Recommended Requirements ##

[BeamNG](https://wiki.beamng.com/Requirements) recommends the following hardware requirements:

* OS: Windows 10 64-Bit
* CPU: AMD Ryzen 7 1700 3.0Ghz / Intel Core i7-6700 3.4Ghz (or better)
* RAM: 16 GB RAM
* GPU: AMD R9 290 / Nvidia GeForce GTX 970
* DirectX: Version 11
* Storage: 20 GB available space
* Additional Notes: Recommended spec based on 1080p resolution. Installing game mods will increase required storage space. Gamepad recommended.

>**Note**: BeamNG.research can run on Mac Books, provided that you boot them on Windows, or inside a virtual machine, provided the it is managed by [Parallels](https://www.parallels.com/eu/). Parallels is not a free software, but probably a temporary evaluation license can be obtained. 

> **WARNING:** Running BeamNG inside a VM results in lags and possibly other misbehaviors. So we advice to use a physical machine, if possible. Additionally, running BeamNG inside VMs managed by hypervisors other than Parallels (e.g., VMWare or VirtualBox) did not work smoothly in our evaluations.


## Usage ##

### Input ###

* A trained model in h5 format. The default one is in the folder `data/trained_models_colab`

* The seeds used for the input generation. The default ones are in the folder `data/member_seeds/initial_population`

* The feature combination to use for the generation. 

To specify the feature combination to use for the generation, directly edit the `core/config.py` file. The available configuration options are already listed inside the `core/config.py` file, uncomment the chosen one that or leave the default configuration:

```
self.Feature_Combination = ["SegmentCount", "MeanLateralPosition"]
```

### Output ###

When the run is finished, the tool produces the following outputs in the `logs` folder:

* maps representing inputs distribution;
* json files containing the final reports of the run;
* additional folders containing the generated inputs (as png and json files).

### Run the Tool ###
To run the tool, activate the python virtual environment and run the `core/mapelites_bng.py` script:

```
.\.venv\Scripts\activate

py.exe core/mapelites_bng.py
```

If everything works fine, you should see that BeamNG.research starts automatically, loads a newly generated test scenario, and the virtual car moves autonomously. As described in the paper, the car is driven by a controller based on the NVidia DAVE2 architecture that receives images from the camera mounted on the virtual car (not from the user camera that shows the simulator on the screen) and predict the steering angle to apply. On the console, you might see messages about CUDA and other CNN-related libraries.

Click on the image to watch the demo:
[![Watch the video](https://img.youtube.com/vi/a_fE4QRpCBQ/hqdefault.jpg)](https://www.youtube.com/watch?v=a_fE4QRpCBQ)

### Troubleshooting

Since the setup is complex, it might now work out-of-the box. Possible issues include:

- Problems with `matplotlib` that might raise the following error: `from matplotlib import ft2font: "ImportError: DLL load failed: The specified procedure could not be found."`. The solution proposed in this [SO](https://stackoverflow.com/questions/24251102/from-matplotlib-import-ft2font-importerror-dll-load-failed-the-specified-pro
) is to downgrade `matplotlib`:
    
    ```
    pip uninstall matplotlib
    pip install matplotlib==3.0.3
    ```

- Missing DLLs that prevent Tensorflow to be installed or used (required to NVidia Dave 2, i.e., the test subject). The error message is self-explanatory, and the fix is to download and install the missing libraries (see this [SO](https://stackoverflow.com/questions/60157335/cant-pip-install-tensorflow-msvcp140-1-dll-missing) for more details)

- BeamNG executable cannot be found. This can be solved by setting the `BNG_HOME` env variable to point to the directory where you have extracted BeamNG.research (e.g, `C:\BeamNG.research.v1.7.0.1`)

- BeamNG cannot find the `registration.key`. BeamNG looks for this file in predefined directories, but you can force a specific directory by setting the `BNG_USER` env variable.

> **NOTE**: A known BeamNG bug is caused by whitespaces in path, so avoid to use paths that contains whitespaces for `BNG_HOME` and `BNG_USER`


### Generate Processed Data and Rescaled Maps ###

* [__DeepHyperion-BNG/report_generator__](../DeepHyperion-BNG/report_generator)


## More Usages ##

### Train a New Predictor ###

* Run `udacity_integration/train-dataset-recorder-brewer.py`  to generate a new training set;
* Run `udacity_integration/train-from-recordings.py`  to train the ML model.

### Generate New Seeds ###

Run `self_driving/main_beamng_generate_seeds.py`
