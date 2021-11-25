# Experimental Evaluation: Data and Scripts #

## General Information ##

This folder contains the data we obtained by conducting the experimental procedure described in the paper (Section 4). We used this data to generate the plots reported in the paper.

## Dependencies ##

The scripts require `make` and `python 3.7`

### Configure Ubuntu ###
Pull an Ubuntu Docker image, run and configure it by typing in the terminal:

``` 
docker pull ubuntu:bionic
docker run -it --rm ubuntu:bionic
apt-get update && apt-get upgrade -y && apt-get clean
apt-get install make
```
And check if Make is correctly installed, by typing the following command:
```
make -version
GNU Make 4.1
```

Check that the version of Make matches `4.1`.

### Copy the project into the docker container ###

To copy DeepHyperion inside the docker container, open another console and run:

``` 
docker cp <DEEP_HYPERION_HOME>/ <DOCKER_ID>:/
```

Where `<DEEP_HYPERION_HOME>` is the location in which you downloaded the artifact and `<DOCKER_ID>` is the ID of the ubuntu docker image just started.

You can find the id of the docker image using the following command:

```
docker ps -a

CONTAINER ID   IMAGE           COMMAND       CREATED          STATUS          PORTS     NAMES
13e590d65e60   ubuntu:bionic   "/bin/bash"   2 minutes ago   Up 2 minutes             recursing_bhabha
```

### Installing Python 3.7 ###
Install Python 3.7:

``` 
apt-get install -y curl python3.7 python3.7-dev python3.7-distutils
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1
update-alternatives --set python3 /usr/bin/python3.7
curl -s https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py --force-reinstall && \
    rm get-pip.py
apt-get install -y python3.7-venv
```

And check if it is correctly installed, by typing the following command:

``` 
python3 -V

Python 3.7.5
```

Check that the version of python matches `3.7.*`.

### Installing pip ###

Use the following commands to install pip and upgrade it to the latest version:

``` 
apt install -y python3-pip
python3 -m pip install --upgrade pip
```

Once the installation is complete, verify the installation by checking the pip version:

``` 
python3 -m pip --version

pip 21.1.1 from /usr/local/lib/python3.7/dist-packages/pip (python 3.7)
```

## (Re)Generate the plots ##

To regenerate the plots, run the following command from the **current** folder:

```
cd DeepHyperion/experiments
make plot-all
```

This command sets up a python virtual environment in a predefined location (`./.venv`), extracts into the `data/mnist` and `data/beamng` folders the raw data (feature-maps, probability-maps, etc.), and creates the various plots.

The command checks that the expected version of python is installed and fails otherwise.

The command produces a verbose output (sorry) and a few warnings like the following one:

```
...
plotting_utils.py:354: RuntimeWarning: Mean of empty slice
  avg_probabilities = np.nanmean(all_probabilities, axis=0)
...
```
 
Those warnings are expected, so do not worry.

If everything worked as expected you'll find the following plots under `./plots`:

* `RQ1-MNIST.pdf` (Figure 3: RQ1: Misbehaviours found by DeepHyperion, DeepJanus and DLFuzz on MNIST)
* `RQ1-BeamNG.pdf` (Figure 4: RQ1: Misbehaviours found by DeepHyperion and DeepJanus on BeamNG)  
* `RQ2-MNIST.pdf` (Figure 5: RQ2: Map cells filled by DeepHyperion, DeepJanus and DLFuzz on MNIST)
* `RQ2-BeamNG.pdf` (Figure 6: RQ2: Map cells filled by DeepHyperion and DeepJanus on BeamNG)
* `RQ3-MNIST.pdf`  (Figure 7: RQ3: Probability maps and feature discrimination for MNIST)
* `RQ3-BeamNG.pdf`  (Figure 8: RQ3: Probability maps and feature discrimination for BeamNG)

Those plots corresponds to the ones reported in Figures 3 -- 8 of the (pre-print) version of the paper.
