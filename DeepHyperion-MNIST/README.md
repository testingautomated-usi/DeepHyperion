# DeepHyperion-MNIST #

Input test generator using illumination search algorithm

## General Information ##
This folder contains the application of the DeepHyperion  approach to the handwritten digit classification problem.
This tool is developed in Python on top of the DEAP evolutionary computation framework. It has been tested on a machine featuring an i7 processor, 16 GB of RAM, an Nvidia GeForce 940MX GPU with 2GB of memory. These instructions are for Ubuntu 18.04 (bionic) OS and python 3.6.

## Dependencies ##

> NOTE: If you want to use DeepHyperion-MNIST easily without configuring environment, you can also see [__Getting Started__](./documentation/getting_started.md)

### Configure Ubuntu ###
Pull an Ubuntu Docker image, run and configure it by typing in the terminal:

``` 
docker pull ubuntu:bionic
docker run -it --rm ubuntu:bionic
apt update && apt-get update
apt-get install -y software-properties-common
```


### Installing Python 3.6 ###
Install Python 3.6
``` 
add-apt-repository ppa:deadsnakes/ppa
apt update
apt install -y python3.6
```

And check if it is correctly installed, by typing the following command:

``` 
$ python3
```

You should have a message that tells you are using python 3.6.x, similar to the following:

``` 
Python 3.6.9 (default, Apr 18 2020, 01:56:04) 
[GCC 8.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
```

Exit from python.

### Installing pip ###
Use the following commands to install pip and upgrade it to the latest version:
``` 
apt install -y python3-pip
python3 -m pip install --upgrade pip
```

Once the installation is complete, verify the installation by checking the pip version:

``` 
python3 -m pip --version
```

### Installing git ###
Use the following command to install git
``` 
apt install -y git
```

To check the correct installation of git, insert the command git in the terminal. If git is correctly installed, the usage information will be shown.

### Installing Python Binding to the Potrace library ###
Installing Python Binding to the Potrace library
``` 
$ sudo apt-get install build-essential python-dev libagg-dev libpotrace-dev pkg-config
``` 

Install pypotrace:

``` 
$ git clone https://github.com/flupke/pypotrace.git
$ cd pypotrace
$ pip install numpy
$ pip install .
``` 

Installing PyCairo and PyGObject

Instructions provided by https://pygobject.readthedocs.io/en/latest/getting_started.html#ubuntu-getting-started.


``` 
$ apt-get install python3-gi python3-gi-cairo gir1.2-gtk-3.0
$ apt-get install libgirepository1.0-dev gcc libcairo2-dev pkg-config python3-dev gir1.2-gtk-3.0 librsvg2-dev
``` 

Installing Other Dependencies

This tool has other dependencies such as tensorflow and deap.

To easily install the dependencies with pip:

``` 
$ pip install -r requirements.txt
``` 

Otherwise, you can manually install each required library listed in the requirements.txt file using pip.

## Usage ##
### Input ###

* A trained model in h5 format. The default one is in the folder models;
* A list of seeds used for the input generation. In this implementation, the seeds are indexes of elements of the MNIST dataset. The default list is in the file bootstraps_five;
* properties.py containing the configuration of the tool selected by the user.

### Output ###

When the run is finished, the tool produces the following outputs in the logs folder:

* maps representing inputs distribution;
* json files containing the final reports of the run;
* folders containing the generated inputs (in image format).

### Run the Tool ###

Run the command: python mapelites_mnist.py

### Generate Processed Data and Rescaled Maps ###

* [__DeepHyperion-MNIST/report_generator__](../DeepHyperion-MNIST/report_generator)



### Troubleshooting ###

* If tensorflow cannot be installed successfully, try to upgrade the pip version. Tensorflow cannot be installed by old versions of pip. We recommend the pip version 20.1.1.
* If the import of cairo, potrace or other modules fails, check that the correct version is installed. The correct version is reported in the file requirements.txt. The version of a module can be checked with the following command:
```
$ pip3 show modulename | grep Version
```
To fix the problem and install a specific version, use the following command:
```
$ pip3 install 'modulename==moduleversion' --force-reinstall
```


