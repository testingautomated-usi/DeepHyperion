# Getting Started #

Follow the steps below to set up DeepHyperion and validate its general functionality.


## Step 1: Configure the environment  ##

Pull our pre-configured Docker image for DeepHyperion-MNIST:

``` 
docker pull zohdit/deephyperion:latest
```

Run it by typing in the terminal the following commands:

```
docker run -it --rm zohdit/deephyperion:latest
source .venv/bin/activate
```

## Step 2: Run DeepHyperion ##
Use the following commands to start a 3 minutes run of DeepHyperion-MNIST with the "Bitmaps - Orientation" combination of features:

```
cd DeepHyperion/DeepHyperion-MNIST
python mapelites_mnist.py
```
> NOTE: properties.py contains the tool configuration. The user should edit this file to change the configuration. 
> 
> NOTE: If you want to run _DeepHyperion-MNIST_ with the same configuration as in the paper, you need to set RUNTIME in properties.py as follows:
```
RUNTIME  = int(os.getenv('DH_RUNTIME', '3600'))
```

When the run is finished, the tool produces the following outputs in the _logs/run_XXX_ folder (where XXX is the timestamp value):

* _heatmap_Bitmaps_Orientation.png_: image representing the feature map;
* _heatmap_Bitmaps_Orientation.json_: file containing the final report of the run;
* _all_: folder containing all the inputs generated during the run (in image and npy formats);
* _archive_: folder containing the solutions found during the run (in image and npy formats).


## Step 3: Generate Maps  ##

To generate rescaled maps and process the output of a run, use the following commands:

```
python report_generator/app.py generate-samples ./logs/run_XXX
python report_generator/app.py extract-stats --parsable --feature bitmaps --feature orientation ./logs/run_XXX/archive
```
Where _logs/run_XXX_ is the path of a folder containing the results of a run, e.g. the results obtained in Step 2.
You should get an output similar to the following:
  
```
2020-12-22 22:41:02,764 INFO     Process Started
name=orientation,min=7,max=94,missing=0
name=bitmaps,min=3,max=207,missing=0
```
This output reports, for each feature specified as input: its name, its min and max values, and the count of cells in the interval [min:max] in which inputs are missing (i.e. not found by this run). 

To generate the map and the report, run the following command:

```
python report_generator/app.py generate-map --feature bitmaps <MIN feature 1> <MAX feature 1> 25 --feature orientation <MIN feature 2> <MAX feature 2> 25 ./logs/run_XXX/archive
```
> NOTE: You should set the <MIN> <MAX> values for each feature based on previous command's output, otherwise, you might loose some individuals which are out of your defined bind. In our example, we ran the following command:
```
python report_generator/app.py generate-map --feature bitmaps 7 94 25 --feature orientation 3 207 25 ./logs/run_XXX/archive
```  

The output can be found in the _logs/run_XXX/archive_ folder:

* coverage-DeepHyperion-X-orientation-bitmaps-Orientation-Bitmaps-black-box-rescaled.npy
* misbehaviour-DeepHyperion-X-orientation-bitmaps-Orientation-Bitmaps-black-box-rescaled.npy
* probability-DeepHyperion-X-orientation-bitmaps-Orientation-Bitmaps-black-box-rescaled.npy
* DeepHyperion-X-Orientation-Bitmaps-black-box-rescaled-stats.json
* probability-DeepHyperion-X-orientation-bitmaps-Orientation-Bitmaps-black-box-rescaled.pdf

<p align="center">
<img src="probability-DeepHyperion-X-orientation-bitmaps-Orientation-Bitmaps-black-box-rescaled.PNG" alt="map" style="width:1px;"/></p>

To check the results and maps, you should copy the files from docker to your system, as follows:
```
docker cp <YOUR_DOCKER_NAME>:/DeepHyperion/DeepHyperion-MNIST/logs/run_XXX/archive/  /path-to-your-Desktop/
```
> NOTE: you can find your docker's name using this command:
```
docker ps -a
```

## Step 4: Reproduce Experimental Results ##

We provided the data of all runs of tools in _experiments/data_. To regenerate the plots reported in the paper, run the following commands on the provided docker:

```
cd DeepHyperion/experiments
source .venv/bin/activate
python rq1.py
python rq2.py
python rq3.py
```

Then, you will find the following files in _plots_ folder:


* RQ1-BeamNG.pdf
* RQ1-MNIST.pdf
* RQ2-BeamNG.pdf
* RQ2-MNIST.pdf
* RQ3-BeamNG.pdf
* RQ3-MNIST.pdf

These plots corresponds to the ones reported in Figures 3 -- 8 of the (pre-print) version of the paper.
To check the results, you should copy the files from docker to your system, as follows:
```
docker cp <YOUR_DOCKER_NAME>:/DeepHyperion/experiments/plots  /path-to-your-Desktop/
```

