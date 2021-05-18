# Getting Started #

Follow the steps below to set up DeepHyperion and validate its general functionality.


## Step 1: Configure the environment  ##

Pull our pre-configured Docker image for DeepHyperion-MNIST:

``` 
docker pull zohdit/deephyperion:v1.2
```

Run it by typing in the terminal the following commands:

```
docker run -it --rm zohdit/deephyperion:v1.2
. .venv/bin/activate
```

## Step 2: Run DeepHyperion ##
Use the following commands to start a 3 minutes run of DeepHyperion-MNIST with the "Bitmaps - Orientation" combination of features:

```
cd DeepHyperion/DeepHyperion-MNIST
python mapelites_mnist.py
```

> NOTE: `properties.py` contains the tool configuration. You should edit this file to change the configuration. For example, if you want to run <i>DeepHyperion-MNIST</i> with the same configuration as in the paper, you need to set the `RUNTIME` variable inside `properties.py` as follows:
```
RUNTIME  = int(os.getenv('DH_RUNTIME', '3600'))
```

When the run ends, on the console you should see a message like this:

```
2021-05-14 14:27:41,494 INFO     Best overall value: -1.0 produced by individual <individual.Individual object at 0x7f48f91cc4a8> and placed at (54, 93)
Exporting inputs ...
Done
```

The tool produces the following outputs in the `logs/run_XXX_/log_800_YYY` folder (where XXX is the timestamp value):

* `log_800_YYY` folder (where YYY is the number of iterations) containing:
  * `heatmap_Bitmaps_Orientation.png`: image representing the feature map;
  * `heatmap_Bitmaps_Orientation.json`: file containing the final report of the run;
* `all`: folder containing all the inputs generated during the run (in image and npy formats);
* `archive`: folder containing the solutions found during the run (in image and npy formats).


## Step 3: Generate Maps  ##

To generate rescaled maps and process the output of a run, use the following commands:
> NOTE: Run these commands from the DeepHyperion/DeepHyperion-MNIST directory

```
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
python report_generator/app.py generate-samples ./logs/run_XXX
python report_generator/app.py extract-stats --parsable --feature bitmaps --feature orientation ./logs/run_XXX/archive
```
Where `logs/run_XXX` is the path of a folder containing the results of a run, e.g. the results obtained in Step 2.
You should get an output similar to the following:
  
```
2020-12-22 22:41:02,764 INFO     Process Started
name=orientation,min=7,max=94,missing=0
name=bitmaps,min=3,max=207,missing=0
```
This output reports, for each feature specified as input: 

- its name
- its min and max values
- the count of cells in the interval [min:max] in which inputs are missing (i.e., not found by this run). 

To generate the map and the report, run the following command:

```
python report_generator/app.py generate-map --feature bitmaps [MIN feature 1] [MAX feature 1] 25 --feature orientation [MIN feature 2] [MAX feature 2] 25 ./logs/run_XXX/archive
```

> NOTE: You should set the minimum and maximum values for each feature based on previous command's output, otherwise, you might loose some individuals which are out of your defined bounds. In our example, we ran the following command:

```
python report_generator/app.py generate-map --feature bitmaps 7 94 25 --feature orientation 3 207 25 ./logs/run_XXX/archive
```  

> NOTE: This command may produce RuntimeWarnings. Do not worry and proceed with the next steps.

This command produces many files in the `logs/run_XXX/archive` folder; the most relevant ones are:

* `coverage-DeepHyperion-<RUN_ID>-orientation-bitmaps-Orientation-Bitmaps.npy`
* `misbehaviour-DeepHyperion-<RUN_ID>-orientation-bitmaps-Orientation-Bitmaps.npy`
* `probability-DeepHyperion-<RUN_ID>-orientation-bitmaps-Orientation-Bitmap.npy`
* `DeepHyperion-<RUN_ID>-Orientation-Bitmaps-stats.json`
* `probability-DeepHyperion-<RUN_ID>-orientation-bitmaps-Orientation-Bitmaps.pdf`

The `.npy` files contain the raw data collected from the tool's execution, the `-stats.json` file contains the statistics of the execution, while the `-rescaled.pdf` file contains a visualization of the Misbehavior Probability map, similar to the following:

![](./probability-DeepHyperion-X-orientation-bitmaps-Orientation-Bitmaps-black-box-rescaled.PNG)


You can copy those files from the running docker image to your system, as follows:


> NOTE: you should run this command outside the docker, i.e., opening a new terminal window

```
docker cp <YOUR_DOCKER_NAME>:/DeepHyperion/DeepHyperion-MNIST/logs/run_XXX/archive/  /path-to-your-Desktop/
```

You can find your docker's name and id using the following command:

```
docker ps -a

CONTAINER ID   IMAGE                      COMMAND       CREATED          STATUS          PORTS     NAMES
3a77c777954d   zohdit/deephyperion:v1.1   "/bin/bash"   25 minutes ago   Up 25 minutes             tender_zhukovsky
```

In this case <YOUR_DOCKER_NAME> is tender_zhukovsky


## Step 4: Reproduce Experimental Results ##

In case you want to regenerate the plots in the paper without re-running all the 100h+ of experiments, we provided the data of all runs of all the tools in `experiments/data`. 

To regenerate the plots reported in the paper, run the commands we report below on the provided docker.
> **NOTE**: Be sure to deactivate the virtual environment you used for steps 1 -- 3 (simply by running the `deactivate` command) and activate the one inside the `experiments` folder. 
> Despite they share the same name, those virtual environments contain different libraries.

```
cd /DeepHyperion/experiments
. .venv/bin/activate
python rq1.py
python rq2.py
python rq3.py
```

> NOTE: These commands may produce RuntimeWarnings. Do not worry about them. The commands are successful if the plots are stored.

Then, you will find the following files in `plots` folder:

* `RQ1-MNIST.pdf` (Figure 3: RQ1: Misbehaviours found by DeepHyperion, DeepJanus and DLFuzz on MNIST)
* `RQ1-BeamNG.pdf` (Figure 4: RQ1: Misbehaviours found by DeepHyperion and DeepJanus on BeamNG)  
* `RQ2-MNIST.pdf` (Figure 5: RQ2: Map cells filled by DeepHyperion, DeepJanus and DLFuzz on MNIST)
* `RQ2-BeamNG.pdf` (Figure 6: RQ2: Map cells filled by DeepHyperion and DeepJanus on BeamNG)
* `RQ3-MNIST.pdf`  (Figure 7: RQ3: Probability maps and feature discrimination for MNIST)
* `RQ3-BeamNG.pdf`  (Figure 8: RQ3: Probability maps and feature discrimination for BeamNG)


These plots correspond to the ones reported in Figures 3 -- 8 of the (pre-print) version of the paper.
To check the results, you should copy the files from the running docker to your system, as follows:

```
docker cp <YOUR_DOCKER_NAME>:/DeepHyperion/experiments/plots  /path-to-your-Desktop/
```

