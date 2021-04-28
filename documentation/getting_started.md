# Getting Started #

Set up DeepHyperion and validate its general functionality

### Run the tool  ###
Pull a prepared Docker image for DeepHyperion-MNIST, run it by typing in the terminal:

``` 
docker pull zohdit:deephyperion
docker run -it --rm zohdit:deephyperion
```

Use the following command to start a 3 minutes run of DeepHyperion-MNIST with one combination of features (Bitmaps and Orientation):

```
source .venv/bin/activate
cd DeepHyperion/DeepHyperion-MNIST
python mapelites_mnist.py
```
> NOTE: properties.py contains the configuration of the tool selected by the user. 

When the run is finished, the tool produces the following outputs in the _logs/run_XXX_ folder:

* _heatmap_Bitmaps_Orientation.png_ representing inputs distribution;
* _heatmap_Bitmaps_Orientation.json_ file containing the final reports of the run;
* folders _all_ and _archive_ containing the generated inputs (in image and npy formats).

To uniform anlysis and generate rescaled maps and more processed data, use the following commands:

```
python report_generator/app.py generate-samples ./logs/run_XXX
python report_generator/app.py extract-stats --parsable --feature bitmaps --feature orintation ./logs/run_XXX
```
You should get an output similar to:
  
```
2020-12-22 22:41:02,764 INFO     Process Started
name=orientation,min=12,max=100,missing=0
name=bitmaps,min=333,max=482,missing=0
name=moves,min=inf,max=-inf,missing=132
```
This outputs report for each feature specified in input its name, its min/max values, and the count of samples found for which that feature was not present.

To generate a map and generate a report run the following command (add `--visualize` if you want to visualize the map):

```
python report_generator/app.py generate-map --feature bitmaps <MIN> <MAX> 25 --feature orientation <MIN> <MAX> 25 ./logs/run_XXX
```
> NOTE: You should set the <MIN> <MAX> values for each feature based on previous command's output, otherwise, you might loose some individuals which are out of your defined bind.  

Then you can find these outputs in _logs/run_XXX/archive_:


* coverage-DeepHyperion-X-orientation-bitmaps-Orientation-Bitmaps-black-box-rescaled.npy
* misbehaviour-DeepHyperion-X-orientation-bitmaps-Orientation-Bitmaps-black-box-rescaled.npy
* probability-DeepHyperion-X-orientation-bitmaps-Orientation-Bitmaps-black-box-rescaled.npy
* DeepHyperion-X-Orientation-Bitmaps-black-box-rescaled-stats.json
* probability-DeepHyperion-X-orientation-bitmaps-Orientation-Bitmaps-black-box-rescaled.pdf





