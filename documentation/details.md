# DeepHyperion

DeepHyperion is a tool for generating test inputs and feature maps using illumination-search algorithm.


## DeepHyperion-MNIST ##
To set up the environment and run the DeepHyperion tool adapted to the handwritten digit classification case study, follow the instructions [here](/DeepHyperion-MNIST/README.md).


## DeepHyperion-BNG ##
To set up the environment and run the DeepHyperion tool adapted to the self-driving car case study, follow the instructions [here](/DeepHyperion-BNG/README.md). 


## Experimental Data and Scripts ##
To regenerate the plots reported in the paper, follow the instructions [here](/experiments/README.md) 


## Extra Use Case Scenarios ##
This section contains plausible scenarios on how DeepHyperion could be extended beyond the experiments performed in the paper.
### Scenario 1: Performing MNIST experiments with digits different from 5s ###

This scenario shows the applicability of the _DeepHyperion-MNIST_ o digit classes of MNIST different from the ones considered in the experimental evaluation.
As an example, you can configure _DeepHyperion-MNIST_ to generate inputs for digit class "6".
To do this, you should modify the configuration in _DeepHyperion-MNIST/properties.py_ file as follows:
```
    EXPECTED_LABEL = int(os.getenv('DH_EXPECTED_LABEL', '6'))

```

### Scenario 2: Test a different DL model ###

This scenario shows the possibility of using trained models for _DeepHyperion-MNIST_ different from the one considered in the experimental evaluation.
To do this, you should train a model and place it in _DeepHyperion-MNIST/models_ folder, then modify the configuration in _DeepHyperion-MNIST/properties.py_ file as follows:
```
    MODEL = os.getenv('DH_MODEL', 'models/<NEW MODEL NAME>')
```


