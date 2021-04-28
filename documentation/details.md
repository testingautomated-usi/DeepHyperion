# DeepHyperion

DeepHyperion is a tool for generating test inputs and feature maps using illumination-search algorithm.


## DeepHyperion-MNIST ##
To set up the environment and run the DeepHyperion tool adapted to the handwritten digit classification case study, follow the instructions [here](/DeepHyperion-MNIST/README.md).


## DeepHyperion-BNG ##
To set up the environment and run the DeepHyperion tool adapted to the self-driving car case study, follow the instructions [here](/DeepHyperion-BNG/README.md). 


## Experimental Data and Scripts ##
To regenerate the plots reported in the paper, follow the instructions [here](/experiments/README.md) 


## Examples ##
- **Perform MNIST experiments with digits different from 5**

This shows the applicability of the _DeepHyperion-MNIST_ on other digit classes of MNIST.
You can configure _DeepHyperion-MNIST_ to generate inputs for digit class "6".
To do this, you should modify the configuration in _DeepHyperion-MNIST/properties.py_ file as follow:
```
    EXPECTED_LABEL = int(os.getenv('DH_EXPECTED_LABEL', '9'))

```

- **Test a different DL model**

This shows the possibility of using different traied models for _DeepHyperion-MNIST_.
You can use a different model for applying the approach in MNIST case study.
To do this, you should train a model and place it in _DeepHyperion-MNIST/models_ folder, then modify the configuration in _DeepHyperion-MNIST/properties.py_ file as follow:
```
    MODEL = os.getenv('DH_MODEL', 'models/<NEW MODEL NAME>')
```


