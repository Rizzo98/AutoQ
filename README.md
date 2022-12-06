# AutoQ
A library **(W.I.P)** that automatically quantizes a Keras Model in Qm.n format. Configurable with [Hydra](https://github.com/facebookresearch/hydra).

## How to install
```
pip install -r requirements.txt
```

## How to run
```
python main.py
```

## How it works
Given a model **M** with **L** layers and a quatization bits space **B**, the number of total configurations is **B^L**. The goal of this tool is to find for **M** the best configuration among the **B^L** possible ones.\
AutoQ follows the following pipeline:
- Compute, for each layer **l**, the best values **m** and **n** in order to minimize the quantization loss.
- Sample a subset of the **B^L** configurations. This will be the training set for the regressor (see following)
- Train a regressor **r** that receives in input a configuration **q** and predicts the performances of the model **M** quantized following **q**.
- Execute an optimization algorithm that explore the configuration space (using **r**), in order to find the best configuration **q** for **M** that preserves the performances while minimizes the number of bits used.\

## How to configure
In [config](./config/config.yaml) the default paremeters are defined:
- **qmnAnalyzer** defines which class is responsible for extracting the informations for each layer
- **model** defines the model **M**
- **dataset** defines the dataset on which **M** will be evaluated
- **explorer** defines the class that will be responsible of creating the training set for the regressor **r**
- **regressor** defines the regressor **r**
- **optimizer** defines the optimizer algorithm
- **bitSpace** defines the bit space **B** 