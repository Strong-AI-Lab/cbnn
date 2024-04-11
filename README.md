# Causal Bayesian Neural Network

Repository of the Causal Bayesian Neural Network (CBNN).


# Installation

Run the following command to install the required packages:
```bash
pip install -r requirements.txt
```

You can install the packages in a virtual environment by running the following commands:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```


# Download data

The data used in the experiments are not included in the repository due to their large memory footprint. They are either automatically downloaded when the dataset is used or they must be downloaded separately, as indicated in the next table. The data can be downloaded from the following links:

| Dataset         | Split | Requires download    | Link                                                                    | File ID                                |
|-----------------|-------|----------------------|-------------------------------------------------------------------------|----------------------------------------|
| MNIST           |       |  auto                |                                                                         |                                        |
| CIFAR10         |       |  auto                |                                                                         |                                        |
| ACRE            | IID   |  X                   | [project homepage](https://wellyzhang.github.io/project/acre.html)      | 1P0WBnnjWolGsrATUQtx4ictiYlOGc-OT      |
|                 | Comp  |  X                   |                                                                         | 1-LZMt08a1v-KSuaQTS1lqD6BCEw47LEY      |
|                 | Comp  |  X                   |                                                                         | 1Sn_tKbe6mMv7Tc_y6hJZnm7lSenjwIys      |
| CONCEPTARC      |       |  X                   |                                                                         |                                        |
| RAVEN           |       |  X                   |                                                                         |                                        |


Download the data using the following command:
```bash
gdown <file_id>
```
`file_id` is the ID of the zip file in google drive, indicated in the last column. The zip must be placed in the corresponding `data/<dataset_name>` folder and unzipped:
```bash
cd data/<dataset_name>
unzip <file_name>.zip
```


# Usage

To run the experiments from a config file, use the following command: 
```bash
python run.py --load_config 'config/<config_file>.yaml' '<model_name>'
```

You can also override any parameter from the config file by passing it as an argument:
```bash
python run.py --load_config 'config/<config_file>.yaml' --<parameter> '<value>' '<model_name>' --<model_parameter> '<value>'
```

Use the `-h` flag to see the available options:
```bash
python run.py -h
```

To see the model options, use the following command:
```bash
python run.py '<model_name>' -h
```

Common arguments:
- `--load_config`: Load the configuration file.
- `--save_config`: Save the configuration file.
- `--data`: Dataset to use.
- `--save`: Path to save the model.
- `--train`, `--test`, `--train_and_test`: Train, test, or train and test the model. Default option is `train_and_test`.
- `--max_epochs`: Maximum number of epochs. You might want to set it as the default value of Pytorch Lightning is 1000.


## Architecture
![Image Description](assets/cbnn_architecture.png)