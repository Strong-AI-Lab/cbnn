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