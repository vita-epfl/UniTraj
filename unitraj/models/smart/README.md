## This part is to add [SMART](https://github.com/rainmaker22/SMART) model in the Unitraj pipeline

0. Create environment & preperation

```bash
conda create -n unitraj_smart python=3.9
conda activate unitraj_smart
```

1. Install ScenarioNet

2. Install Unitraj as in the main README

3. Install SMART requirement from [SMART](https://github.com/rainmaker22/SMART) or try install from here:
```bash
pip install -r requirements.txt
```

You can verify the installation of UniTraj via running the training script:

```bash
python train.py method=SMART
```

The model will be trained on several sample data.

## Training

### 1. Data Preparation

UniTraj takes data from [ScenarioNet](https://github.com/metadriverse/scenarionet) as input. Process the data with
ScenarioNet in advance.

### 2. Configuration

SMART model has its own configuration file in `unitraj/config/method/SMART.yaml`.


### 2. Train
```python train.py```

The default training setups are the same as SMART-7M and when train a SMART model in Unitraj, this should be uncommentted in train.py:

```bash
# accumulate_grad_batches=cfg.method.Trainer.accumulate_grad_batches,
```
The latest Unitraj-SMART version ckpt has minADE=0.827，minFDE=3.300 for reference.
