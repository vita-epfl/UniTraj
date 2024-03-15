# UniTraj

**A Unified Framework for Cross-Dataset Generalization of Vehicle Trajectory Prediction**

💡UniTraj allows users to train and evaluate trajectory prediction models from real-world datasets like Waymo, nuPlan, 
nuScenes and Argoverse2 in a unified pipeline. 

![system](docs/assets/framework.png)

🔥Powered by [Hydra](https://hydra.cc/docs/intro/), [Pytorch-lightinig](https://lightning.ai/docs/pytorch/stable/), and [WandB](https://wandb.ai/site), the framework is easy to configure, train and logging.

![system](docs/assets/support.png)

## 🛠 Quick Start
0. Create a new conda environment
```bash
conda create -n unitraj python=3.9
conda activate unitraj
```
1. Install ScenarioNet: https://scenarionet.readthedocs.io/en/latest/install.html

2. Install Unitraj:
```bash
git clone https://github.com/vita-epfl/MotionNet.git
cd unitraj
pip install -r requirements.txt
wandb login
```

You can verify the installation of UniTraj via running the training script:
```bash
python train.py method=autobot
```
The model will be trained on several sample data.

## Code Structure
There are three main components in UniTraj: dataset, model and config.
The structure of the code is as follows:
```
motionnet
├── configs
│   ├── config.yaml
│   ├── method
│   │   ├── autobot.yaml
│   │   ├── MTR.yaml
│   │   ├── wayformer.yaml
├── datasets
│   ├── base_dataset.py
│   ├── autobot_dataset.py
│   ├── wayformer_dataset.py
│   ├── MTR_dataset.py
├── models
│   ├── autobot
│   ├── mtr
│   ├── wayformer
│   ├── base_model
├── utils
```
There is a base config, dataset and model class, and each model has its own config, dataset and model class that inherit from the base class.

## Pipeline
### 1. Data Preparation
UniTraj takes data from [ScenarioNet](https://github.com/metadriverse/scenarionet) as input. Process the data with ScenarioNet in advance.

### 2. Configuration
UniTraj uses [Hydra](https://hydra.cc/docs/intro/) to manage configuration files.

Universal configuration file is located in `motionnet/config/config.yaml`.
Each model has its own configuration file in `motionnet/config/method/`, for example, `motionnet/config/method/autobot.yaml`.

The configuration file is organized in a hierarchical structure, and the configuration of the model is inherited from the universal configuration file.

#### Configuration Example
TODO

### 2. Train
```python train.py```

### 3. Evaluation
```python evaluation.py```

### 4. Dataset Analysis
```python data_analysis.py```


## Contribute to UniTraj
### Implement a new model
1. Create a new config file in `motionnet/config/` folder, for example, `motionnet/config/lanegcn.yaml`
2. (Optional) Create a new dataset class in `motionnet/dataset/` folder, for example, `motionnet/dataset/lanegcn_dataset.py`, and inherit `motionnet/dataset/base_dataset.py`, implement `def process(data)` function
2. Create a new model class in `motionnet/model/` folder, for example, `motionnet/model/lanegcn.py`, and inherit from pl.LightningModule

### Internal Format
ScenarioNet data will be further preprocessed to internal format for easy processing. 

``obj_trajs [num_centered_obj, num_surrounding_objs, past_time_steps, num_attribute=29]
``

``
[0:3] position (x, y, z)  
[3:6] size (l, w, h)
[6:11] type_onehot
[11:23] time_onehot
[23:35] heading_encoding
[25:37] vx,vy
[27:39] ax,ay
``

``
map_polylines [num_centered_obj, num_surrounding_lines, max_points_per_lane, num_attribute=9]
``

``
[0:3] position (x, y, z)  
[3:6] direction (x, y, z)
[6] type
[7:9] previous_point_xy 
``

## Training on RCP
0. Install runAI CLI and Kubernetes: https://wiki.rcp.epfl.ch/home/CaaS/Quick_Start
1. Update the motionnet/run_rcp/wandb-secret.yaml according to the instruction: https://wiki.rcp.epfl.ch/en/home/CaaS/how-to-use-secret 
2. Clone the repo to RCP server, and modify the config file
3. Modify the motionnet/run_rcp/train.yaml, especially configs related to file path, username, etc.
3. Run the following command to train on RCP server
```kubectl create -f train.yaml```

