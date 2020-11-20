# Flow Classification Pipeline
A pipeline for training a flow classification neural network model

#### Path to <i>FlappyBird</i> dataset
pipeline/flappy-bird-dataset/flappy_bird_dataset

#### Installation steps
```bash
(pipeline) poetry install
```

#### Intitiating training of <i>Gunnar-Farneback</i> reward model
```bash
(pipeline) poetry run python3.6 -m gunnar_farneback_reward_model
```

#### Visualizing performance of the model
```bash
(GunnarFarnebackRewardModel) tensorboard --logdir .  
```
