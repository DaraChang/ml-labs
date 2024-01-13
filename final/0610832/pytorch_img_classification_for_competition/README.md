### Environment Setup

### 1. Virtual Environment Creation
- Download Anaconda and then use conda commands to create a new virtual environment, specifying the Python version as 3.6.
```
$ conda create --name FP python=3.6
$ conda activate FP
```

### 2. Packages
- pytorch 1.2.0
- torch 1.5.0
- torchvision 0.4.0
- CUDA 10.0
- tqdm 4.46.1
- scikit-learn 0.22.1
- matplotlib 3.1.3
- pandas 1.0.3
- progress 1.3

### Execution Instructions
- The training dataset downloaded from the competition website should be placed in the './C1-P1_Train Dev_fixed/C1-P1_Train' directory.
- The validation dataset downloaded from the competition website should be placed in the './C1-P1_Train Dev_fixed/C1-P1_Dev' directory.
- The testing dataset downloaded from the competition website should be placed in the './C1-P1_Test' directory.
- Generated checkpoints would be stored in the './checkpoints' directory.
- The 'config.py' file defines all hyperparameters such as batch size, learning rate, training epochs, and number of classes as well as paths for storing checkpoints, datasets, and prediction results, etc. When conducting training or prediction, you can modify the parameters in this file before execution.


- training
```
(FP)$ python main.py
```

- resume
	- If you want to resume training from where it was last terminated, change the 'resume' parameter in 'config.py' from 'None' to the path where the checkpoints are stored.

- testing
```
(FP)$ python test.py
```
	-The checkpoint with the highest validation accuracy during training will be loaded for prediction. The prediction results are stored in the './submits' directory.
