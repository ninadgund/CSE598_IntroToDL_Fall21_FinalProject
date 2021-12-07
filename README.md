# CSE598_IntroToDL_Fall21_FinalProject

How to replicate the results and execute the code.

Environment used :
```
OS: Windows 10
Software: MATLAB 2021b
Hardware: NVIDIA GeForce GTX 1660Ti, 4GB
```

1. First download the dataset from:
	- https://www.kaggle.com/c/dog-breed-identification/data

2. Use this script to modify the dataset so it can be used by the networks:
	- ModifyDataset.mlx

3. Each model was exported in its own live MATLAB script file.
	- Running the respective live scripts (notebooks) will load the network and start training:
		- TransferLearningAlexNet.mlx
		- TransferLearningAlexNetDesigner.mlx
		- TransferLearningGoogLeNetFinal.mlx
		- TransferLearningResNet18.mlx

4. Final trained networks, their output and workspace variables are stored in the respective .mat files, and can be imported in the workspace directly.
	- TransferLearningAlexNet.mat
	- TransferLearningAlexNetDesignerTest.mat
	- TransferLearningGoogLeNet.mat
	- AlexNet_params_2021_12_05__20_36_16.mat
	- GoogLeNetparams_2021_12_05__19_36_29.mat
	- ResNet18_params_2021_12_05__21_57_15.mat

5. Live scripts of manual validation are exported as both pdf and html.

6. Code and ReadMe for ResNet50V2 are in its own directory
	- /ResNet50V2

