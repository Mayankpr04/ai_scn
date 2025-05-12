This project aims to replicate the Scale-Wise Convolution model as presented in the [original repository](https://github.com/ychfan/scn).

## Models
Two versions of the model are available in this repository:
1. **Original Model**: A simpler version with basic functionality.
2. **Adaptive Model**: An enhanced version with additional features like scale-wise attention.

Pre-trained weights for both models are included, allowing you to directly run the models on any image or image dataset without additional training.

## Getting Started

### Running the Model
To see available options for running the model, use:
```bash
python3 evaluate.py -h
```
### Training the Model
To see available options for training the model, use:
```bash
python3 train.py -h
```

### Additional Scripts
There are additional scripts in the additional scripts folder to generate a scaled down version of any dataset incase you want to train the model on your own dataset.

The pre-trained weights for both the original and adaptive model can be found at https://drive.google.com/drive/folders/1TTUqeYsLSTWpvjYGmhuBkVHEIC050LsP?usp=sharing

### Dataset images 
The Set14 images with x2 version is available here https://drive.google.com/drive/folders/1misaxqZ2j23aHipXRLAnWiaR8sK3c0U5?usp=sharing

### requirement.txt and requirement_1.txt
Since pip may not provide the necessary requirements as this is done in a conda environment, a requirement.txt conda file is also present
```bash
pip install requirement_1.txt
```
or using conda 
```bash
conda create --name my_env --file requirement.txt
```
```bash
conda activate my_env
```

