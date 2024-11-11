# Project for ECE570: Artificial Intelligence

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