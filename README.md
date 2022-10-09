# PlantClassification

## Requirements
The requirements for this project are given in `requirements.txt`. You will also need to install PyTorch which can be done by following the install instruction given in this [link](https://pytorch.org/get-started/locally/).

To install the other requirements:
```
pip3 install -r requirements.txt
```

## Train the Model
The model weights are given in `models/checkpoints/model_epoch_100.pkl`.
However if you want to train the model yourself you can run:
```python
python3 main.py
```

## Test the Model
You can test the model by running:
```python
python3 run_test_img.py
```
