# Bathymetry Recovering

## Setup

- Install python3
- Install [tensorflow](https://www.tensorflow.org/install/) (tested on Release 1.3.0, 1.4.0, 1.5.0, 1.6.0, 1.7.0)
- Install [neuralgym](https://github.com/htn274/neuralgym) toolkit (run ``pip install git+https://github.com/htn274/neuralgym``)
- Clone project: ``git clone https://github.com/Hydroviet/bathymetry_recover.git && cd bathymetry_recover``
- Install other dependencies: ``pip install -r requirements.txt``

## Training

## Testing with pretrainned model

## TensorBoard

Visualization on [TensorBoard](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard) for training and validation is supported. Run `tensorboard --logdir model_logs --port 6006` to view training progress.

## License

CC 4.0 Attribution-NonCommercial International

The software is for educational and academic research purposes only.

## Acknowledgements

We adapted the GitHub repository generative_inpainting to the setting of Digital Elevation Models. 
