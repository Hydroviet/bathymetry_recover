# Bathymetry Recovering

This GitHub repository implements and evaluates the application of deep ipainting network [1] to extrapolate reservoir's bathymetry in my thesis 2020.

[1] J. Yu, Z. Lin, J. Yang, X. Shen, X. Lu, and T. S. Huang, Free-Form Image Inpainting with Gated Convolution, in The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018.

---

## Setup

- I recommend you to install by `conda`
- Create a virtual enviroment by conda with python 3.6

```bash
conda create -n <your-virtual-name-env> python==3.6.6
conda activate <your-virtual-name-env
```
- Install [tensorflow](https://www.tensorflow.org/install/) (tested on Release 1.3.0, 1.4.0, 1.5.0, 1.6.0, 1.7.0)
- Install [neuralgym](https://github.com/htn274/neuralgym) toolkit (run ``pip install git+https://github.com/htn274/neuralgym``)
- Clone project: ``git clone https://github.com/Hydroviet/bathymetry_recover.git && cd bathymetry_recover``
- Install other dependencies: ``pip install -r requirements.txt && pip install --upgrade numpy`` 

## Testing with pretrainned model

Download the desired model(s) [here](), create a `model_logs/` directory in `bathymetry_recover/` and extract the zip folder to there.

There are 2 ways to test:

- If you want to resize the input image, run:

```bash
python test.py --image data/aus_test/Hume.tif --mask data/aus_test/Hume_mask.png --output data/Hume_out.tif --checkpoint_dir model_logs/aus_128/
```
- If you want to split the input image into tiles, run:

```bash
python test.py --image data/aus_test/Hume.tif --mask data/aus_test/Hume_mask.png --output data/Hume_out.tif --checkpoint_dir model_logs/aus_128/ --num_tiles_x 2 --num_tiles_y 2
```

## Result

We compared the generated bathymetry with [texas data]() on 4 dams and get the best result (scored on RMSE) as following:

| Dams   |  Fairbairn | Hume | Eucumbene |  Burragorang |
|----------|------:|------:|------:|------:|
| Resize |  6.105376 | **7.126056** | **14.054130** | **22.074372** |
| Tiling | **4.555024** |  13.844635 | 26.814401 |   25.755498 |

To reproduce the result, run:

```bash
python3 test_dams.py --dams_list data/dams --input_dir data/aus_test/ --output_dir data/aus_test/ --checkpoint_dir model_logs/aus_128/
```

If you want to see the result of tiling input image, add `--tiling` to the end of the command.

## License

CC 4.0 Attribution-NonCommercial International

The software is for educational and academic research purposes only.

## Acknowledgements

We adapted the GitHub repository  [generative_inpainting](https://github.com/JiahuiYu/generative_inpainting) to the setting of Digital Elevation Models. 
