# [Joint Self-Supervised Video Alignment and Action Segmentation (ICCV 2025)](https://arxiv.org/abs/2503.16832)

This repository contains the video alignment model (VAOT) only.

If you use the code, please cite our paper:
```
@inproceedings{ali2025joint,
  title={Joint Self-Supervised Video Alignment and Action Segmentation},
  author={Ali, Ali Shah and Mahmood, Syed Ahmed and Saeed, Mubin and Konin, Andrey and Zia, M Zeeshan and Tran, Quoc-Huy},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  year={2025}
}
```

For our recent works, please check out our research page (https://retrocausal.ai/research/).


## Installation
Create an environment and install required packages
```
conda env create --name VAOT --file=vaot_env.yml
conda activate VAOT
```

If you face any pytorch related issues during training, uninstall the pytorch first
```
pip3 uninstall torch torchvision torchaudio
```

Go to https://pytorch.org/get-started/locally/ and install the suitable pytorch as per you machine requirements.
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```


## Video-to-Frame Conversion
```
python video_to_frames.py videos/
```


## Training/Testing Splits
Split your data into train and test and your directory should look like this
```
$YOUR_PATH_TO_DATASET
    ├─train
        ├──vid1/
        |   ├──000001.jpg
        |   ├──000002.jpg
        |   ├──...
    ├──val
        ├──vid2/
        |   ├──000001.jpg
        |   ├──000002.jpg
        |   ├──...
        ├──...
```


## Training
```
python train.py
```


## Testing
```
python evaluations.py --model_path path/to/model --dest path/to/log/dest --device 0
```

The expected structure of evaluation is like this:
```
├──<PATH>
    ├──test
        ├──vid2
        |   ├──000001.jpg
        |   ├──000002.jpg
        |   ├──...
```
