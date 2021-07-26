# WearMask3D

This repository contains code for fitting masks to face images in the wild by utilizing the 3D morphable model. 
The code is provided without any warranty. If using this code, please cite our work as shown below.
```
@inproceedings{wearmask3d, 
    author = {Hong*, Je Hyeong and Kim*, Hanjo and Kim, Minsoo and Nam, Gi Pyo and Cho, Junghyun and Ko, Hyeong-Seok and Kim, Ig-Jae}, 
    booktitle = {2021 IEEE Conference on Image Processing (ICIP)}, 
    title = {A {3D} model-based approach for fitting masks to faces in the wild}, 
    year = {2021}, 
}
```

## How to run WearMask3D
1. Download the dlib landmark pre-trained model from Jianzhu Guo's [Google Drive](https://drive.google.com/file/d/1kxgOZSds1HuUIlvo5sRH3PJv377qZAkE/view)
and store the file in the **models** directory.

2. Install the requirements listed in **requirements.txt**, e.g. by typing
```
pip install -r requirements.txt**
```

3. Run the demo code by running
```
python main.py
```

### Parameters
Below is a list of tunable hyperparameters (coming soon).


## Masked Faces in the Wild (MFW-mini)
You can find a list of images [here](https://docs.google.com/spreadsheets/d/1iooymtDPA8k2KUbB5K1jnuay4Eq-aRK4Mo4FbxbARqg/edit?usp=sharing)
If you wish to obtain a cropped copy of the images, please email *kface-at-imrc.kist.re.kr*.

## Bug report
Please raise an issue on Github for issues related to this code.
