# Recognition-of-emotions-using-CNN
Student project of e-Health system. Real-time recognition of emotions and pulse detection from webcam video.
More information about this project is in file `scientific_paper_about_project.pdf`. Till now it is only written in polish language.
## Dataset
We are not allowed to publish FER2013 dataset.
If you want to train model then you should download it on your own from [Kaggle](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data).
Download `fer2013.tar.gz` from this website and decompress `fer2013.csv` in the `./training/dataset` folder.

## Requirements
To install all dependecies you will need pip.
```bash
pip3 install -r requirements.txt
```
## Usage
These are the commands that will allow you to start the program. The first one is used to train the model. You will need to download the FER2013 dataset before training. The second command is used to run an application used to detect pulse and emotions from the webcam image. Run it from the directory
of this project.
```bash
# To train model
$ python3 -m training.train_model.py
# To use the app
$ python3 -m  application.app.py
```
