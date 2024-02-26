# meme_classification
Classify whether an image is meme or not

**Model specification and training details:**
Model - DenseNet161
Train images - 2700
Validation Images - 900
Testing Images - 900
Epoch - 10
Optimizer - Adam
Learning Rate - 0.000001

Training Accuracy & Loss - 97.48% & 0.0779
Validation Accuracy & Loss - 97.33% & 0.0633

Testing Data (out of sample - unseen data):
  Accuracy - 96.88%
  Precision - 97.09%
  Recall - 96.67%

**Dataset details:**
Dataset has been collected through different kaggle datasets and google.
To create non-meme dataset, cartoon images and frames from movie clips are considered to match the same scenario as of actual memes.

**Inference**
python inference.py -path testing_data/meme_460.jpg -model modelv2.h5


**Google drive link for dataset and trained model**
Dataset: https://drive.google.com/file/d/1MOHsw7w-a4gxfUrT8e6aN_DUs1sn0Txj/view?usp=drive_link
Trained model: https://drive.google.com/file/d/1MOHsw7w-a4gxfUrT8e6aN_DUs1sn0Txj/view?usp=drive_link


model file = https://drive.google.com/file/d/1v71z1yUn8PxPD5ErufQBk5xxJZVbVjUo/view?usp=drive_link
data = 'https://drive.google.com/file/d/1MOHsw7w-a4gxfUrT8e6aN_DUs1sn0Txj/view?usp=drive_link'
