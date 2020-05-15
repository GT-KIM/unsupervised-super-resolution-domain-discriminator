# unsupervised-super-resolution-domain-discriminator
This is a project of CVPR2020 workshop paper "Unsupervised Real-World Super Resolution with Cycle Generative Adversarial Network and Domain Discriminator", which achieved 5th place in NTIRE2020 Real World Super Resolution Challenge Track 1
This code is based on tensorflow implementation of ESRGAN made by hiram64(github.com/hiram64/ESRGAN-tensorflow). Thanks you!

![image1](./image/example1.jpg)


## Dependencies
Python==3.5.2
Numpy==1.17.2
Scipy==1.2.0
OpenCV==3.4.4.19
Tensorflow-gpu==1.12.0


## Train
### Stage 1-1
python pre_train1.py
### Stage 1-2
python pre_train2.py
### Stage 2
python main_train.py

## Evaluate
### Track 1
python test.py --data_dir ./data/track1/ --checkpoint_dir ./checkpoint_track1/ --test_result_dir ./test_result_track1
### Track 2
python test.py --data_dir ./data/track2/ --checkpoint_dir ./checkpoint_track2/ --test_result_dir ./test_result_track2
