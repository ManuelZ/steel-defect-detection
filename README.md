## Use of Mask-RCNN in the Steel Defect Competition Kaggle dataset

https://www.kaggle.com/c/severstal-steel-defect-detection



## How to use Mask-RCNN with Tensorflow 2 on Windows 10
Mask-RCNN is only available on Tensorflow 1, to use Tensorflow 2, follow the instructions here:

https://www.immersivelimit.com/tutorials/mask-rcnn-for-windows-10-tensorflow-2-cuda-101

It's basically cloning a fork —created by [Adam Kelly](https://github.com/akTwelve)— of the original Mask-RCNN repository. It works perfectly on Tensorflow 2.3 with Cuda 10.1 on Windows 10.


## How to use Mask-RCNN with Tensorflow 2 on Linux
Same as on Windows. But gotta modify the code in model.py, in train(), when calling keras_model.fit and set workers to zero (one also works). 


## How to start a [Deep Learning container](https://github.com/ManuelZ/Deep-Learning-Docker)

    sudo docker create --gpus all -it --volume /home/user/steel-defect-detection:/root/steel-defect-detection --name dl-container dl

    sudo docker container start --interactive dl-container


