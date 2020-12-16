## Use of Mask-RCNN in the Steel Defect Competition Kaggle dataset

https://www.kaggle.com/c/severstal-steel-defect-detection



## How to use Mask-RCNN with Tensorflow 2 on Windows 10
The [original Mask-RCNN](https://github.com/matterport/Mask_RCNN) is only available on Tensorflow 1. To use Tensorflow 2, follow these instructions:

    https://www.immersivelimit.com/tutorials/mask-rcnn-for-windows-10-tensorflow-2-cuda-101

It's basically cloning a fork —created by [Adam Kelly](https://github.com/akTwelve)— of the original Mask-RCNN repository. It works perfectly on Tensorflow 2.3 with Cuda 10.1 on Windows 10. But there are some small details to change in the `requirements.txt` file:

Clone the repository:

    git clone https://github.com/akTwelve/Mask_RCNN.git aktwelve_mask_rcnn

Change into the directory:

    cd aktwelve_mask_rcnn

Modify the `requirements.txt` file:
  - If your current tensorflow installation is `tensorflow-gpu`, change the requirements file to that, otherwise you would end up installing the Tensorflow CPU version.
  - Specify version `0.16.2` for the package `scikit-image`, otherwise you will end up with lots of useless warnings.

Install dependencies:

    pip install -r requirements.txt

## How to use Mask-RCNN with Tensorflow 2 on Linux
Same as on Windows. But gotta modify the code in `model.py`, in `train()`, when calling `keras_model.fit` and set workers to `0` (`1` also works). 


## How to start a [Deep Learning container](https://github.com/ManuelZ/Deep-Learning-Docker)

    sudo docker create --gpus all -it --volume /home/user/steel-defect-detection:/root/steel-defect-detection -p 8888:8888 --name dl-container dl

    sudo docker container start --interactive dl-container

## How to run Tensorboard

### Allow connections through the firewall (in case it's active)  
Read [this answer](https://stackoverflow.com/a/54064225)

    sudo ufw allow out on docker0 from 172.17.0.0/16

### Actually run Tensorboard

    sudo docker run -d -p 6006:6006 -v ~/steel-defect-detection/data:/root/steel-defect-detection/data tensorflow/tensorflow:2.3.1-gpu tensorboard --bind_all --logdir /root/steel-defect-detection/data    


## How to run Jupyter

    jupyter notebook --allow-root --ip 0.0.0.0 

    
