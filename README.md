# Showcase: Deep learning in the cloud. Building a handwritten character recognition model with [GNU R](https://www.r-project.org/) and [MXNet](http://mxnet.io/) with [Amazon Web Services](https://aws.amazon.com/) on [GPU instances](http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using_cluster_computing.html).

## Business story behind the showcase

Optical character recognition (OCR) for handwritten text is applicable in areas where several documents are being used in business for processing large amounts of paper documents. One example could be a traditional post operator that wants to automatically process information on the envelopes. The envelopes are usually not neatly adressed and standard OCR solutions fail in this area. A custom-made character recognition engine designed for this task might receive much better quality level.

## Approach
The showcase presents two main things:

* Construction of deep laerning models with MXNet library
* How to run GPU computing in the cloud with Amazon Web Services



## Prerequisites

We use bitfusion.io [Scientific Computing AMI](https://aws.amazon.com/marketplace/seller-profile?id=3b372560-86bf-4e3d-9ec0-016892a64bed)

The AMI contains Ubuntu 14  along with a R installation along with CUDA drivers.
Additionally we have installed MXNet running the following commands:

* `sudo apt-get update`
* `sudo apt-get install -y build-essential git libblas-dev libopencv-dev`
* `git clone --recursive https://github.com/dmlc/mxnet`

Next modify config.mk by setting the following keys:

     USE_CUDA = 1
     USE_CUDA_PATH = /usr/local/cuda
     USE_BLAS = atlas
     
Finally, compile mxnet with the command `make –j4`

## Usage instruction

1. Install R packages `01_install_packages.R`
2. Prepare the dataset `02_download_datasets.R`
3. Declare layers for the deep neural network `03_declare_mlp_model.R`
4. Create data iterators for seuential data reading `04_prepare_data_iterators.R`
5. Fit the model to the data `05_fit_mlp_model.R`
6. If the fitting process is interrupted a script for resuming computation state can be used: `06_restart_mlp_model.R`
7. Perform predictions and observe the results `07_predict_mlp_model.R`

## What next

This example presents one possible usage of deep learning models for classification of images. 
One important problem is selection of an optimal structure for a deep neural network. 
This reuires execution of several experiments for measuring predictive capabilities for various network topologies. 
Anazon Web Services comes forward to this need and offers very large GPU instances. The flagship offering is a p2.16xlarge offering 16 x GPU Nvidia TESLA K80 with a total of 80'000 GPU cores. This machine availabe from around $2.10 on AWS spot market would make it to the list of Top Supercomputers just 10 years ago.





