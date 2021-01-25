# Setup instructions for TensorFlow

## Notebooks vs integrated development environments
Notebooks (Jupyter, iPython, etc) are suitable for presenting results or for smaller projects. 
If you implement something of a magnitude beyond your coursework, you may write your code in an integrated development environment (IDE) (e.g. Pycharm and Spyder) which has better debugging tools and makes coding much more enjoyable.

In this course, you can use either. In addition, we will also be using Python notebooks in [Google Colaboratory](https://colab.research.google.com) which is a cloud service for machine learning research. The reasons why we choose it as an additonal tool are as following:

* TensorFlow is set up by default
* Google Colab provides a free GPU/TPU. 

However, we still encourage you to set up everything on your local machine.

The instructions below are for MacOS, Linux and Windows users. Note that setting up TensorFlow using Anaconda or virtualenv is highly recommended. You can follow the official instructions to install Anaconda [here](https://docs.anaconda.com/anaconda/install/) and Pycharm [here](https://www.jetbrains.com/help/pycharm/installation-guide.html) (Community version is fine).


## Python environment and IDE setup with Anaconda

After Anaconda is installed, open Anaconda prompt and check your Python version:
```
python --version
```
and update the base Python install:
```
conda update conda
conda update anaconda
conda update python
conda update --all
```

Then we create a new virtual environment by choosing a Python interpreter and making a ./venv directory to hold it using command line:
```
conda create -n venv pip python=3.6 
```
Note that the number should be your current Python version. After the update, you can run the first command line again to check.

Next, activate the virtual environment. 

For **Windows users**,
```
activate venv 
```
or for **macOS/Linux users**,
```
source activate venv
```
And to exit the environment:
```
deactivate venv
```
or 
```
source deactivate venv
```

## Install TensorFlow

### Hardware requirements for GPU version
There are a CPU version and a GPU version of TensorFlow. The GPU version only supports NVIDIA® GPU card with CUDA® Compute Capability 3.5 or higher.  Before installing, you will need to check your GPU card. Generally speaking, if you are installing on your laptop, we recommend you to install the CPU version. If you've got one of [these GPU cards](https://developer.nvidia.com/cuda-gpus), feel free to install the GPU version.

### Installation

Firstly, activate the virtual environment. Within the virtual environment, install TensorFlow using the following command line and the correct URL from [this list](https://www.tensorflow.org/install/pip#package-location):
```
(venv) pip install --ignore-installed --upgrade URL
```
Make sure the URL matches your Python version, operating system as well as the CPU or GPU version of the TensorFlow chosen. 

Upgrade TensorRlow after installation:
```
(venv) pip install --upgrade tensorflow
```
Check if TensorFlow is installed:
```
(venv) python -c "import tensorflow as tf; print(tf.__version__)"
```
If your TensorFlow is installed successfully, you should be able to see this:

![alt text](https://github.com/lse-st449/lectures/blob/master/Week01/Class/graphs/version.png)

## Configuring Conda environment in Pycharm

Open your Pycharm and follow the official instructions [here](https://www.jetbrains.com/help/pycharm/conda-support-creating-conda-virtual-environment.html).

If your Pycharm fails to figure out where the location of Anaconda or Conda executable is, open your `cmd.exe` and type:
```
where conda
```
and find the location for `conda.exe`, e.g., if the location is `C:\Anaconda3\Scripts\conda.exe`, set path using command:
```
set PATH=%PATH%;C:\Anaconda3;C:\Anaconda3\Scripts\
```
and now try to configure your Conda environment in Pycharm again. 
