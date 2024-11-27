# TensorFlow ROCm port: Building From Source

## Intro

This instruction provides a starting point for build TensorFlow ROCm port from source.
*Note*: it is recommended to start with a clean Ubuntu 18.04 system

## Install ROCm

Follow steps at [Basic Installation](https://github.com/ROCmSoftwarePlatform/tensorflow-upstream/blob/develop-upstream/rocm_docs/tensorflow-install-basic.md#install-rocm) to install ROCm stack.
*NOTE*: ROCm install instructions recommend a purge and reinstall of ROCm rather than upgrading from previous release.
For details of the ROCm instructions, please refer to the [ROCm QuickStart Installation Guide](https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html).

To build with ROCm3.10, set the following environment variables, and add those environment variables at the end of ~/.bashrc 
```
export ROCM_PATH=/opt/rocm/
export HCC_HOME=$ROCM_PATH/hcc
export HIP_PATH=$ROCM_PATH/hip
export PATH=$HCC_HOME/bin:$HIP_PATH/bin:$PATH
export ROCM_TOOLKIT_PATH=$ROCM_PATH
```

## Install required python packages

Install the following python dependencies:
```
sudo apt-get update && sudo apt-get install -y \
    python-dev \
    python-pip \
    python-wheel \
    python3-numpy \
    python3-dev \
    python3-wheel \
    python3-mock \
    python3-future \
    python3-pip \
    python3-yaml \
    python3-setuptools && \
    sudo apt-get clean

pip3 install keras_preprocessing setuptools keras_applications jupyter --upgrade
pip3 install numpy==1.18.5
```

## Build Tensorflow 1.15 release
### Install bazel

```
echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
sudo apt-get update && sudo apt-get install -y openjdk-8-jdk openjdk-8-jre unzip wget git
cd ~/ && wget https://github.com/bazelbuild/bazel/releases/download/0.24.1/bazel-0.24.1-installer-linux-x86_64.sh
sudo bash ~/bazel*.sh
```

### Build TensorFlow 1.15 ROCm backend 

```
# Clone tensorflow source code 
cd ~ && git clone -b r1.15-rocm-enhanced https://github.com/ROCmSoftwarePlatform/tensorflow-upstream.git tensorflow

# Python 3: Build and install TensorFlow ROCm port pip3 package
cd ~/tensorflow && ./build_rocm_python3
```

## Build Tensorflow 2.3 release
### Install bazel

```
echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
sudo apt-get update && sudo apt-get install -y openjdk-8-jdk openjdk-8-jre unzip wget git
cd ~ && rm -rf bazel*.sh && wget https://github.com/bazelbuild/bazel/releases/download/3.1.0/bazel-3.1.0-installer-linux-x86_64.sh  && bash bazel*.sh && rm -rf ~/*.sh
```

### Build TensorFlow 2.3 ROCm backend

```
# Clone tensorflow source code 
cd ~ && git clone -b r2.3-rocm-enhanced https://github.com/ROCmSoftwarePlatform/tensorflow-upstream.git tensorflow

# Build and install TensorFlow ROCm port pip3 package
cd ~/tensorflow && ./build_rocm_python3
```
