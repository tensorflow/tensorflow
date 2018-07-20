# TensorFlow ROCm port: Building From Source

## Intro
This instruction provides a starting point for build TensorFlow ROCm port from source.
*Note*: it is recommended to start with a clean Ubuntu 16.04 system

## Install ROCm

Follow steps at [Basic Installation](https://github.com/ROCmSoftwarePlatform/tensorflow-upstream/blob/develop-upstream/rocm_docs/tensorflow-install-basic.md#install-rocm) to install ROCm stack.

Setup environment variables, and add those environment variables at the end of ~/.bashrc 
```
export HCC_HOME=/opt/rocm/hcc
export HIP_PATH=/opt/rocm/hip
export PATH=$HCC_HOME/bin:$HIP_PATH/bin:$PATH
```

## Install required python packages

On Python 2-based systems:
```
sudo apt-get update && sudo apt-get install -y \
    python-numpy \
    python-dev \
    python-wheel \
    python-mock \
    python-future \
    python-pip \
    python-yaml \
    python-setuptools && \
    sudo apt-get clean && \
    sudo rm -rf /var/lib/apt/lists/*
```

On Python 3-based systems:
```
sudo apt-get update && sudo apt-get install -y \
    python3-numpy \
    python3-dev \
    python3-wheel \
    python3-mock \
    python3-future \
    python3-pip \
    python3-yaml \
    python3-setuptools && \
    sudo apt-get clean && \
    sudo rm -rf /var/lib/apt/lists/*
```

## Install bazel
```
echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
sudo apt-get update && sudo apt-get install -y openjdk-8-jdk openjdk-8-jre unzip && sudo apt-get clean && sudo rm -rf /var/lib/apt/lists/* 
cd ~/ && wget https://github.com/bazelbuild/bazel/releases/download/0.10.0/bazel-0.10.0-installer-linux-x86_64.sh 
sudo bash ~/bazel*.sh
```

## Build TensorFlow ROCm port
```
# Clone it
cd ~ && git clone -b r1.8-rocm https://github.com/ROCmSoftwarePlatform/tensorflow-upstream.git

# Configure TensorFlow ROCm port
# Enter all the way
cd ~/tensorflow && ./configure 

# Build and install TensorFlow ROCm port pip package
./build

# Build and install TensorFlow ROCm port pip3 package for Python 3-based systems
./build_python3
```
