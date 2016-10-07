#!/bin/bash

set -ex

HOME_CLEAN=$(/usr/bin/realpath "${HOME}")

PATH=${HOME_CLEAN}/bin/:$PATH
export PATH

install_cuda=
if [ "$1" = "--cuda" ]; then
    install_cuda=yes
fi

BAZEL_URL=https://github.com/bazelbuild/bazel/releases/download/0.4.2/bazel-0.4.2-installer-linux-x86_64.sh
BAZEL_SHA256=b76b62a8c0eead1fc2215699382f1608c7bb98529fc48c5e9ef3dfa1b8b7585e

CUDA_URL=https://developer.nvidia.com/compute/cuda/8.0/prod/local_installers/cuda_8.0.44_linux-run
CUDA_SHA256=64dc4ab867261a0d690735c46d7cc9fc60d989da0d69dc04d1714e409cacbdf0

CUDNN_URL=http://developer.download.nvidia.com/compute/redist/cudnn/v5.1/cudnn-8.0-linux-x64-v5.1.tgz
CUDNN_SHA256=c10719b36f2dd6e9ddc63e3189affaa1a94d7d027e63b71c3f64d449ab0645ce

# $1 url
# $2 sha256
download()
{
    fname=`basename $1`

    /usr/bin/wget $1 -O ~/dls/$fname && echo "$2  ${HOME_CLEAN}/dls/$fname" | sha256sum -c --strict -
}

# Download stuff
mkdir -p ~/dls || true
download $BAZEL_URL $BAZEL_SHA256

if [ ! -z "${install_cuda}" ]; then
    download $CUDA_URL $CUDA_SHA256
    download $CUDNN_URL $CUDNN_SHA256
fi;

# For debug
ls -hal ~/dls/

# Install Bazel in ~/bin
mkdir -p ~/bin || true
pushd ~/bin
/bin/bash ~/dls/`basename "${BAZEL_URL}"` --user
popd

# For debug
bazel version

if [ ! -z "${install_cuda}" ]; then
    # Install CUDA and CuDNN
    mkdir -p ~/DeepSpeech/CUDA/ || true
    pushd ~
    CUDA_FILE=`basename ${CUDA_URL}`
    PERL5LIB=. sh ~/dls/${CUDA_FILE} --silent --verbose --override --toolkit --toolkitpath=${HOME_CLEAN}/DeepSpeech/CUDA/

    CUDNN_FILE=`basename ${CUDNN_URL}`
    tar xvf ~/dls/${CUDNN_FILE} --strip-components=1 -C ${HOME_CLEAN}/DeepSpeech/CUDA/
    popd

    LD_LIBRARY_PATH=${HOME_CLEAN}/DeepSpeech/CUDA/lib64/:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH

    ls -halR ~/DeepSpeech/CUDA/lib64/
else
    echo "No CUDA/CuDNN to install"
fi

# Configure Python virtualenv
pushd ~/DeepSpeech/
/usr/bin/virtualenv tf-venv && ./tf-venv/bin/pip install numpy scipy python_speech_features wheel
popd
