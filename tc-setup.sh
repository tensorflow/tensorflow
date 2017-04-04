#!/bin/bash

set -ex

source $(dirname $0)/tc-vars.sh

install_cuda=
if [ "$1" = "--cuda" ]; then
    install_cuda=yes
fi

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
