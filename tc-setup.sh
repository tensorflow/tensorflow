#!/bin/bash

set -ex

source $(dirname $0)/tc-vars.sh

if [ "${OS}" = "Linux" ]; then
    SHA_SUM="sha256sum -c --strict"
    WGET=/usr/bin/wget
elif [ "${OS}" = "Darwin" ]; then
    SHA_SUM="shasum -a 256 -c"
    WGET=wget
fi;

install_cuda=
if [ "$1" = "--cuda" ]; then
    install_cuda=yes
fi

# $1 url
# $2 sha256
download()
{
    fname=`basename $1`

    ${WGET} $1 -O ${DS_ROOT_TASK}/dls/$fname && echo "$2  ${DS_ROOT_TASK}/dls/$fname" | ${SHA_SUM} -
}

# Download stuff
mkdir -p ${DS_ROOT_TASK}/dls || true
download $BAZEL_URL $BAZEL_SHA256

if [ ! -z "${install_cuda}" ]; then
    download $CUDA_URL $CUDA_SHA256
    download $CUDNN_URL $CUDNN_SHA256
fi;

# For debug
ls -hal ${DS_ROOT_TASK}/dls/

# Install Bazel in ${DS_ROOT_TASK}/bin
BAZEL_INSTALL_FILENAME=$(basename "${BAZEL_URL}")
if [ "${OS}" = "Linux" ]; then
    BAZEL_INSTALL_FLAGS="--user"
elif [ "${OS}" = "Darwin" ]; then
    BAZEL_INSTALL_FLAGS="--bin=${DS_ROOT_TASK}/bin --base=${DS_ROOT_TASK}/.bazel"
fi;
mkdir -p ${DS_ROOT_TASK}/bin || true
pushd ${DS_ROOT_TASK}/bin
    /bin/bash ${DS_ROOT_TASK}/dls/${BAZEL_INSTALL_FILENAME} ${BAZEL_INSTALL_FLAGS}
popd

# For debug
bazel version

if [ ! -z "${install_cuda}" ]; then
    # Install CUDA and CuDNN
    mkdir -p ${DS_ROOT_TASK}/DeepSpeech/CUDA/ || true
    pushd ${DS_ROOT_TASK}
        CUDA_FILE=`basename ${CUDA_URL}`
        PERL5LIB=. sh ${DS_ROOT_TASK}/dls/${CUDA_FILE} --silent --verbose --override --toolkit --toolkitpath=${DS_ROOT_TASK}/DeepSpeech/CUDA/

        CUDNN_FILE=`basename ${CUDNN_URL}`
        tar xvf ${DS_ROOT_TASK}/dls/${CUDNN_FILE} --strip-components=1 -C ${DS_ROOT_TASK}/DeepSpeech/CUDA/
    popd

    LD_LIBRARY_PATH=${DS_ROOT_TASK}/DeepSpeech/CUDA/lib64/:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH
else
    echo "No CUDA/CuDNN to install"
fi

# Configure Python virtualenv
pushd ${DS_ROOT_TASK}/DeepSpeech/
    TF_VENV=tf-venv
    if [ "${OS}" = "Linux" ]; then
        /usr/bin/virtualenv ${TF_VENV}/
        DS_PIP=./${TF_VENV}/bin/pip
    elif [ "${OS}" = "Darwin" ]; then
        pyenv install ${PYENV_VERSION}
        pyenv virtualenv ${PYENV_VERSION} ${TF_VENV}
        DS_PIP=${PYENV_ROOT}/versions/${PYENV_VERSION}/envs/${TF_VENV}/bin/pip
    fi;

    ${DS_PIP} install numpy scipy python_speech_features wheel
popd

mkdir -p ${TASKCLUSTER_ARTIFACTS} || true
