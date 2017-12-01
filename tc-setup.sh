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

### Copied from DeepSpeech repo
install_pyenv()
{
  if [ -z "${PYENV_ROOT}" ]; then
    echo "No PYENV_ROOT set";
    exit 1;
  fi;

  git clone --quiet https://github.com/pyenv/pyenv.git ${PYENV_ROOT}
  pushd ${PYENV_ROOT}
    git checkout --quiet 0c909f7457a027276a1d733d78bfbe70ba652047
  popd
  eval "$(pyenv init -)"
}

install_pyenv_virtualenv()
{
  PYENV_VENV=$1

  if [ -z "${PYENV_VENV}" ]; then
    echo "No PYENV_VENV set";
    exit 1;
  fi;

  git clone --quiet https://github.com/pyenv/pyenv-virtualenv.git ${PYENV_VENV}
  pushd ${PYENV_VENV}
      git checkout --quiet 27270877575fe8c3e7be5385b8b6a1e4089b39aa
  popd
  eval "$(pyenv virtualenv-init -)"
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

    LD_LIBRARY_PATH=${DS_ROOT_TASK}/DeepSpeech/CUDA/lib64/:${DS_ROOT_TASK}/DeepSpeech/CUDA/lib64/stubs/:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH

    # We might lack libcuda.so.1 symlink, let's fix as upstream does:
    # https://github.com/tensorflow/tensorflow/pull/13811/files?diff=split#diff-2352449eb75e66016e97a591d3f0f43dR96
    if [ ! -h "${DS_ROOT_TASK}/DeepSpeech/CUDA/lib64/stubs/libcuda.so.1" ]; then
        ln -s "${DS_ROOT_TASK}/DeepSpeech/CUDA/lib64/stubs/libcuda.so" "${DS_ROOT_TASK}/DeepSpeech/CUDA/lib64/stubs/libcuda.so.1"
    fi;

else
    echo "No CUDA/CuDNN to install"
fi

# We don't produce Python package for RPi3, so don't give a Python version there
if [ ! -z "${PYENV_VERSION}" ]; then
    # Configure Python virtualenv
    unset PYTHON_BIN_PATH
    unset PYTHONPATH

    install_pyenv "${PYENV_ROOT}"
    install_pyenv_virtualenv "$(pyenv root)/plugins/pyenv-virtualenv"

    pyenv install ${PYENV_VERSION}
    pyenv virtualenv ${PYENV_VERSION} ${TF_VENV}

    source ${PYENV_ROOT}/versions/${PYENV_VERSION}/envs/${TF_VENV}/bin/activate

    pip install numpy scipy python_speech_features wheel
fi;

mkdir -p ${TASKCLUSTER_ARTIFACTS} || true
