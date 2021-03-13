#!/bin/bash
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Pip install TensorFlow and run basic test on the pip package.

set -e
set -x

# CPU size
MAC_CPU_MAX_WHL_SIZE=185M
LINUX_CPU_MAX_WHL_SIZE=153M
WIN_CPU_MAX_WHL_SIZE=113M
# GPU size
LINUX_GPU_MAX_WHL_SIZE=410M
WIN_GPU_MAX_WHL_SIZE=252M

function run_smoke_test() {

  # Upload the PIP package if whl test passes.
  if [ ${IN_VENV} -eq 0 ]; then
    VENV_TMP_DIR=$(mktemp -d)

    ${PYTHON_BIN_PATH} -m pip install virtualenv

    ${PYTHON_BIN_PATH} -m virtualenv -p ${PYTHON_BIN_PATH} "${VENV_TMP_DIR}" || \
        die "FAILED: Unable to create virtualenv"

    source "${VENV_TMP_DIR}/bin/activate" || \
        die "FAILED: Unable to activate virtualenv "
  fi

  # install tensorflow
  python -m pip install ${WHL_NAME} || \
      die "pip install (forcing to reinstall tensorflow) FAILED"
      echo "Successfully installed pip package ${WHL_NAME}"

  # Test TensorflowFlow imports
  test_tf_imports

  # Test TensorFlow whl file size
  test_tf_whl_size

  RESULT=$?

  # Upload the PIP package if whl test passes.
  if [ ${IN_VENV} -eq 0 ]; then
    # Deactivate from virtualenv.
    deactivate || source deactivate || die "FAILED: Unable to deactivate from existing virtualenv."
    sudo rm -rf "${KOKORO_GFILE_DIR}/venv"
  fi

  return $RESULT
}

function test_tf_imports() {
  TMP_DIR=$(mktemp -d)
  pushd "${TMP_DIR}"

  # test for basic import and perform tf.add operation.
  RET_VAL=$(python -c "import tensorflow as tf; t1=tf.constant([1,2,3,4]); t2=tf.constant([5,6,7,8]); print(tf.add(t1,t2).shape)")
  if ! [[ ${RET_VAL} == *'(4,)'* ]]; then
    echo "Unexpected return value: ${RET_VALUE}"
    echo "PIP test on virtualenv FAILED, will not upload ${WHL_NAME} package."
     return 1
  fi

  # test basic keras is available
  RET_VAL=$(python -c "import tensorflow as tf; print(tf.keras.__name__)")
  if ! [[ ${RET_VAL} == *'tensorflow.keras'* ]]; then
    echo "Unexpected return value: ${RET_VALUE}"
    echo "PIP test on virtualenv FAILED, will not upload ${WHL_NAME} package."
    return 1
  fi

  # similar test for estimator
  RET_VAL=$(python -c "import tensorflow as tf; print(tf.estimator.__name__)")
  if ! [[ ${RET_VAL} == *'tensorflow_estimator.python.estimator.api._v2.estimator'* ]]; then
    echo "Unexpected return value: ${RET_VALUE}"
    echo "PIP test on virtualenv FAILED, will not upload ${WHL_NAME} package."
    return 1
  fi

  RESULT=$?

  popd
  return $RESULT
}

function test_tf_whl_size() {
  # First, list all wheels with their sizes:
  echo "Found these wheels: "
  find $WHL_NAME -type f -exec ls -lh {} \;
  echo "===================="
  # Check CPU whl size.
  if [[ "$WHL_NAME" == *"_cpu"* ]]; then
    # Check MAC CPU whl size.
    if [[ "$WHL_NAME" == *"-macos"* ]] && [[ $(find $WHL_NAME -type f -size +${MAC_CPU_MAX_WHL_SIZE}) ]]; then
        echo "Mac CPU whl size has exceeded ${MAC_CPU_MAX_WHL_SIZE}. To keep
within pypi's CDN distribution limit, we must not exceed that threshold."
      return 1
    fi
    # Check Linux CPU whl size.
    if [[ "$WHL_NAME" == *"-manylinux"* ]] && [[ $(find $WHL_NAME -type f -size +${LINUX_CPU_MAX_WHL_SIZE}) ]]; then
        echo "Linux CPU whl size has exceeded ${LINUX_CPU_MAX_WHL_SIZE}. To keep
within pypi's CDN distribution limit, we must not exceed that threshold."
      return 1
    fi
    # Check Windows CPU whl size.
    if [[ "$WHL_NAME" == *"-win"* ]] && [[ $(find $WHL_NAME -type f -size +${WIN_CPU_MAX_WHL_SIZE}) ]]; then
        echo "Windows CPU whl size has exceeded ${WIN_CPU_MAX_WHL_SIZE}. To keep
within pypi's CDN distribution limit, we must not exceed that threshold."
      return 1
    fi
  # Check GPU whl size
  elif [[ "$WHL_NAME" == *"_gpu"* ]]; then
    # Check Linux GPU whl size.
    if [[ "$WHL_NAME" == *"-manylinux"* ]] && [[ $(find $WHL_NAME -type f -size +${LINUX_GPU_MAX_WHL_SIZE}) ]]; then
        echo "Linux GPU whl size has exceeded ${LINUX_GPU_MAX_WHL_SIZE}. To keep
within pypi's CDN distribution limit, we must not exceed that threshold."
      return 1
    fi
    # Check Windows GPU whl size.
    if [[ "$WHL_NAME" == *"-win"* ]] && [[ $(find $WHL_NAME -type f -size +${WIN_GPU_MAX_WHL_SIZE}) ]]; then
        echo "Windows GPU whl size has exceeded ${WIN_GPU_MAX_WHL_SIZE}. To keep
within pypi's CDN distribution limit, we must not exceed that threshold."
      return 1
    fi
  fi
}

###########################################################################
# Main
###########################################################################
if [[ -z "${1}" ]]; then
  echo "TF WHL path not given, unable to install and test."
  return 1
fi

IN_VENV=$(python -c 'import sys; print("1" if sys.version_info.major == 3 and sys.prefix != sys.base_prefix else "0")')
WHL_NAME=${1}
run_smoke_test
