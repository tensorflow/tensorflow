#!/bin/bash
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

set -e
set -x

source tensorflow/tools/ci_build/release/common.sh

# Strip leading and trailing whitespaces
str_strip () {
  echo -e "$1" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'
}

# Clean up bazel build & test flags with proper configuration.
update_bazel_flags() {
  # Add git tag override flag if necessary.
  GIT_TAG_STR=" --action_env=GIT_TAG_OVERRIDE"
  if [[ -z "${GIT_TAG_OVERRIDE}" ]] && \
    ! [[ ${TF_BUILD_FLAGS} = *${GIT_TAG_STR}* ]]; then
    TF_BUILD_FLAGS+="${GIT_TAG_STR}"
  fi
  # Clean up whitespaces
  TF_BUILD_FLAGS=$(str_strip "${TF_BUILD_FLAGS}")
  TF_TEST_FLAGS=$(str_strip "${TF_TEST_FLAGS}")
  # Cleaned bazel flags
  echo "Bazel build flags (cleaned):\n" "${TF_BUILD_FLAGS}"
  echo "Bazel test flags (cleaned):\n" "${TF_TEST_FLAGS}"
}

DEFAULT_PROJECT_NAME="tensorflow"
DEFAULT_AUDITWHEEL_TARGET_PLAT="manylinux2014"
PROJECT_NAME=${TF_PROJECT_NAME:-$DEFAULT_PROJECT_NAME}
AUDITWHEEL_TARGET_PLAT=${TF_AUDITWHEEL_TARGET_PLAT:-$DEFAULT_AUDITWHEEL_TARGET_PLAT}

sudo install -o ${CI_BUILD_USER} -g ${CI_BUILD_GROUP} -d /tmpfs
sudo install -o ${CI_BUILD_USER} -g ${CI_BUILD_GROUP} -d /tensorflow
sudo chown -R ${CI_BUILD_USER}:${CI_BUILD_GROUP} /usr/local/lib/python*
sudo chown -R ${CI_BUILD_USER}:${CI_BUILD_GROUP} /usr/local/bin
sudo chown -R ${CI_BUILD_USER}:${CI_BUILD_GROUP} /usr/lib/python3/dist-packages

# Update bazel
install_bazelisk

# Need to update the version of auditwheel used for aarch64
python3 -m pip install auditwheel~=5.3.0

# Need to use the python from the venv
export PYTHON_BIN_PATH=$(which python3)

# Env vars used to avoid interactive elements of the build.
export HOST_C_COMPILER=(which gcc)
export HOST_CXX_COMPILER=(which g++)
export TF_ENABLE_XLA=1
export TF_DOWNLOAD_CLANG=0
export TF_SET_ANDROID_WORKSPACE=0
export TF_NEED_MPI=0
export TF_NEED_ROCM=0
export TF_NEED_GCP=0
export TF_NEED_S3=0
export TF_NEED_OPENCL_SYCL=0
export TF_NEED_CUDA=0
export TF_NEED_HDFS=0
export TF_NEED_OPENCL=0
export TF_NEED_JEMALLOC=1
export TF_NEED_VERBS=0
export TF_NEED_AWS=0
export TF_NEED_GDR=0
export TF_NEED_OPENCL_SYCL=0
export TF_NEED_COMPUTECPP=0
export TF_NEED_KAFKA=0
export TF_NEED_TENSORRT=0

# Export required variables for running the tests
export OS_TYPE="UBUNTU"
export CONTAINER_TYPE="CPU"

# Get the default test targets for bazel
source tensorflow/tools/ci_build/build_scripts/DEFAULT_TEST_TARGETS.sh

# Get the extended skip test list for arm
source tensorflow/tools/ci_build/build_scripts/ARM_SKIP_TESTS_EXTENDED.sh

# Export optional variables for running the tests
export TF_BUILD_FLAGS="--config=mkl_aarch64_threadpool --copt=-flax-vector-conversions"
export TF_TEST_FLAGS="${TF_BUILD_FLAGS} \
    --test_env=TF_ENABLE_ONEDNN_OPTS=1 --test_env=TF2_BEHAVIOR=1 --define=tf_api_version=2 \
    --test_lang_filters=py --flaky_test_attempts=3 --test_size_filters=small,medium \
    --test_output=errors --verbose_failures=true --test_keep_going --notest_verbose_timeout_warnings"
export TF_TEST_TARGETS="${DEFAULT_BAZEL_TARGETS} ${ARM_SKIP_TESTS}"
export TF_FILTER_TAGS="-no_oss,-oss_excluded,-oss_serial,-v1only,-benchmark-test,-no_aarch64,-gpu,-tpu,-no_oss_py39,-no_oss_py310"
export TF_AUDITWHEEL_TARGET_PLAT="manylinux2014"

if [ ${IS_NIGHTLY} == 1 ]; then
  ./tensorflow/tools/ci_build/update_version.py --nightly
fi

sudo sed -i '/^build --profile/d' /usertools/aarch64.bazelrc
sudo sed -i '\@^build.*=\"/usr/local/bin/python3\"$@d' /usertools/aarch64.bazelrc
sudo sed -i '/^build --profile/d' /usertools/aarch64_clang.bazelrc
sudo sed -i '\@^build.*=\"/usr/local/bin/python3\"$@d' /usertools/aarch64_clang.bazelrc
sed -i '$ aimport /usertools/aarch64_clang.bazelrc' .bazelrc

# Local variables
WHL_DIR="${KOKORO_ARTIFACTS_DIR}/tensorflow/whl"
sudo install -o ${CI_BUILD_USER} -g ${CI_BUILD_GROUP} -d ${WHL_DIR}
WHL_DIR=$(realpath "${WHL_DIR}") # Get absolute path

# configure may have chosen the wrong setting for PYTHON_LIB_PATH so
# determine here the correct setting
PY_SITE_PACKAGES=$(${PYTHON_BIN_PATH} -c "import site ; print(site.getsitepackages()[0])")

# Determine the major.minor versions of python being used (e.g., 3.7).
# Useful for determining the directory of the local pip installation.
PY_MAJOR_MINOR_VER=$(${PYTHON_BIN_PATH} -c "print(__import__('sys').version)" 2>&1 | awk '{ print $1 }' | head -n 1 | cut -d. -f1-2)

update_bazel_flags

bazel build \
    --action_env=PYTHON_BIN_PATH=${PYTHON_BIN_PATH} \
    --action_env=PYTHON_LIB_PATH=${PY_SITE_PACKAGES} \
    ${TF_BUILD_FLAGS} \
    --repo_env=WHEEL_NAME=${PROJECT_NAME} \
    //tensorflow/tools/pip_package:wheel \
    || die "Error: Bazel build failed for target: //tensorflow/tools/pip_package:wheel"

find ./bazel-bin/tensorflow/tools/pip_package -iname "*.whl" -exec cp {} $WHL_DIR \;

PY_DOTLESS_MAJOR_MINOR_VER=$(echo $PY_MAJOR_MINOR_VER | tr -d '.')
if [[ $PY_DOTLESS_MAJOR_MINOR_VER == "2" ]]; then
  PY_DOTLESS_MAJOR_MINOR_VER="27"
fi

# Set wheel path and verify that there is only one .whl file in the path.
WHL_PATH=$(ls "${WHL_DIR}"/"${PROJECT_NAME}"-*"${PY_DOTLESS_MAJOR_MINOR_VER}"*"${PY_DOTLESS_MAJOR_MINOR_VER}"*.whl)
if [[ $(echo "${WHL_PATH}" | wc -w) -ne 1 ]]; then
  echo "ERROR: Failed to find exactly one built TensorFlow .whl file in "\
  "directory: ${WHL_DIR}"
fi

# Print the size of the wheel file and log to sponge.
WHL_SIZE=$(ls -l ${WHL_PATH} | awk '{print $5}')
echo "Size of the PIP wheel file built: ${WHL_SIZE}"

# Repair the wheels for manylinux2014
echo "auditwheel repairing ${WHL_PATH}"
auditwheel repair --plat ${AUDITWHEEL_TARGET_PLAT}_$(uname -m) -w "${WHL_DIR}" "${WHL_PATH}"

if [[ $(ls ${WHL_DIR} | grep ${AUDITWHEEL_TARGET_PLAT} | wc -l) == 1 ]] ; then
  WHL_PATH=${WHL_DIR}/$(ls ${WHL_DIR} | grep ${AUDITWHEEL_TARGET_PLAT})
  echo "Repaired ${AUDITWHEEL_TARGET_PLAT} wheel file at: ${WHL_PATH}"
else
  die "WARNING: Cannot find repaired wheel."
fi

start-stop-daemon -b -n portserver.py -a /usr/local/bin/python3 -S -- /usr/local/bin/portserver.py

bazel test ${TF_TEST_FLAGS} \
    --action_env=PYTHON_BIN_PATH=${PYTHON_BIN_PATH} \
    --action_env=PYTHON_LIB_PATH=${PY_SITE_PACKAGES} \
    --test_env=PORTSERVER_ADDRESS=@unittest-portserver \
    --build_tag_filters=${TF_FILTER_TAGS} \
    --test_tag_filters=${TF_FILTER_TAGS} \
    --local_test_jobs=$(grep -c ^processor /proc/cpuinfo) \
    --build_tests_only \
    -- ${TF_TEST_TARGETS}

# remove duplicate wheel and copy wheel to mounted volume for local access
rm -rf ${WHL_DIR}/*linux_aarch64.whl && cp -r ${WHL_DIR} .
