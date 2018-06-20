#!/usr/bin/env bash
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
#
# Usage:
#   ci_parameterized_build.sh
#
# The script obeys the following required environment variables:
#   TF_BUILD_CONTAINER_TYPE:   (CPU | GPU | ANDROID | ANDROID_FULL)
#   TF_BUILD_PYTHON_VERSION:   (PYTHON2 | PYTHON3 | PYTHON3.5)
#   TF_BUILD_IS_PIP:           (NO_PIP | PIP | BOTH)
#
# The below environment variable is required, but will be deprecated together
# with TF_BUILD_MAVX and both will be replaced by TF_BUILD_OPTIONS.
#   TF_BUILD_IS_OPT:           (NO_OPT | OPT)
#
# Note:
#   1) Certain combinations of parameter values are regarded
# as invalid and will cause the script to exit with code 0. For example:
#   NO_OPT & PIP     (PIP builds should always use OPT)
#   ANDROID & PIP    (Android and PIP builds are mutually exclusive)
#
#   2) TF_BUILD_PYTHON_VERSION is set to PYTHON3, the build will use the version
# pointed to by "which python3" on the system, which is typically python3.4. To
# build for python3.5, set the environment variable to PYTHON3.5
#
#
# Additionally, the script follows the directions of optional environment
# variables:
#   TF_BUILD_DRY_RUN:  If it is set to any non-empty value that is not "0",
#                      the script will just generate and print the final
#                      command, but not actually run it.
#   TF_BUILD_APPEND_CI_DOCKER_EXTRA_PARAMS:
#                      String appended to the content of CI_DOCKER_EXTRA_PARAMS
#   TF_BUILD_APPEND_ARGUMENTS:
#                      Additional command line arguments for the bazel,
#                      pip.sh or android.sh command
#   TF_BUILD_MAVX:     (Soon to be deprecated, use TF_BUILD_OPTIONS instead)
#                      (unset | MAVX | MAVX2)
#                      If set to MAVX or MAVX2, will cause bazel to use the
#                      additional flag --copt=-mavx or --copt=-mavx2, to
#                      perform AVX or AVX2 builds, respectively. This requires
#                      AVX- or AVX2-compatible CPUs.
#   TF_BUILD_BAZEL_TARGET:
#                      Used to override the default bazel build target:
#                      //tensorflow/... -//tensorflow/compiler
#   TF_BUILD_BAZEL_CLEAN:
#                      Will perform "bazel clean", if and only if this variable
#                      is set to any non-empty and non-0 value
#   TF_GPU_COUNT:
#                      Run this many parallel tests for serial builds.
#                      For now, only can be edited for PIP builds.
#                      TODO(gunan): Find a way to pass this environment variable
#                      to the script bazel runs (using --run_under).
#   TF_BUILD_TEST_TUTORIALS:
#                      If set to any non-empty and non-0 value, will perform
#                      tutorials tests (Applicable only if TF_BUILD_IS_PIP is
#                      PIP or BOTH).
#                      See builds/test_tutorials.sh
#   TF_BUILD_INTEGRATION_TESTS:
#                      If set this will perform integration tests. See
#                      builds/integration_tests.sh.
#   TF_BUILD_RUN_BENCHMARKS:
#                      If set to any non-empty and non-0 value, will perform
#                      the benchmark tests (see *_logged_benchmark targets in
#                      tools/test/BUILD)
#   TF_BUILD_OPTIONS:
#                     (FASTBUILD | OPT | OPTDBG | MAVX | MAVX2_FMA | MAVX_DBG |
#                      MAVX2_FMA_DBG)
#                     Use the specified configurations when building.
#                     When set, overrides TF_BUILD_IS_OPT and TF_BUILD_MAVX
#                     options, as this will replace the two.
#   TF_SKIP_CONTRIB_TESTS:
#                     If set to any non-empty or non-0 value, will skipp running
#                     contrib tests.
#   TF_NIGHTLY:
#                     If this run is being used to build the tf_nightly pip
#                     packages.
#   TF_CUDA_CLANG:
#                     If set to 1, builds and runs cuda_clang configuration.
#                     Only available inside GPU containers.
#
# This script can be used by Jenkins parameterized / matrix builds.

# Helper function: Convert to lower case
to_lower () {
  echo "$1" | tr '[:upper:]' '[:lower:]'
}

# Helper function: Strip leading and trailing whitespaces
str_strip () {
  echo -e "$1" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'
}

# Helper function: Exit on failure
die () {
  echo $@
  exit 1
}

##########################################################
# Default configuration
CI_BUILD_DIR="tensorflow/tools/ci_build"

# Command to call when Docker is available
DOCKER_MAIN_CMD="${CI_BUILD_DIR}/ci_build.sh"
# Command to call when Docker is unavailable
NO_DOCKER_MAIN_CMD="${CI_BUILD_DIR}/builds/configured"

# Additional option flags to apply when Docker is unavailable (e.g., on Mac)
NO_DOCKER_OPT_FLAG="--genrule_strategy=standalone"

DO_DOCKER=1

BAZEL_CMD="bazel test"
BAZEL_BUILD_ONLY_CMD="bazel build"
BAZEL_CLEAN_CMD="bazel clean"

DEFAULT_BAZEL_CONFIGS="--config=gcp --config=hdfs"

PIP_CMD="${CI_BUILD_DIR}/builds/pip.sh"
PIP_TEST_TUTORIALS_FLAG="--test_tutorials"
PIP_INTEGRATION_TESTS_FLAG="--integration_tests"
ANDROID_CMD="${CI_BUILD_DIR}/builds/android.sh"
ANDROID_FULL_CMD="${CI_BUILD_DIR}/builds/android_full.sh"

TF_GPU_COUNT=${TF_GPU_COUNT:-8}
PARALLEL_GPU_TEST_CMD='//tensorflow/tools/ci_build/gpu_build:parallel_gpu_execute'

BENCHMARK_CMD="${CI_BUILD_DIR}/builds/benchmark.sh"

EXTRA_PARAMS=""
BAZEL_TARGET="//tensorflow/... -//tensorflow/compiler/..."

if [[ -n "$TF_SKIP_CONTRIB_TESTS" ]]; then
  BAZEL_TARGET="$BAZEL_TARGET -//tensorflow/contrib/..."
else
  BAZEL_TARGET="${BAZEL_TARGET} -//tensorflow/contrib/lite/..."
  BAZEL_TARGET="${BAZEL_TARGET} //tensorflow/contrib/lite:context_test"
  BAZEL_TARGET="${BAZEL_TARGET} //tensorflow/contrib/lite:framework"
  BAZEL_TARGET="${BAZEL_TARGET} //tensorflow/contrib/lite:interpreter_test"
  BAZEL_TARGET="${BAZEL_TARGET} //tensorflow/contrib/lite:model_test"
  BAZEL_TARGET="${BAZEL_TARGET} //tensorflow/contrib/lite/toco:toco"
  BAZEL_TARGET="${BAZEL_TARGET} //tensorflow/contrib/lite:simple_memory_arena_test"
  BAZEL_TARGET="${BAZEL_TARGET} //tensorflow/contrib/lite:string_util_test"
  BAZEL_TARGET="${BAZEL_TARGET} //tensorflow/contrib/lite/kernels:activations_test"
  BAZEL_TARGET="${BAZEL_TARGET} //tensorflow/contrib/lite/kernels:add_test"
  BAZEL_TARGET="${BAZEL_TARGET} //tensorflow/contrib/lite/kernels:basic_rnn_test"
  BAZEL_TARGET="${BAZEL_TARGET} //tensorflow/contrib/lite/kernels:concatenation_test"
  BAZEL_TARGET="${BAZEL_TARGET} //tensorflow/contrib/lite/kernels:conv_test"
  BAZEL_TARGET="${BAZEL_TARGET} //tensorflow/contrib/lite/kernels:depthwise_conv_test"
  BAZEL_TARGET="${BAZEL_TARGET} //tensorflow/contrib/lite/kernels:embedding_lookup_test"
  BAZEL_TARGET="${BAZEL_TARGET} //tensorflow/contrib/lite/kernels:embedding_lookup_sparse_test"
  BAZEL_TARGET="${BAZEL_TARGET} //tensorflow/contrib/lite/kernels:fully_connected_test"
  BAZEL_TARGET="${BAZEL_TARGET} //tensorflow/contrib/lite/kernels:hashtable_lookup_test"
  BAZEL_TARGET="${BAZEL_TARGET} //tensorflow/contrib/lite/kernels:local_response_norm_test"
  BAZEL_TARGET="${BAZEL_TARGET} //tensorflow/contrib/lite/kernels:lsh_projection_test"
  BAZEL_TARGET="${BAZEL_TARGET} //tensorflow/contrib/lite/kernels:lstm_test"
  BAZEL_TARGET="${BAZEL_TARGET} //tensorflow/contrib/lite/kernels:l2norm_test"
  BAZEL_TARGET="${BAZEL_TARGET} //tensorflow/contrib/lite/kernels:mul_test"
  BAZEL_TARGET="${BAZEL_TARGET} //tensorflow/contrib/lite/kernels:pooling_test"
  BAZEL_TARGET="${BAZEL_TARGET} //tensorflow/contrib/lite/kernels:reshape_test"
  BAZEL_TARGET="${BAZEL_TARGET} //tensorflow/contrib/lite/kernels:resize_bilinear_test"
  BAZEL_TARGET="${BAZEL_TARGET} //tensorflow/contrib/lite/kernels:skip_gram_test"
  BAZEL_TARGET="${BAZEL_TARGET} //tensorflow/contrib/lite/kernels:softmax_test"
  BAZEL_TARGET="${BAZEL_TARGET} //tensorflow/contrib/lite/kernels:space_to_depth_test"
  BAZEL_TARGET="${BAZEL_TARGET} //tensorflow/contrib/lite/kernels:svdf_test"
fi

TUT_TEST_DATA_DIR="/tmp/tf_tutorial_test_data"

##########################################################

echo "Parameterized build starts at: $(date)"
echo ""
START_TIME=$(date +'%s')

# Convert all the required environment variables to lower case
TF_BUILD_CONTAINER_TYPE=$(to_lower ${TF_BUILD_CONTAINER_TYPE})
TF_BUILD_PYTHON_VERSION=$(to_lower ${TF_BUILD_PYTHON_VERSION})
TF_BUILD_IS_OPT=$(to_lower ${TF_BUILD_IS_OPT})
TF_BUILD_IS_PIP=$(to_lower ${TF_BUILD_IS_PIP})

if [[ ! -z "${TF_BUILD_MAVX}" ]]; then
  TF_BUILD_MAVX=$(to_lower ${TF_BUILD_MAVX})
fi


# Print parameter values
echo "Required build parameters:"
echo "  TF_BUILD_CONTAINER_TYPE=${TF_BUILD_CONTAINER_TYPE}"
echo "  TF_BUILD_PYTHON_VERSION=${TF_BUILD_PYTHON_VERSION}"
echo "  TF_BUILD_IS_OPT=${TF_BUILD_IS_OPT}"
echo "  TF_BUILD_IS_PIP=${TF_BUILD_IS_PIP}"
echo "Optional build parameters:"
echo "  TF_BUILD_DRY_RUN=${TF_BUILD_DRY_RUN}"
echo "  TF_BUILD_MAVX=${TF_BUILD_MAVX}"
echo "  TF_BUILD_APPEND_CI_DOCKER_EXTRA_PARAMS="\
"${TF_BUILD_APPEND_CI_DOCKER_EXTRA_PARAMS}"
echo "  TF_BUILD_APPEND_ARGUMENTS=${TF_BUILD_APPEND_ARGUMENTS}"
echo "  TF_BUILD_BAZEL_TARGET=${TF_BUILD_BAZEL_TARGET}"
echo "  TF_BUILD_BAZEL_CLEAN=${TF_BUILD_BAZEL_CLEAN}"
echo "  TF_BUILD_TEST_TUTORIALS=${TF_BUILD_TEST_TUTORIALS}"
echo "  TF_BUILD_INTEGRATION_TESTS=${TF_BUILD_INTEGRATION_TESTS}"
echo "  TF_BUILD_RUN_BENCHMARKS=${TF_BUILD_RUN_BENCHMARKS}"
echo "  TF_BUILD_OPTIONS=${TF_BUILD_OPTIONS}"


# Function that tries to determine CUDA capability, if deviceQuery binary
# is available on path
function get_cuda_capability_version() {
  if [[ ! -z $(which deviceQuery) ]]; then
    # The first listed device is used
    deviceQuery | grep "CUDA Capability .* version" | \
        head -1 | awk '{print $NF}'
  fi
}

# Container type, e.g., CPU, GPU
CTYPE=${TF_BUILD_CONTAINER_TYPE}

# Determine if the machine is a Mac
OPT_FLAG="--test_output=errors"
if [[ "$(uname -s)" == "Darwin" ]]; then
  DO_DOCKER=0

  echo "It appears this machine is a Mac. "\
"We will perform this build without Docker."
  echo "Also, the additional option flags will be applied to the build:"
  echo "  ${NO_DOCKER_OPT_FLAG}"
  MAIN_CMD="${NO_DOCKER_MAIN_CMD} ${CTYPE}"
  OPT_FLAG="${OPT_FLAG} ${NO_DOCKER_OPT_FLAG}"
fi

# In DO_DOCKER mode, appends environment variable to docker's run invocation.
# Otherwise, exports the corresponding variable.
function set_script_variable() {
  local VAR="$1"
  local VALUE="$2"
  if [[ $DO_DOCKER == "1" ]]; then
    TF_BUILD_APPEND_CI_DOCKER_EXTRA_PARAMS="${TF_BUILD_APPEND_CI_DOCKER_EXTRA_PARAMS} -e $VAR=$VALUE"
  else
    export $VAR="$VALUE"
  fi
}


# Process container type
if [[ ${CTYPE} == "cpu" ]] || [[ ${CTYPE} == "debian.jessie.cpu" ]]; then
  :
elif [[ ${CTYPE} == "gpu" ]]; then
  set_script_variable TF_NEED_CUDA 1

  if [[ $TF_CUDA_CLANG == "1" ]]; then
    OPT_FLAG="${OPT_FLAG} --config=cuda_clang"

    set_script_variable TF_CUDA_CLANG 1
    # For cuda_clang we download `clang` while building.
    set_script_variable TF_DOWNLOAD_CLANG 1
  else
    OPT_FLAG="${OPT_FLAG} --config=cuda"
  fi

  # Attempt to determine CUDA capability version automatically and use it if
  # CUDA capability version is not specified by the environment variables.
  CUDA_CAPA_VER=$(get_cuda_capability_version)

  if [[ ! -z ${CUDA_CAPA_VER} ]]; then
    AUTO_CUDA_CAPA_VER=0
    if [[ ${DO_DOCKER} == "1" ]] && \
       [[ "${TF_BUILD_APPEND_CI_DOCKER_EXTRA_PARAMS}" != \
           *"TF_CUDA_COMPUTE_CAPABILITIES="* ]]; then
      AUTO_CUDA_CAPA_VER=1
      TF_BUILD_APPEND_CI_DOCKER_EXTRA_PARAMS=\
"${TF_BUILD_APPEND_CI_DOCKER_EXTRA_PARAMS} -e "\
"TF_CUDA_COMPUTE_CAPABILITIES=${CUDA_CAPA_VER}"

      echo "Docker GPU build: TF_BUILD_APPEND_CI_DOCKER_EXTRA_PARAMS="\
"\"${TF_BUILD_APPEND_CI_DOCKER_EXTRA_PARAMS}\""
    elif [[ ${DO_DOCKER} == "0" ]] && \
         [[ -z "${TF_CUDA_COMPUTE_CAPABILITIES}" ]]; then
      AUTO_CUDA_CAPA_VER=1
      TF_CUDA_COMPUTE_CAPABILITIES="${CUDA_CAPA_VER}"

      echo "Non-Docker GPU build: TF_CUDA_COMPUTE_CAPABILITIES="\
"\"${TF_CUDA_COMPUTE_CAPABILITIES}\""
    fi

    if [[ ${AUTO_CUDA_CAPA_VER} == "1" ]]; then
      echo "TF_CUDA_COMPUTE_CAPABILITIES is not set:"
      echo "Using CUDA capability version from deviceQuery: ${CUDA_CAPA_VER}"
      echo ""
    fi
  fi
elif [[ ${CTYPE} == "android" ]] || [[ ${CTYPE} == "android_full" ]]; then
  :
else
  die "Unrecognized value in TF_BUILD_CONTAINER_TYPE: "\
"\"${TF_BUILD_CONTAINER_TYPE}\""
fi

# Determine if this is a benchmarks job
RUN_BENCHMARKS=0
if [[ ! -z "${TF_BUILD_RUN_BENCHMARKS}" ]] &&
   [[ "${TF_BUILD_RUN_BENCHMARKS}" != "0" ]]; then
  RUN_BENCHMARKS=1
fi

# Process Bazel "-c opt" flag
if [[ -z "${TF_BUILD_OPTIONS}" ]]; then
  if [[ ${TF_BUILD_IS_OPT} == "no_opt" ]]; then
    # PIP builds are done only with the -c opt flag
    if [[ ${TF_BUILD_IS_PIP} == "pip" ]]; then
      echo "Skipping parameter combination: ${TF_BUILD_IS_OPT} & "\
"${TF_BUILD_IS_PIP}"
      exit 0
    fi

  elif [[ ${TF_BUILD_IS_OPT} == "opt" ]]; then
    OPT_FLAG="${OPT_FLAG} -c opt"
  else
    die "Unrecognized value in TF_BUILD_IS_OPT: \"${TF_BUILD_IS_OPT}\""
  fi

  # Process MAVX option
  if [[ ! -z "${TF_BUILD_MAVX}" ]]; then
    if [[ "${TF_BUILD_MAVX}" == "mavx" ]]; then
      OPT_FLAG="${OPT_FLAG} --copt=-mavx"
    elif [[ "${TF_BUILD_MAVX}" == "mavx2" ]]; then
      OPT_FLAG="${OPT_FLAG} --copt=-mavx2"
    else
      die "Unsupported value in TF_BUILD_MAVX: ${TF_BUILD_MAVX}"
    fi
  fi
else
  case $TF_BUILD_OPTIONS in
    FASTBUILD)
      echo "Running FASTBUILD mode (noopt, nodbg)."
      ;;
    OPT)
      OPT_FLAG="${OPT_FLAG} -c opt"
      ;;
    OPTDBG)
      OPT_FLAG="${OPT_FLAG} -c opt --copt=-g"
      ;;
    MAVX)
      OPT_FLAG="${OPT_FLAG} -c opt --copt=-mavx"
      ;;
    MAVX_DBG)
      OPT_FLAG="${OPT_FLAG} -c opt --copt=-g --copt=-mavx"
      ;;
    MAVX2_FMA)
      OPT_FLAG="${OPT_FLAG} -c opt --copt=-mavx2 --copt=-mfma"
      ;;
    MAVX2_FMA_DBG)
      OPT_FLAG="${OPT_FLAG} -c opt --copt=-g --copt=-mavx2 --copt=-mfma"
      ;;
  esac
fi

# Strip whitespaces from OPT_FLAG
OPT_FLAG=$(str_strip "${OPT_FLAG}")


# 1) Filter out benchmark tests if this is not a benchmarks job;
# 2) Filter out tests with the "nomac" tag if the build is on Mac OS X.
EXTRA_ARGS=${DEFAULT_BAZEL_CONFIGS}
IS_MAC=0
if [[ "$(uname)" == "Darwin" ]]; then
  IS_MAC=1
fi
if [[ "${TF_BUILD_APPEND_ARGUMENTS}" == *"--test_tag_filters="* ]]; then
  ITEMS=(${TF_BUILD_APPEND_ARGUMENTS})

  for ITEM in "${ITEMS[@]}"; do
    if [[ ${ITEM} == *"--test_tag_filters="* ]]; then
      NEW_ITEM="${ITEM}"
      if [[ ${NEW_ITEM} != *"benchmark-test"* ]]; then
        NEW_ITEM="${NEW_ITEM},-benchmark-test"
      fi
      if [[ ${IS_MAC} == "1" ]] && [[ ${NEW_ITEM} != *"nomac"* ]]; then
        NEW_ITEM="${NEW_ITEM},-nomac"
      fi
      EXTRA_ARGS="${EXTRA_ARGS} ${NEW_ITEM}"
    else
      EXTRA_ARGS="${EXTRA_ARGS} ${ITEM}"
    fi
  done
else
  EXTRA_ARGS="${EXTRA_ARGS} ${TF_BUILD_APPEND_ARGUMENTS} --test_tag_filters=-no_oss,-oss_serial,-benchmark-test"
  if [[ ${IS_MAC} == "1" ]]; then
    EXTRA_ARGS="${EXTRA_ARGS},-nomac"
  fi
fi

# For any "tool" dependencies in genrules, Bazel will build them for host
# instead of the target configuration. We can save some build time by setting
# this flag, and it only affects a few tests.
EXTRA_ARGS="${EXTRA_ARGS} --distinct_host_configuration=false"

# Process PIP install-test option
if [[ ${TF_BUILD_IS_PIP} == "no_pip" ]] ||
   [[ ${TF_BUILD_IS_PIP} == "both" ]]; then
  # Process optional bazel target override
  if [[ ! -z "${TF_BUILD_BAZEL_TARGET}" ]]; then
    BAZEL_TARGET=${TF_BUILD_BAZEL_TARGET}
  fi

  if [[ ${CTYPE} == "cpu" ]] || \
     [[ ${CTYPE} == "debian.jessie.cpu" ]]; then
    # CPU only command, fully parallel.
    NO_PIP_MAIN_CMD="${MAIN_CMD} ${BAZEL_CMD} ${OPT_FLAG} ${EXTRA_ARGS} -- "\
"${BAZEL_TARGET}"
  elif [[ ${CTYPE} == "gpu" ]]; then
    # GPU only command, run as many jobs as the GPU count only.
    NO_PIP_MAIN_CMD="${BAZEL_CMD} ${OPT_FLAG} "\
"--local_test_jobs=${TF_GPU_COUNT} "\
"--run_under=${PARALLEL_GPU_TEST_CMD} ${EXTRA_ARGS} -- ${BAZEL_TARGET}"
  elif [[ ${CTYPE} == "android" ]]; then
    # Run android specific script for android build.
    NO_PIP_MAIN_CMD="${ANDROID_CMD} ${OPT_FLAG} "
  elif [[ ${CTYPE} == "android_full" ]]; then
    # Run android specific script for full android build.
    NO_PIP_MAIN_CMD="${ANDROID_FULL_CMD} ${OPT_FLAG} "
  fi

fi

if [[ ${TF_BUILD_IS_PIP} == "pip" ]] ||
   [[ ${TF_BUILD_IS_PIP} == "both"  ]]; then
  # Android builds conflict with PIP builds
  if [[ ${CTYPE} == "android" ]]; then
    echo "Skipping parameter combination: ${TF_BUILD_IS_PIP} & "\
"${TF_BUILD_CONTAINER_TYPE}"
    exit 0
  fi

  PIP_MAIN_CMD="${MAIN_CMD} ${PIP_CMD} ${CTYPE} ${EXTRA_ARGS} ${OPT_FLAG}"

  # Add flag for integration tests
  if [[ ! -z "${TF_BUILD_INTEGRATION_TESTS}" ]] &&
     [[ "${TF_BUILD_INTEGRATION_TESTS}" != "0" ]]; then
    PIP_MAIN_CMD="${PIP_MAIN_CMD} ${PIP_INTEGRATION_TESTS_FLAG}"
  fi

  # Add command for tutorial test
  if [[ ! -z "${TF_BUILD_TEST_TUTORIALS}" ]] &&
     [[ "${TF_BUILD_TEST_TUTORIALS}" != "0" ]]; then
    PIP_MAIN_CMD="${PIP_MAIN_CMD} ${PIP_TEST_TUTORIALS_FLAG}"

    # Prepare data directory for tutorial tests
    mkdir -p "${TUT_TEST_DATA_DIR}" ||
    die "FAILED to create data directory for tutorial tests: "\
        "${TUT_TEST_DATA_DIR}"

    if [[ "${DO_DOCKER}" == "1" ]]; then
      EXTRA_PARAMS="${EXTRA_PARAMS} -v ${TUT_TEST_DATA_DIR}:${TUT_TEST_DATA_DIR}"
    fi
  fi
fi


if [[ ${RUN_BENCHMARKS} == "1" ]]; then
  MAIN_CMD="${BENCHMARK_CMD} ${OPT_FLAG}"
elif [[ ${TF_BUILD_IS_PIP} == "no_pip" ]]; then
  MAIN_CMD="${NO_PIP_MAIN_CMD}"
elif [[ ${TF_BUILD_IS_PIP} == "pip" ]]; then
  MAIN_CMD="${PIP_MAIN_CMD}"
elif [[ ${TF_BUILD_IS_PIP} == "both" ]]; then
  MAIN_CMD="${NO_PIP_MAIN_CMD} && ${PIP_MAIN_CMD}"
else
  die "Unrecognized value in TF_BUILD_IS_PIP: \"${TF_BUILD_IS_PIP}\""
fi

# Check if this is a tf_nightly build
if [[ "${TF_NIGHTLY}" == "1" ]]; then
  EXTRA_PARAMS="${EXTRA_PARAMS} -e TF_NIGHTLY=1"
fi

# Process Python version
if [[ ${TF_BUILD_PYTHON_VERSION} == "python2" ]]; then
  :
elif [[ ${TF_BUILD_PYTHON_VERSION} == "python3" || \
        ${TF_BUILD_PYTHON_VERSION} == "python3.4" || \
        ${TF_BUILD_PYTHON_VERSION} == "python3.5" || \
        ${TF_BUILD_PYTHON_VERSION} == "python3.6" ]]; then
  # Supply proper environment variable to select Python 3
  if [[ "${DO_DOCKER}" == "1" ]]; then
    EXTRA_PARAMS="${EXTRA_PARAMS} -e CI_BUILD_PYTHON=${TF_BUILD_PYTHON_VERSION}"
  else
    # Determine the path to python3
    PYTHON3_PATH=$(which "${TF_BUILD_PYTHON_VERSION}" | head -1)
    if [[ -z "${PYTHON3_PATH}" ]]; then
      die "ERROR: Failed to locate ${TF_BUILD_PYTHON_VERSION} binary on path"
    else
      echo "Found ${TF_BUILD_PYTHON_VERSION} binary at: ${PYTHON3_PATH}"
    fi

    export PYTHON_BIN_PATH="${PYTHON3_PATH}"
  fi

else
  die "Unrecognized value in TF_BUILD_PYTHON_VERSION: "\
"\"${TF_BUILD_PYTHON_VERSION}\""
fi

# Append additional Docker extra parameters
EXTRA_PARAMS="${EXTRA_PARAMS} ${TF_BUILD_APPEND_CI_DOCKER_EXTRA_PARAMS}"

# Finally, do a dry run or call the command

# The command, which may consist of multiple parts (e.g., in the case of
# TF_BUILD_SERIAL_TESTS=1), are written to a bash script, which is
# then called. The name of the script is randomized to make concurrent
# builds on the node possible.
TMP_SCRIPT="$(mktemp)_ci_parameterized_build.sh"

if [[ "${DO_DOCKER}" == "1" ]]; then
  # Map the tmp script into the Docker container
  EXTRA_PARAMS="${EXTRA_PARAMS} -v ${TMP_SCRIPT}:/tmp/tf_build.sh"

  if [[ ! -z "${TF_BUILD_BAZEL_CLEAN}" ]] &&
     [[ "${TF_BUILD_BAZEL_CLEAN}" != "0" ]] &&
     [[ "${TF_BUILD_IS_PIP}" != "both" ]]; then
    # For TF_BUILD_IS_PIP == both, "bazel clean" will have already
    # been performed before the "bazel test" step
    EXTRA_PARAMS="${EXTRA_PARAMS} -e TF_BUILD_BAZEL_CLEAN=1"
  fi

  EXTRA_PARAMS=$(str_strip "${EXTRA_PARAMS}")

  echo "Exporting CI_DOCKER_EXTRA_PARAMS: ${EXTRA_PARAMS}"
  export CI_DOCKER_EXTRA_PARAMS="${EXTRA_PARAMS}"
fi

# Write to the tmp script
echo "#!/usr/bin/env bash" > ${TMP_SCRIPT}
if [[ ! -z "${TF_BUILD_BAZEL_CLEAN}" ]] &&
   [[ "${TF_BUILD_BAZEL_CLEAN}" != "0" ]]; then
  echo ${BAZEL_CLEAN_CMD} >> ${TMP_SCRIPT}
fi
echo ${MAIN_CMD} >> ${TMP_SCRIPT}

echo "Executing final command (${TMP_SCRIPT})..."
echo "=========================================="
cat ${TMP_SCRIPT}
echo "=========================================="
echo ""


TMP_DIR=""
DOCKERFILE_FLAG=""
if [[ "${TF_BUILD_PYTHON_VERSION}" == "python3.5" ]] ||
  [[ "${TF_BUILD_PYTHON_VERSION}" == "python3.6" ]]; then
  # Modify Dockerfile for Python3.5 | Python3.6 build
  TMP_DIR=$(mktemp -d)
  echo "Docker build will occur in temporary directory: ${TMP_DIR}"

  # Copy the files required for the docker build
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  cp -r "${SCRIPT_DIR}/install" "${TMP_DIR}/install" || \
      die "ERROR: Failed to copy directory ${SCRIPT_DIR}/install"

  DOCKERFILE="${SCRIPT_DIR}/Dockerfile.${TF_BUILD_CONTAINER_TYPE}"
  cp "${DOCKERFILE}" "${TMP_DIR}/" || \
      die "ERROR: Failed to copy Dockerfile at ${DOCKERFILE}"
  DOCKERFILE="${TMP_DIR}/Dockerfile.${TF_BUILD_CONTAINER_TYPE}"

  # Replace a line in the Dockerfile
  if sed -i \
      "s/RUN \/install\/install_pip_packages.sh/RUN \/install\/install_${TF_BUILD_PYTHON_VERSION}_pip_packages.sh/g" \
      "${DOCKERFILE}"
  then
    echo "Copied and modified Dockerfile for ${TF_BUILD_PYTHON_VERSION} build: ${DOCKERFILE}"
  else
    die "ERROR: Faild to copy and modify Dockerfile: ${DOCKERFILE}"
  fi

  DOCKERFILE_FLAG="--dockerfile ${DOCKERFILE}"
fi

chmod +x ${TMP_SCRIPT}

# Map TF_BUILD container types to containers we actually have.
if [[ "${CTYPE}" == "android_full" ]]; then
  CONTAINER="android"
else
  CONTAINER=${CTYPE}
fi

FAILURE=0
if [[ ! -z "${TF_BUILD_DRY_RUN}" ]] && [[ ${TF_BUILD_DRY_RUN} != "0" ]]; then
  # Do a dry run: just print the final command
  echo "*** This is a DRY RUN ***"
else
  # Actually run the command
  if [[ "${DO_DOCKER}" == "1" ]]; then
    ${DOCKER_MAIN_CMD} ${CONTAINER} ${DOCKERFILE_FLAG} /tmp/tf_build.sh
  else
    ${TMP_SCRIPT}
  fi

  if [[ $? != "0" ]]; then
    FAILURE=1
  fi
fi

[[ ${FAILURE} == "0" ]] && RESULT="SUCCESS" || RESULT="FAILURE"

rm -f ${TMP_SCRIPT}

END_TIME=$(date +'%s')
echo ""
echo "Parameterized build ends with ${RESULT} at: $(date) "\
"(Elapsed time: $((END_TIME - START_TIME)) s)"


# Clean up temporary directory if it exists
if [[ ! -z "${TMP_DIR}" ]]; then
  echo "Cleaning up temporary directory: ${TMP_DIR}"
  rm -rf "${TMP_DIR}"
fi

exit ${FAILURE}
