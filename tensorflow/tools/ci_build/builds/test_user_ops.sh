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
# Test user-defined ops against installation of TensorFlow.
#
# Usage: test_user_ops.sh [--virtualenv] [--gpu]
#
# If the flag --virtualenv is set, the script will use "python" as the Python
# binary path. Otherwise, it will use tools/python_bin_path.sh to determine
# the Python binary path.
#
# The --gpu flag informs the script that this is a GPU build, so that the
# appropriate test denylists can be applied accordingly.
#

echo ""
echo "=== Testing user ops ==="

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/builds_common.sh"


# Process input arguments
IS_VIRTUALENV=0
IS_GPU=0
while true; do
  if [[ "$1" == "--virtualenv" ]]; then
    IS_VIRTUALENV=1
  elif [[ "$1" == "--gpu" ]]; then
    IS_GPU=1
  fi
  shift

  if [[ -z "$1" ]]; then
    break
  fi
done

TMP_DIR=$(mktemp -d)
mkdir -p "${TMP_DIR}"

cleanup() {
  rm -rf "${TMP_DIR}"
}

die() {
  echo $@
  cleanup
  exit 1
}


# Obtain the path to Python binary
if [[ ${IS_VIRTUALENV} == "1" ]]; then
  PYTHON_BIN_PATH="$(which python)"
else
  source tools/python_bin_path.sh
  # Assume: PYTHON_BIN_PATH is exported by the script above
fi
echo "PYTHON_BIN_PATH: ${PYTHON_BIN_PATH}"


pushd "${TMP_DIR}"

# Obtain compilation and linking flags
TF_CFLAGS=( $("${PYTHON_BIN_PATH}" \
	      -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $("${PYTHON_BIN_PATH}" \
	      -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

if [[ -z "${TF_CFLAGS[*]}" || -z "${TF_LFLAGS[*]}" ]]; then
  die "FAILED to determine TensorFlow compilation or linking flags"
else
  echo "TensorFlow compile flags: ${TF_CFLAGS[*]}"
  echo "TensorFlow link flags: ${TF_LFLAGS[*]}"
fi

# Check g++ availability
GPP_BIN="g++"
if [[ -z $(which "${GPP_BIN}") ]]; then
  die "ERROR: ${GPP_BIN} not on path"
fi

echo ""
echo "g++ version:"
"${GPP_BIN}" -v
echo ""


IS_MAC=0
if [[ $(uname) == "Darwin" ]]; then
  echo "Detected Mac OS X environment"
  IS_MAC=1
fi

EXTRA_GPP_FLAGS=""
if [[ ${IS_MAC} == "1" ]]; then
  # Extra flags required on Mac OS X, where dynamic_lookup is not the default
  # behavior.
  EXTRA_GPP_FLAGS="${EXTRA_GPP_FLAGS} -undefined dynamic_lookup"
fi

echo "Extra GPP flag: ${EXTRA_GPP_FLAGS}"

# Input to the user op
OP_INPUT="[42, 43, 44]"

if [[ ${IS_GPU} == "0" ]]; then
  echo "Testing user ops in CPU environment"

  # Expected output from user op
  EXPECTED_OUTPUT="[42, 0, 0]"

  # Locate the op kernel C++ file
  OP_KERNEL_CC="${SCRIPT_DIR}/user_ops/zero_out_op_kernel_1.cc"
  OP_KERNEL_CC=$(realpath "${OP_KERNEL_CC}")

  if [[ ! -f "${OP_KERNEL_CC}" ]]; then
    die "ERROR: Unable to find user-op kernel C++ file at: ${OP_KERNEL_CC}"
  fi

  # Copy the file to a non-TensorFlow source directory
  cp "${OP_KERNEL_CC}" ./

  # Compile the op kernel into an .so file
  SRC_FILE=$(basename "${OP_KERNEL_CC}")

  echo "Compiling user op C++ source file ${SRC_FILE}"

  USER_OP_SO="zero_out.so"

  "${GPP_BIN}" -std=c++11 ${EXTRA_GPP_FLAGS} \
    -shared "${SRC_FILE}" -o "${USER_OP_SO}" \
    -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]}  || \
    die "g++ compilation of ${SRC_FILE} FAILED"

else
  echo "Testing user ops in GPU environment"

  # Expected output from user op
  EXPECTED_OUTPUT="[43, 44, 45]"

  # Check nvcc availability
  NVCC_BIN="/usr/local/cuda/bin/nvcc"
  if [[ -z $(which "${NVCC_BIN}") ]]; then
    die "ERROR: ${NVCC_BIN} not on path"
  fi

  echo ""
  echo "nvcc version:"
  "${NVCC_BIN}" --version
  echo ""

  OP_KERNEL_CU="${SCRIPT_DIR}/user_ops/cuda_op_kernel.cu.cc"
  OP_KERNEL_CU=$(realpath "${OP_KERNEL_CU}")
  if [[ ! -f "${OP_KERNEL_CU}" ]]; then
    die "ERROR: Unable to find user-op kernel CUDA file at: ${OP_KERNEL_CU}"
  fi

  OP_KERNEL_CC="${SCRIPT_DIR}/user_ops/cuda_op_kernel.cc"
  OP_KERNEL_CC=$(realpath "${OP_KERNEL_CC}")
  if [[ ! -f "${OP_KERNEL_CC}" ]]; then
    die "ERROR: Unable to find user-op kernel C++ file at: ${OP_KERNEL_CC}"
  fi

  # Copy the file to a non-TensorFlow source directory
  cp "${OP_KERNEL_CU}" ./
  cp "${OP_KERNEL_CC}" ./

  OP_KERNEL_O=$(echo "${OP_KERNEL_CC}" | sed -e 's/\.cc/\.o/')
  "${NVCC_BIN}" -std=c++11 \
      -c -o "${OP_KERNEL_O}" "${OP_KERNEL_CU}" \
      ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC || \
      die "nvcc compilation of ${OP_KERNEL_CC} FAILED"

  CUDA_LIB_DIR="/usr/local/cuda/lib64"
  if [[ ! -d "${CUDA_LIB_DIR}" ]]; then
    CUDA_LIB_DIR="/usr/local/cuda/lib"
  fi
  if [[ ! -d "${CUDA_LIB_DIR}" ]]; then
    die "ERROR: Failed to find CUDA library directory at either of "
"/usr/local/cuda/lib and /usr/local/cuda/lib64"
  fi

  echo "Found CUDA library directory at: ${CUDA_LIB_DIR}"
  echo ""

  # USER_OP_SO=$(basename $(echo "${OP_KERNEL_CC}" | sed -e 's/\.cc/\.so/'))
  USER_OP_SO="add_one.so"
  "${GPP_BIN}" -std=c++11 ${EXTRA_GPP_FLAGS} \
      -shared -o "${USER_OP_SO}" "${OP_KERNEL_CC}" \
      "${OP_KERNEL_O}" ${TF_CFLAGS[@]} -L "${CUDA_LIB_DIR}" ${TF_LFLAGS[@]} \
      -fPIC -lcudart || \
      die "g++ compilation of ${OP_KERNEL_CC}" FAILED
fi

# Try running the op
USER_OP=$(echo "${USER_OP_SO}" | sed -e 's/\.so//')
echo "Invoking user op ${USER_OP} defined in file ${USER_OP_SO} "\
"via pip installation"

function run_op() {
  local ORIG_OUTPUT=$1
  local ADDITIONAL_LOG=$2

  # Format OUTPUT for analysis
  if [[ -z $(echo "${ORIG_OUTPUT}" | grep -o ',') ]]; then
    if [[ ${IS_MAC} == "1" ]]; then
      local OUTPUT=$(echo "${ORIG_OUTPUT}" | sed -E -e 's/[ \t]+/,/g')
    else
      local OUTPUT=$(echo "${ORIG_OUTPUT}" | sed -r -e 's/[ \t]+/,/g')
    fi
  else
    local OUTPUT="${ORIG_OUTPUT}"
  fi

  local EQUALS_EXPECTED=$("${PYTHON_BIN_PATH}" -c "print(${OUTPUT} == ${EXPECTED_OUTPUT})")

  if [[ "${EQUALS_EXPECTED}" != "True" ]]; then
    local ERROR="FAILED: Output from user op (${OUTPUT}) does not match expected "\
  "output ${EXPECTED_OUTPUT}"${ADDITIONAL_LOG}
    die ${ERROR}
  else
    echo "Output from user op (${OUTPUT}) matches expected output"
  fi
}

printf "\nTesting execution of user-defined op under graph mode:\n\n"
run_op "$("${PYTHON_BIN_PATH}" -c "import tensorflow as tf; print(tf.Session('').run(tf.load_op_library('./${USER_OP_SO}').${USER_OP}(${OP_INPUT})))")"

if [[ "${IS_GPU}" == "0" ]]; then
  printf "\nTesting execution of user-defined op under eager mode:\n\n"
  run_op "$("${PYTHON_BIN_PATH}" -c "import tensorflow as tf; tf.enable_eager_execution(); print(tf.load_op_library('./${USER_OP_SO}').${USER_OP}(${OP_INPUT}).numpy())")" " in eager mode"
else
  printf "\nSKIPPING the testing of execution of user-defined GPU kernel under eager mode. See b/122972785.\n\n"
fi


popd

cleanup

echo ""
echo "SUCCESS: Testing of user ops PASSED"
