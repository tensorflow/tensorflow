#!/usr/bin/env bash
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

function is_absolute {
  [[ "$1" = /* ]] || [[ "$1" =~ ^[a-zA-Z]:[/\\].* ]]
}

function real_path() {
  is_absolute "$1" && echo "$1" || echo "$PWD/${1#./}"
}

function cp_external() {
  local src_dir=$1
  local dest_dir=$2

  pushd .
  cd "$src_dir"
  for f in `find . ! -type d ! -name '*.py' ! -path '*local_config_cuda*' ! -path '*local_config_tensorrt*' ! -path '*local_config_syslibs*' ! -path '*org_tensorflow*' ! -path '*llvm-project/llvm/*'`; do
    mkdir -p "${dest_dir}/$(dirname ${f})"
    cp "${f}" "${dest_dir}/$(dirname ${f})/"
  done
  popd

  mkdir -p "${dest_dir}/local_config_cuda/cuda/cuda/"
  cp "${src_dir}/local_config_cuda/cuda/cuda/cuda_config.h" "${dest_dir}/local_config_cuda/cuda/cuda/"
}

function copy_xla_aot_runtime_sources() {
  local src_dir=$1
  local dst_dir=$2

  local srcs_txt="tensorflow/tools/pip_package/xla_compiled_cpu_runtime_srcs.txt"

  if [ ! -f "${src_dir}/${srcs_txt}" ]; then
    echo Could not find source list file "${src_dir}/${srcs_txt}". 1>&2
    return 0
  fi

  pushd $src_dir
  for file in $(cat "${srcs_txt}")
  do
    # Sometimes $file has a prefix bazel-out/host/ we want to remove.
    prefix=${file%%tensorflow/*}  # Find the location of "tensorflow/*"
    candidate_file=${file#$prefix}  # Remove the prefix
    if [ ! -z "$candidate_file" ]; then
      file=$candidate_file
    fi
    dn=$(dirname $file)
    if test -f "$file"; then
      mkdir -p "${dst_dir}/${dn}"
      cp $file "${dst_dir}/${file}"
    else
      echo "Missing xla source file: ${file}" 1>&2
    fi
  done
  cp tensorflow/tools/pip_package/xla_build/CMakeLists.txt "${dst_dir}"
  popd
}

function move_to_root_if_exists () {
  arg_to_move="$1"
  if [ -e "${arg_to_move}" ]; then
    mv ${arg_to_move} ./
  fi
}

function reorganize_includes() {
  TMPDIR="${1%/}"
  pushd "${TMPDIR}/tensorflow/include/"

  move_to_root_if_exists external/com_google_absl/absl

  move_to_root_if_exists external/eigen_archive/Eigen
  move_to_root_if_exists external/eigen_archive/unsupported

  move_to_root_if_exists external/jsoncpp_git/include
  rm -rf external/jsoncpp_git

  move_to_root_if_exists external/com_google_protobuf/src/google
  rm -rf external/com_google_protobuf/python

  popd
}

PLATFORM="$(uname -s | tr 'A-Z' 'a-z')"
function is_windows() {
  if [[ "${PLATFORM}" =~ (cygwin|mingw32|mingw64|msys)_nt* ]]; then
    true
  else
    false
  fi
}

function prepare_src() {
  if [ $# -lt 1 ] ; then
    echo "No destination dir provided"
    exit 1
  fi

  TMPDIR="${1%/}"
  mkdir -p "$TMPDIR"
  EXTERNAL_INCLUDES="${TMPDIR}/tensorflow/include/external"
  XLA_AOT_RUNTIME_SOURCES="${TMPDIR}/tensorflow/xla_aot_runtime_src"

  echo $(date) : "=== Preparing sources in dir: ${TMPDIR}"

  if [ ! -d bazel-bin/tensorflow ]; then
    echo "Could not find bazel-bin.  Did you run from the root of the build tree?"
    exit 1
  fi

  if is_windows; then
    rm -rf ./bazel-bin/tensorflow/tools/pip_package/simple_console_for_window_unzip
    mkdir -p ./bazel-bin/tensorflow/tools/pip_package/simple_console_for_window_unzip
    echo "Unzipping simple_console_for_windows.zip to create runfiles tree..."
    unzip -o -q ./bazel-bin/tensorflow/tools/pip_package/simple_console_for_windows.zip -d ./bazel-bin/tensorflow/tools/pip_package/simple_console_for_window_unzip
    echo "Unzip finished."
    # runfiles structure after unzip the python binary
    cp -L \
      bazel-bin/tensorflow/tools/pip_package/simple_console_for_window_unzip/runfiles/org_tensorflow/LICENSE \
      "${TMPDIR}"
    cp -LR \
      bazel-bin/tensorflow/tools/pip_package/simple_console_for_window_unzip/runfiles/org_tensorflow/tensorflow \
      "${TMPDIR}"
    cp_external \
      bazel-bin/tensorflow/tools/pip_package/simple_console_for_window_unzip/runfiles \
      "${EXTERNAL_INCLUDES}/"
    copy_xla_aot_runtime_sources \
      bazel-bin/tensorflow/tools/pip_package/simple_console_for_window_unzip/runfiles/org_tensorflow \
      "${XLA_AOT_RUNTIME_SOURCES}/"
    RUNFILES=bazel-bin/tensorflow/tools/pip_package/simple_console_for_window_unzip/runfiles/org_tensorflow
    # If oneDNN was built with openMP then copy the omp libs over
    if [ -f "bazel-bin/external/llvm_openmp/libiomp5md.dll" ]; then
      cp bazel-bin/external/llvm_openmp/libiomp5md.dll ${TMPDIR}/tensorflow/python
      cp bazel-bin/external/llvm_openmp/libiomp5md.dll.if.lib ${TMPDIR}/tensorflow/python
    fi
  else
    RUNFILES=bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles/org_tensorflow
    if [ -d bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles/org_tensorflow/external ]; then
      # Old-style runfiles structure (--legacy_external_runfiles).
      cp -L \
        bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles/org_tensorflow/LICENSE \
        "${TMPDIR}"
      cp -LR \
        bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles/org_tensorflow/tensorflow \
        "${TMPDIR}"
      cp_external \
        bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles/org_tensorflow/external \
        "${EXTERNAL_INCLUDES}"
      copy_xla_aot_runtime_sources \
        bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles/org_tensorflow \
        "${XLA_AOT_RUNTIME_SOURCES}"
      # Copy MKL libs over so they can be loaded at runtime
      so_lib_dir=$(ls $RUNFILES | grep solib) || true
      if [ -n "${so_lib_dir}" ]; then
        mkl_so_dir=$(ls ${RUNFILES}/${so_lib_dir} | grep mkl) || true
        if [ -n "${mkl_so_dir}" ]; then
          mkdir "${TMPDIR}/${so_lib_dir}"
          cp -R ${RUNFILES}/${so_lib_dir}/${mkl_so_dir} "${TMPDIR}/${so_lib_dir}"
        fi
      fi
    else
      # New-style runfiles structure (--nolegacy_external_runfiles).
      cp -L \
        bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles/org_tensorflow/LICENSE \
        "${TMPDIR}"
      cp -LR \
        bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles/org_tensorflow/tensorflow \
        "${TMPDIR}"
      cp_external \
        bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles \
        "${EXTERNAL_INCLUDES}"
      copy_xla_aot_runtime_sources \
        bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles/org_tensorflow \
        "${XLA_AOT_RUNTIME_SOURCES}"
      # Copy MKL libs over so they can be loaded at runtime
      so_lib_dir=$(ls $RUNFILES | grep solib) || true
      if [ -n "${so_lib_dir}" ]; then
        mkl_so_dir=$(ls ${RUNFILES}/${so_lib_dir} | grep mkl) || true
        if [ -n "${mkl_so_dir}" ]; then
          mkdir "${TMPDIR}/${so_lib_dir}"
          cp -R ${RUNFILES}/${so_lib_dir}/${mkl_so_dir} "${TMPDIR}/${so_lib_dir}"
        fi
      fi
    fi
  fi

  mkdir -p ${TMPDIR}/third_party
  cp -R $RUNFILES/third_party/eigen3 ${TMPDIR}/third_party

  reorganize_includes "${TMPDIR}"

  cp tensorflow/tools/pip_package/MANIFEST.in ${TMPDIR}
  cp tensorflow/tools/pip_package/README ${TMPDIR}/README.md
  cp tensorflow/tools/pip_package/setup.py ${TMPDIR}

  rm -f ${TMPDIR}/tensorflow/libtensorflow_framework.so
  rm -f ${TMPDIR}/tensorflow/libtensorflow_framework.so.[0-9].*

  # TODO(annarev): copy over API files from tensorflow/api/_vN to tensorflow/
  #   except tensorflow/api/_vN/lite/.

  # Copy over keras API folder to the root directory
  # so that autocomplete works as expected for all keras subimports.
  if [ -d "${TMPDIR}/tensorflow/_api/v1/" ]
  then
    cp -r ${TMPDIR}/tensorflow/python/keras/api/_v1/keras/ ${TMPDIR}/tensorflow/keras/
    sed -i'.original' -e 's/.python.keras.api._v1/tensorflow/g' ${TMPDIR}/tensorflow/__init__.py
  else
    cp -r ${TMPDIR}/tensorflow/python/keras/api/_v2/keras/ ${TMPDIR}/tensorflow/keras/
    sed -i'.original' -e 's/.python.keras.api._v2/tensorflow/g' ${TMPDIR}/tensorflow/__init__.py
  fi
}

function build_wheel() {
  if [ $# -lt 2 ] ; then
    echo "No src and dest dir provided"
    exit 1
  fi

  TMPDIR="$1"
  DEST="$2"
  PKG_NAME_FLAG="$3"

  # Before we leave the top-level directory, make sure we know how to
  # call python.
  if [[ -e tools/python_bin_path.sh ]]; then
    source tools/python_bin_path.sh
  fi

  pushd ${TMPDIR} > /dev/null

  rm -f MANIFEST
  echo $(date) : "=== Building wheel"
  "${PYTHON_BIN_PATH:-python}" setup.py bdist_wheel ${PKG_NAME_FLAG} >/dev/null
  mkdir -p ${DEST}
  cp dist/* ${DEST}
  popd > /dev/null
  echo $(date) : "=== Output wheel file is in: ${DEST}"
}

function usage() {
  echo "Usage:"
  echo "$0 [--src srcdir] [--dst dstdir] [options]"
  echo "$0 dstdir [options]"
  echo ""
  echo "    --src                 prepare sources in srcdir"
  echo "                              will use temporary dir if not specified"
  echo ""
  echo "    --dst                 build wheel in dstdir"
  echo "                              if dstdir is not set do not build, only prepare sources"
  echo ""
  echo "  Options:"
  echo "    --project_name <name> set project name to name"
  echo "    --cpu                 build tensorflow_cpu"
  echo "    --gpu                 build tensorflow_gpu"
  echo "    --gpudirect           build tensorflow_gpudirect"
  echo "    --rocm                build tensorflow_rocm"
  echo "    --nightly_flag        build tensorflow nightly"
  echo ""
  exit 1
}

function main() {
  PKG_NAME_FLAG=""
  PROJECT_NAME=""
  CPU_BUILD=0
  GPU_BUILD=0
  GPUDIRECT_BUILD=0
  ROCM_BUILD=0
  NIGHTLY_BUILD=0
  SRCDIR=""
  DSTDIR=""
  CLEANSRC=1
  while true; do
    if [[ "$1" == "--help" ]]; then
      usage
      exit 1
    elif [[ "$1" == "--nightly_flag" ]]; then
      NIGHTLY_BUILD=1
    elif [[ "$1" == "--gpu" ]]; then
      GPU_BUILD=1
    elif [[ "$1" == "--cpu" ]]; then
      CPU_BUILD=1
    elif [[ "$1" == "--gpudirect" ]]; then
      GPUDIRECT_BUILD=1
    elif [[ "$1" == "--rocm" ]]; then
      ROCM_BUILD=1
    elif [[ "$1" == "--project_name" ]]; then
      shift
      if [[ -z "$1" ]]; then
        break
      fi
      PROJECT_NAME="$1"
    elif [[ "$1" == "--src" ]]; then
      shift
      SRCDIR="$(real_path $1)"
      CLEANSRC=0
    elif [[ "$1" == "--dst" ]]; then
      shift
      DSTDIR="$(real_path $1)"
    else
      DSTDIR="$(real_path $1)"
    fi
    shift

    if [[ -z "$1" ]]; then
      break
    fi
  done

  if [[ $(( GPU_BUILD + CPU_BUILD + GPUDIRECT_BUILD + ROCM_BUILD )) -gt "1" ]]; then
    echo "Only one of [--gpu, --cpu, --gpudirect, --rocm] may be provided."
    usage
    exit 1
  fi

  if [[ -z "$DSTDIR" ]] && [[ -z "$SRCDIR" ]]; then
    echo "No destination dir provided"
    usage
    exit 1
  fi

  if [[ -z "$SRCDIR" ]]; then
    # make temp srcdir if none set
    SRCDIR="$(mktemp -d -t tmp.XXXXXXXXXX)"
  fi

  prepare_src "$SRCDIR"

  if [[ -z "$DSTDIR" ]]; then
      # only want to prepare sources
      exit
  fi

  if [[ -n ${PROJECT_NAME} ]]; then
    PKG_NAME_FLAG="--project_name ${PROJECT_NAME}"
  elif [[ ${NIGHTLY_BUILD} == "1" && ${GPU_BUILD} == "1" ]]; then
    PKG_NAME_FLAG="--project_name tf_nightly_gpu"
  elif [[ ${NIGHTLY_BUILD} == "1" && ${GPUDIRECT_BUILD} == "1" ]]; then
    PKG_NAME_FLAG="--project_name tf_nightly_gpudirect"
  elif [[ ${NIGHTLY_BUILD} == "1" && ${ROCM_BUILD} == "1" ]]; then
    PKG_NAME_FLAG="--project_name tf_nightly_rocm"
  elif [[ ${NIGHTLY_BUILD} == "1" && ${CPU_BUILD} == "1" ]]; then
    PKG_NAME_FLAG="--project_name tf_nightly_cpu"
  elif [[ ${NIGHTLY_BUILD} == "1" ]]; then
    PKG_NAME_FLAG="--project_name tf_nightly"
  elif [[ ${GPU_BUILD} == "1" ]]; then
    PKG_NAME_FLAG="--project_name tensorflow_gpu"
  elif [[ ${GPUDIRECT_BUILD} == "1" ]]; then
    PKG_NAME_FLAG="--project_name tensorflow_gpudirect"
  elif [[ ${ROCM_BUILD} == "1" ]]; then
    PKG_NAME_FLAG="--project_name tensorflow_rocm"
  elif [[ ${CPU_BUILD} == "1" ]]; then
    PKG_NAME_FLAG="--project_name tensorflow_cpu"
  fi

  build_wheel "$SRCDIR" "$DSTDIR" "$PKG_NAME_FLAG"

  if [[ $CLEANSRC -ne 0 ]]; then
    rm -rf "${TMPDIR}"
  fi
}

main "$@"
