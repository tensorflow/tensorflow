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

# Read the value of VERSION from vercod.bzl
VERSION=$(grep 'VERSION = ' tensorflow/tensorflow.bzl | sed -E 's/VERSION = "(.*)"/\1/g')
VERSION_MAJOR=$(echo "$VERSION" | cut -d '.' -f1)
echo TensorFlow Version: ${VERSION}
echo TensorFlow Major Version: ${VERSION_MAJOR}

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
  for f in `find . ! -type d ! -name '*.py' ! -path '*local_config_cuda*' ! -path '*local_config_tensorrt*' ! -path '*pypi*' ! -path '*python_x86_64*' ! -path '*python_aarch64*' ! -path '*local_config_syslibs*' ! -path '*org_tensorflow*' ! -path '*llvm-project/llvm/*' ! -path '*local_tsl*' ! -path '*local_xla*'`; do
    mkdir -p "${dest_dir}/$(dirname ${f})"
    cp "${f}" "${dest_dir}/$(dirname ${f})/"
  done
  popd

  mkdir -p "${dest_dir}/local_config_cuda/cuda/cuda/"
  cp "${src_dir}/local_config_cuda/cuda/cuda/cuda_config.h" "${dest_dir}/local_config_cuda/cuda/cuda/"
}

function cp_local_config_python() {
  local src_dir=$1
  local dest_dir=$2
  pushd .
  cd "$src_dir"
  mkdir -p "${dest_dir}/local_config_python/numpy_include/"
  cp -r "pypi_numpy/site-packages/numpy/core/include/numpy" "${dest_dir}/local_config_python/numpy_include/"
  mkdir -p "${dest_dir}/local_config_python/python_include/"
  if is_windows; then
    cp -r python_*/include/* "${dest_dir}/local_config_python/python_include/"
  else
    cp -r python_*/include/python*/* "${dest_dir}/local_config_python/python_include/"
  fi
  popd
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

    # For XLA/TSL, we need to remove the prefix "../local_{xla|tsl}/".
    dst_file=$file
    dst_file=${dst_file#"../local_xla/"}
    dst_file=${dst_file#"../local_tsl/"}

    if test -f "$file"; then
      mkdir -p "${dst_dir}/$(dirname $dst_file)"
      cp $file "${dst_dir}/${dst_file}"
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

  cp -R external/ml_dtypes ./

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

function is_macos() {
  if [[ "${PLATFORM}" =~ darwin* ]]; then
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
  echo TMPDIR: ${TMPDIR}
  EXTERNAL_INCLUDES="${TMPDIR}/tensorflow/include/external"
  XLA_AOT_RUNTIME_SOURCES="${TMPDIR}/tensorflow/xla_aot_runtime_src"

  echo $(date) : "=== Preparing sources in dir: ${TMPDIR}"

  if [ ! -d bazel-bin/tensorflow ]; then
    echo "Could not find bazel-bin.  Did you run from the root of the build tree?"
    exit 1
  fi

  if is_windows; then
    cp -L \
      bazel-bin/tensorflow/tools/pip_package/build_pip_package.exe.runfiles/org_tensorflow/LICENSE \
      "${TMPDIR}"
      
    # Change the format of file path (TMPDIR-->TMPDIR_rsync) which is input to the rsync from
    # Windows-compatible to Linux-compatible to resolve the error below 
    # error: ssh: Could not resolve hostname c: No such host is known. 
    
    TMPDIR_rsync=`cygpath $TMPDIR`  
    rsync -a \
      bazel-bin/tensorflow/tools/pip_package/build_pip_package.exe.runfiles/org_tensorflow/tensorflow \
      "${TMPDIR_rsync}"
    cp_external \
      bazel-bin/tensorflow/tools/pip_package/build_pip_package.exe.runfiles \
      "${EXTERNAL_INCLUDES}/"
    cp_local_config_python \
      bazel-bin/tensorflow/tools/pip_package/build_pip_package.exe.runfiles \
      "${EXTERNAL_INCLUDES}/"
    copy_xla_aot_runtime_sources \
      bazel-bin/tensorflow/tools/pip_package/build_pip_package.exe.runfiles/org_tensorflow \
      "${XLA_AOT_RUNTIME_SOURCES}/"
    RUNFILES=bazel-bin/tensorflow/tools/pip_package/build_pip_package.exe.runfiles/org_tensorflow
    # If oneDNN was built with openMP then copy the omp libs over
    if [ -f "bazel-bin/external/llvm_openmp/libiomp5md.dll" ]; then
      cp bazel-bin/external/llvm_openmp/libiomp5md.dll ${TMPDIR}/tensorflow/python
      cp bazel-bin/external/llvm_openmp/libiomp5md.dll.if.lib ${TMPDIR}/tensorflow/python
    fi
  else
    RUNFILES=bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles/org_tensorflow
    # Resolved the issue of a missing symlink to libtensorflow_cc.so.2 b/264967822#comment25
    if is_macos; then
      if [ ! -L "${RUNFILES}/tensorflow/libtensorflow_cc.${VERSION_MAJOR}.dylib" ]; then
        ln -s "$(dirname "$(readlink "${RUNFILES}/tensorflow/libtensorflow_cc.${VERSION}.dylib")")/libtensorflow_cc.${VERSION_MAJOR}.dylib" \
         "${RUNFILES}/tensorflow/libtensorflow_cc.${VERSION_MAJOR}.dylib"
        echo "Created symlink: $(dirname "$(readlink "${RUNFILES}/tensorflow/libtensorflow_cc.${VERSION}.dylib")")/libtensorflow_cc.${VERSION_MAJOR}.dylib -> \
          ${RUNFILES}/tensorflow/libtensorflow_cc.${VERSION_MAJOR}.dylib"
      else
        echo "Symlink already exists: ${RUNFILES}/tensorflow/libtensorflow_cc.${VERSION_MAJOR}.dylib"
      fi
    else
      # cp -P ${RUNFILES}/tensorflow/libtensorflow_cc.so.${VERSION} ${RUNFILES}/tensorflow/libtensorflow_cc.so.${VERSION_MAJOR}
      if [ ! -L "${RUNFILES}/tensorflow/libtensorflow_cc.so.${VERSION_MAJOR}" ]; then
        ln -s "$(dirname "$(readlink "${RUNFILES}/tensorflow/libtensorflow_cc.so.${VERSION}")")/libtensorflow_cc.so.${VERSION_MAJOR}" \
          "${RUNFILES}/tensorflow/libtensorflow_cc.so.${VERSION_MAJOR}"
      else
        echo "Symlink already exists: ${RUNFILES}/tensorflow/libtensorflow_cc.so.${VERSION_MAJOR}"
      fi
    fi
    cp -L \
      bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles/org_tensorflow/LICENSE \
      "${TMPDIR}"
    # Check if it is a tpu build
    if [[ ${TPU_BUILD} == "1" ]]; then
      # Check if libtpu.so exists
      if [[ -f "./tensorflow/lib/libtpu.so" ]]; then
        if [[ ! -L "${RUNFILES}/tensorflow/lib/libtpu.so" ]]; then
          mkdir "$(real_path ${RUNFILES}/tensorflow/lib)"
          ln -s $(real_path ./tensorflow/lib/libtpu.so) $(real_path ${RUNFILES}/tensorflow/lib/libtpu.so)
          echo "Created symlink: $(real_path ./tensorflow/lib/libtpu.so) -> \
            $(real_path ${RUNFILES}/tensorflow/lib/libtpu.so)"
        else
          echo "Symlink already exists: ${RUNFILES}/tensorflow/lib/libtpu.so"
        fi
      else
        echo "Libtpu.so is not found in $(real_path ./tensorflow/lib/)"
        exit 1
      fi
    fi
    cp -LR \
      bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles/org_tensorflow/tensorflow \
      "${TMPDIR}"
    # Prevents pip package bloat. See b/228948031#comment17.
    rm -f ${TMPDIR}/tensorflow/python/lib_pywrap_tensorflow_internal.*
    if [ -d bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles/org_tensorflow/external ]; then
      # Old-style runfiles structure (--legacy_external_runfiles).
      cp_external \
        bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles/org_tensorflow/external \
        "${EXTERNAL_INCLUDES}"
      cp_local_config_python \
        bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles/org_tensorflow/external \
        "${EXTERNAL_INCLUDES}"
    else
      # New-style runfiles structure (--nolegacy_external_runfiles).
      cp_external \
        bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles \
        "${EXTERNAL_INCLUDES}"
      cp_local_config_python \
        bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles \
        "${EXTERNAL_INCLUDES}"
    fi
    copy_xla_aot_runtime_sources \
      bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles/org_tensorflow \
      "${XLA_AOT_RUNTIME_SOURCES}"
    # Copy MKL libs over so they can be loaded at runtime
    so_lib_dir=$(ls $RUNFILES | grep solib)
    if is_macos; then
      chmod +rw ${TMPDIR}/tensorflow/python/_pywrap_tensorflow_internal.so
    else
      chmod +rw ${TMPDIR}/tensorflow/python/_pywrap_tensorflow_internal.so
      chmod +rw ${TMPDIR}/tensorflow/compiler/mlir/quantization/tensorflow/python/pywrap_quantize_model.so
      patchelf --set-rpath $(patchelf --print-rpath ${TMPDIR}/tensorflow/python/_pywrap_tensorflow_internal.so):\$ORIGIN/../../tensorflow/tsl/python/lib/core ${TMPDIR}/tensorflow/python/_pywrap_tensorflow_internal.so
      patchelf --set-rpath $(patchelf --print-rpath ${TMPDIR}/tensorflow/compiler/mlir/quantization/tensorflow/python/pywrap_quantize_model.so):\$ORIGIN/../../../../../python ${TMPDIR}/tensorflow/compiler/mlir/quantization/tensorflow/python/pywrap_quantize_model.so
      patchelf --shrink-rpath ${TMPDIR}/tensorflow/python/_pywrap_tensorflow_internal.so
      patchelf --shrink-rpath ${TMPDIR}/tensorflow/compiler/mlir/quantization/tensorflow/python/pywrap_quantize_model.so
    fi
    mkl_so_dir=$(ls ${RUNFILES}/${so_lib_dir} | grep mkl) || true
    if [ -n "${mkl_so_dir}" ]; then
      mkdir "${TMPDIR}/${so_lib_dir}"
      cp -R ${RUNFILES}/${so_lib_dir}/${mkl_so_dir} "${TMPDIR}/${so_lib_dir}"
    fi
  fi

  # Move vendored files into proper locations
  # This is required because TSL/XLA don't publish their own wheels
  # We copy from bazel-bin/tensorflow instead of bazel-bin/internal to copy
  # headers from TSL/XLA into tensorflow so that InstallHeaders can move
  # them back into tensorflow/include
  if is_windows; then
    cp -RLn bazel-bin/tensorflow/tools/pip_package/build_pip_package.exe.runfiles/local_tsl/tsl/ ${TMPDIR}/tensorflow
    cp -RLn bazel-bin/tensorflow/tools/pip_package/build_pip_package.exe.runfiles/local_xla/xla/ ${TMPDIR}/tensorflow/compiler
  else
    cp -RLn bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles/local_tsl/tsl ${TMPDIR}/tensorflow
    cp -RLn bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles/local_xla/xla ${TMPDIR}/tensorflow/compiler
  fi
  # Fix the proto stubs
  if is_macos; then
    find ${TMPDIR}/tensorflow/ -name "*.py" -type f -exec sed -i '' 's/from tsl\./from tensorflow.tsl./' {} \;
    find ${TMPDIR}/tensorflow/ -name "*.py" -type f -exec sed -i '' 's/from local_xla\.xla/from tensorflow.compiler.xla/' {} \;
    find ${TMPDIR}/tensorflow/ -name "*.py" -type f -exec sed -i '' 's/from xla/from tensorflow.compiler.xla/' {} \;
  else
    find ${TMPDIR}/tensorflow/ -name "*.py" -type f -exec sed -i'' 's/from tsl\./from tensorflow.tsl./' {} \;
    find ${TMPDIR}/tensorflow/ -name "*.py" -type f -exec sed -i'' 's/from local_xla\.xla/from tensorflow.compiler.xla/' {} \;
    find ${TMPDIR}/tensorflow/ -name "*.py" -type f -exec sed -i'' 's/from xla/from tensorflow.compiler.xla/' {} \;
  fi

  mkdir -p ${TMPDIR}/third_party
  cp -LR $RUNFILES/../local_config_cuda/cuda/_virtual_includes/cuda_headers_virtual/third_party/gpus ${TMPDIR}/third_party
  cp $RUNFILES/tensorflow/tools/pip_package/THIRD_PARTY_NOTICES.txt "${TMPDIR}/tensorflow"

  reorganize_includes "${TMPDIR}"

  cp tensorflow/tools/pip_package/MANIFEST.in ${TMPDIR}
  cp tensorflow/tools/pip_package/README ${TMPDIR}/README.md
  cp tensorflow/tools/pip_package/setup.py ${TMPDIR}

  rm -f ${TMPDIR}/tensorflow/libtensorflow_framework.so
  rm -f ${TMPDIR}/tensorflow/libtensorflow_framework.so.[0-9].*

  # Copying symlinks with -L duplicates these libraries.
  rm -f ${TMPDIR}/tensorflow/libtensorflow_framework.dylib
  rm -f ${TMPDIR}/tensorflow/libtensorflow_framework.[0-9].*.dylib
  rm -f ${TMPDIR}/tensorflow/libtensorflow_cc.dylib
  rm -f ${TMPDIR}/tensorflow/libtensorflow_cc.[0-9].*.dylib

  # TODO(annarev): copy over API files from tensorflow/api/_vN to tensorflow/
  #   except tensorflow/api/_vN/lite/.

  # TODO(b/150440817): support autocomplete for tf.keras
  # Copy over keras API folder to the root directory
  # so that autocomplete works as expected for all keras subimports.
  # if [ -d "${TMPDIR}/tensorflow/_api/v1/" ]
  # then
  #   cp -r ${TMPDIR}/tensorflow/python/keras/api/_v1/keras/ ${TMPDIR}/tensorflow/keras/
  #   sed -i'.original' -e 's/.python.keras.api._v1/tensorflow/g' ${TMPDIR}/tensorflow/__init__.py
  # else
  #   cp -r ${TMPDIR}/tensorflow/python/keras/api/_v2/keras/ ${TMPDIR}/tensorflow/keras/
  #   sed -i'.original' -e 's/.python.keras.api._v2/tensorflow/g' ${TMPDIR}/tensorflow/__init__.py
  # fi
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
  if is_windows; then
    PY_DIR=$(find -L ./bazel-tensorflow/external -maxdepth 1 -type d -name "python_*")
    FULL_DIR="$(real_path "$PY_DIR")/python"
    export PYTHONPATH="$PYTHONPATH:$PWD/bazel-tensorflow/external/pypi_wheel/site-packages/"
  else
    PY_DIR=$(find ./bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles/ -maxdepth 1 -type d -name "python_*")
    FULL_DIR="$(real_path "$PY_DIR")/bin/python3"
    export PYTHONPATH="$PYTHONPATH:$PWD/bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles/pypi_wheel/site-packages/"
  fi
  
  pushd ${TMPDIR} > /dev/null

  rm -f MANIFEST
  echo $(date) : "=== Building wheel"
  $FULL_DIR setup.py bdist_wheel ${PKG_NAME_FLAG} >/dev/null
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
  echo "    --tpu                 build tensorflow_tpu"
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
  TPU_BUILD=0
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
    elif [[ "$1" == "--cpu" ]]; then
      CPU_BUILD=1
    elif [[ "$1" == "--tpu" ]]; then
      TPU_BUILD=1
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

  if [[ $(( TPU_BUILD + CPU_BUILD + GPUDIRECT_BUILD + ROCM_BUILD )) -gt "1" ]]; then
    echo "Only one of [--tpu, --cpu, --gpudirect, --rocm] may be provided."
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
  elif [[ ${NIGHTLY_BUILD} == "1" && ${GPUDIRECT_BUILD} == "1" ]]; then
    PKG_NAME_FLAG="--project_name tf_nightly_gpudirect"
  elif [[ ${NIGHTLY_BUILD} == "1" && ${ROCM_BUILD} == "1" ]]; then
    PKG_NAME_FLAG="--project_name tf_nightly_rocm"
  elif [[ ${NIGHTLY_BUILD} == "1" && ${CPU_BUILD} == "1" ]]; then
    PKG_NAME_FLAG="--project_name tf_nightly_cpu"
  elif [[ ${NIGHTLY_BUILD} == "1" && ${TPU_BUILD} == "1" ]]; then
    PKG_NAME_FLAG="--project_name tf_nightly_tpu"
  elif [[ ${NIGHTLY_BUILD} == "1" ]]; then
    PKG_NAME_FLAG="--project_name tf_nightly"
  elif [[ ${GPUDIRECT_BUILD} == "1" ]]; then
    PKG_NAME_FLAG="--project_name tensorflow_gpudirect"
  elif [[ ${ROCM_BUILD} == "1" ]]; then
    PKG_NAME_FLAG="--project_name tensorflow_rocm"
  elif [[ ${CPU_BUILD} == "1" ]]; then
    PKG_NAME_FLAG="--project_name tensorflow_cpu"
  elif [[ ${TPU_BUILD} == "1" ]]; then
    PKG_NAME_FLAG="--project_name tensorflow_tpu"
  fi

  build_wheel "$SRCDIR" "$DSTDIR" "$PKG_NAME_FLAG"

  if [[ $CLEANSRC -ne 0 ]]; then
    rm -rf "${TMPDIR}"
  fi
}

main "$@"
