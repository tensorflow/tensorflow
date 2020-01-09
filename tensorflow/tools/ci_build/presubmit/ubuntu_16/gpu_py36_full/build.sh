#!/bin/bash
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

# Error if we somehow forget to set the path to bazel_wrapper.py
set -u
BAZEL_WRAPPER_PATH=$1
set +u

# From this point on, logs can be publicly available
set -x

function run_build () {
  # Build a unique cache silo string.
  UBUNTU_VERSION=$(lsb_release -a | grep Release | awk '{print $2}')
  IMAGE_VERSION=$(cat /VERSION)
  CACHE_SILO_VAL="gpu-py3-ubuntu-16-${UBUNTU_VERSION}-${IMAGE_VERSION}"

  # Run configure.
  # Do not run configure.py when doing remote build & test:
  # Most things we set with configure.py are not used in a remote build setting,
  # as the build will be defined by pre-configured build files that are checked
  # in.
  # TODO(klimek): Allow using the right set of bazel flags without the need to
  # run configure.py; currently we need to carefully copy them, which is brittle.
  export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"
  # TODO(klimek): Remove once we don't try to read it while setting up the remote
  # config for cuda (we currently don't use it, as it's only used when compiling
  # with clang, but we still require it to be set anyway).
  export TF_CUDA_COMPUTE_CAPABILITIES=6.0
  export ACTION_PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
  export PYTHON_BIN_PATH="/usr/bin/python3"
  export TF2_BEHAVIOR=1
  tag_filters="gpu,-no_gpu,-nogpu,-benchmark-test,-no_oss,-oss_serial,-no_gpu_presubmit""$(maybe_skip_v1)"

  # Get the default test targets for bazel.
  source tensorflow/tools/ci_build/build_scripts/PRESUBMIT_BUILD_TARGETS.sh

  # Run bazel test command. Double test timeouts to avoid flakes.
  # //tensorflow/core:platform_setround_test is not supported. See b/64264700
  # TODO(klimek): Re-enable tensorrt tests (with different runtime image) once
  # we can build them.
  # TODO(klimek): Stop using action_env for things that are only needed during
  # setup - we're artificially poisoning the cache.
  "${BAZEL_WRAPPER_PATH}" \
    test \
    --config=rbe \
    --python_path="${PYTHON_BIN_PATH}" \
    --action_env=PATH="${ACTION_PATH}" \
    --action_env=PYTHON_BIN_PATH="${PYTHON_BIN_PATH}" \
    --action_env=TF2_BEHAVIOR="${TF2_BEHAVIOR}" \
    --action_env=REMOTE_GPU_TESTING=1 \
    --action_env=TF_CUDA_COMPUTE_CAPABILITIES="${TF_CUDA_COMPUTE_CAPABILITIES}" \
    --action_env=TF_CUDA_CONFIG_REPO=@org_tensorflow//third_party/toolchains/preconfig/ubuntu16.04/cuda10.0-cudnn7 \
    --action_env=TF_CUDA_VERSION=10 \
    --action_env=TF_CUDNN_VERSION=7 \
    --action_env=TF_NEED_TENSORRT=0 \
    --action_env=TF_NEED_CUDA=1 \
    --action_env=TF_PYTHON_CONFIG_REPO=@org_tensorflow//third_party/toolchains/preconfig/ubuntu16.04/py3 \
    --test_env=LD_LIBRARY_PATH \
    --test_tag_filters="${tag_filters}" \
    --build_tag_filters="${tag_filters}" \
    --test_lang_filters=cc,py \
    --define=with_default_optimizations=true \
    --define=framework_shared_object=true \
    --define=with_xla_support=true \
    --define=using_cuda_nvcc=true \
    --define=use_fast_cpp_protos=true \
    --define=allow_oversize_protos=true \
    --define=grpc_no_ares=true \
    -c opt \
    --copt="-w" \
    --copt=-mavx \
    --linkopt=-lrt \
    --distinct_host_configuration=false \
    --remote_default_platform_properties="properties:{name:\"build\" value:\"${CACHE_SILO_VAL}\"}" \
    --crosstool_top=//third_party/toolchains/preconfig/ubuntu16.04/gcc7_manylinux2010-nvcc-cuda10.0:toolchain \
    --host_javabase=@bazel_toolchains//configs/ubuntu16_04_clang/1.1:jdk8 \
    --javabase=@bazel_toolchains//configs/ubuntu16_04_clang/1.0:jdk8 \
    --host_java_toolchain=@bazel_tools//tools/jdk:toolchain_hostjdk8 \
    --java_toolchain=@bazel_tools//tools/jdk:toolchain_hostjdk8 \
    --extra_toolchains=//third_party/toolchains/preconfig/ubuntu16.04/gcc7_manylinux2010-nvcc-cuda10.0:toolchain-linux-x86_64 \
    --extra_execution_platforms=@org_tensorflow//third_party/toolchains:rbe_cuda10.0-cudnn7-ubuntu16.04-manylinux2010,@org_tensorflow//third_party/toolchains:rbe_cuda10.0-cudnn7-ubuntu16.04-manylinux2010-gpu \
    --host_platform=@org_tensorflow//third_party/toolchains:rbe_cuda10.0-cudnn7-ubuntu16.04-manylinux2010 \
    --local_test_jobs=4 \
    --remote_timeout=3600 \
    --platforms=@org_tensorflow//third_party/toolchains:rbe_cuda10.0-cudnn7-ubuntu16.04-manylinux2010 \
    -- \
    ${DEFAULT_BAZEL_TARGETS} -//tensorflow/lite/...

  # Copy log to output to be available to GitHub
  ls -la "$(bazel info output_base)/java.log"
  cp "$(bazel info output_base)/java.log" "${KOKORO_ARTIFACTS_DIR}/"
}

source tensorflow/tools/ci_build/release/common.sh
update_bazel_linux
which bazel

run_build
