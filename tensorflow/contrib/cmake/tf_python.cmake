# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# CMake rules for generating the TensorFlow Python bindings.
#
# Known limitations:
# * Generates output in a hard-coded path ${CMAKE_CURRENT_BINARY_DIR}/tf_python.
# * No support for dynamic library loading.
# * Limited support for tf.contrib.
#
# The _pywrap_tensorflow_internal target builds everything.

########################################################
# Resolve installed dependencies
########################################################

# 1. Resolve the installed version of Python (for Python.h and python).
# TODO(mrry): Parameterize the build script to enable Python 3 building.
if(NOT PYTHON_INCLUDE_DIR)
  set(PYTHON_NOT_FOUND false)
  exec_program("${PYTHON_EXECUTABLE}"
    ARGS "-c \"import distutils.sysconfig; print(distutils.sysconfig.get_python_inc())\""
    OUTPUT_VARIABLE PYTHON_INCLUDE_DIR
    RETURN_VALUE PYTHON_NOT_FOUND)
  if(${PYTHON_NOT_FOUND})
    message(FATAL_ERROR
            "Cannot get Python include directory. Is distutils installed?")
  endif(${PYTHON_NOT_FOUND})
endif(NOT PYTHON_INCLUDE_DIR)
FIND_PACKAGE(PythonLibs)

# 2. Resolve the installed version of NumPy (for numpy/arrayobject.h).
if(NOT NUMPY_INCLUDE_DIR)
  set(NUMPY_NOT_FOUND false)
  exec_program("${PYTHON_EXECUTABLE}"
    ARGS "-c \"import numpy; print(numpy.get_include())\""
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    RETURN_VALUE NUMPY_NOT_FOUND)
  if(${NUMPY_NOT_FOUND})
    message(FATAL_ERROR
            "Cannot get NumPy include directory: Is NumPy installed?")
  endif(${NUMPY_NOT_FOUND})
endif(NOT NUMPY_INCLUDE_DIR)


########################################################
# Build the Python directory structure.
########################################################

# TODO(mrry): Configure this to build in a directory other than tf_python/

# Generates the Python protobuf wrappers.
# ROOT_DIR must be absolute; subsequent arguments are interpreted as
# paths of .proto files, and must be relative to ROOT_DIR.
function(RELATIVE_PROTOBUF_GENERATE_PYTHON ROOT_DIR SRCS)
  if(NOT ARGN)
    message(SEND_ERROR "Error: RELATIVE_PROTOBUF_GENERATE_PYTHON() called without any proto files")
    return()
  endif()

  set(${SRCS})
  foreach(FIL ${ARGN})
    set(ABS_FIL ${ROOT_DIR}/${FIL})
    get_filename_component(FIL_WE ${FIL} NAME_WE)
    get_filename_component(FIL_DIR ${ABS_FIL} PATH)
    file(RELATIVE_PATH REL_DIR ${ROOT_DIR} ${FIL_DIR})

    list(APPEND ${SRCS} "${CMAKE_CURRENT_BINARY_DIR}/tf_python/${REL_DIR}/${FIL_WE}_pb2.py")
    add_custom_command(
      OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/tf_python/${REL_DIR}/${FIL_WE}_pb2.py"
      COMMAND ${PROTOBUF_PROTOC_EXECUTABLE}
      ARGS --python_out  ${CMAKE_CURRENT_BINARY_DIR}/tf_python/ -I ${ROOT_DIR} -I ${PROTOBUF_INCLUDE_DIRS} ${ABS_FIL}
      DEPENDS ${PROTOBUF_PROTOC_EXECUTABLE} protobuf
      COMMENT "Running Python protocol buffer compiler on ${FIL}"
      VERBATIM )
  endforeach()
  set(${SRCS} ${${SRCS}} PARENT_SCOPE)
endfunction()

function(RELATIVE_PROTOBUF_GENERATE_CPP SRCS HDRS ROOT_DIR)
  if(NOT ARGN)
    message(SEND_ERROR "Error: RELATIVE_PROTOBUF_GENERATE_CPP() called without any proto files")
    return()
  endif()

  set(${SRCS})
  set(${HDRS})
  foreach(FIL ${ARGN})
    set(ABS_FIL ${ROOT_DIR}/${FIL})
    get_filename_component(FIL_WE ${FIL} NAME_WE)
    get_filename_component(FIL_DIR ${ABS_FIL} PATH)
    file(RELATIVE_PATH REL_DIR ${ROOT_DIR} ${FIL_DIR})

    list(APPEND ${SRCS} "${CMAKE_CURRENT_BINARY_DIR}/${REL_DIR}/${FIL_WE}.pb.cc")
    list(APPEND ${HDRS} "${CMAKE_CURRENT_BINARY_DIR}/${REL_DIR}/${FIL_WE}.pb.h")

    add_custom_command(
      OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${REL_DIR}/${FIL_WE}.pb.cc"
             "${CMAKE_CURRENT_BINARY_DIR}/${REL_DIR}/${FIL_WE}.pb.h"
      COMMAND  ${PROTOBUF_PROTOC_EXECUTABLE}
      ARGS --cpp_out  ${CMAKE_CURRENT_BINARY_DIR} -I ${ROOT_DIR} ${ABS_FIL} -I ${PROTOBUF_INCLUDE_DIRS}
      DEPENDS ${ABS_FIL} protobuf
      COMMENT "Running C++ protocol buffer compiler on ${FIL}"
      VERBATIM )
  endforeach()

  set_source_files_properties(${${SRCS}} ${${HDRS}} PROPERTIES GENERATED TRUE)
  set(${SRCS} ${${SRCS}} PARENT_SCOPE)
  set(${HDRS} ${${HDRS}} PARENT_SCOPE)
endfunction()

file(GLOB_RECURSE tf_protos_python_srcs RELATIVE ${tensorflow_source_dir}
    "${tensorflow_source_dir}/tensorflow/core/*.proto"
    "${tensorflow_source_dir}/tensorflow/core/profiler/*.proto"
    "${tensorflow_source_dir}/tensorflow/python/*.proto"
    "${tensorflow_source_dir}/tensorflow/contrib/boosted_trees/proto/*.proto"
    "${tensorflow_source_dir}/tensorflow/contrib/decision_trees/proto/*.proto"
    "${tensorflow_source_dir}/tensorflow/contrib/session_bundle/*.proto"
    "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/proto/*.proto"
    "${tensorflow_source_dir}/tensorflow/contrib/tensorboard/*.proto"
    "${tensorflow_source_dir}/tensorflow/contrib/tpu/profiler/*.proto"
    "${tensorflow_source_dir}/tensorflow/contrib/training/*.proto"
)
RELATIVE_PROTOBUF_GENERATE_PYTHON(
    ${tensorflow_source_dir} PYTHON_PROTO_GENFILES ${tf_protos_python_srcs}
)

# NOTE(mrry): Avoid regenerating the tensorflow/core protos because this
# can cause benign-but-failing-on-Windows-due-to-file-locking conflicts
# when two rules attempt to generate the same file.
file(GLOB_RECURSE tf_python_protos_cc_srcs RELATIVE ${tensorflow_source_dir}
    "${tensorflow_source_dir}/tensorflow/core/profiler/*.proto"
    "${tensorflow_source_dir}/tensorflow/python/*.proto"
    "${tensorflow_source_dir}/tensorflow/contrib/session_bundle/*.proto"
    "${tensorflow_source_dir}/tensorflow/contrib/tensorboard/*.proto"
    "${tensorflow_source_dir}/tensorflow/contrib/training/*.proto"
)
RELATIVE_PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS
    ${tensorflow_source_dir} ${tf_python_protos_cc_srcs}
)

add_library(tf_python_protos_cc ${PROTO_SRCS} ${PROTO_HDRS})
add_dependencies(tf_python_protos_cc tf_protos_cc)

# tf_python_touchup_modules adds empty __init__.py files to all
# directories containing Python code, so that Python will recognize
# them as modules.
add_custom_target(tf_python_touchup_modules)

# tf_python_copy_scripts_to_destination copies all Python files
# (including static source and generated protobuf wrappers, but *not*
# generated TensorFlow op wrappers) into tf_python/.
add_custom_target(tf_python_copy_scripts_to_destination DEPENDS tf_python_touchup_modules)


# tf_python_srcs contains all static .py files
function(add_python_module MODULE_NAME)
    set(options DONTCOPY)
    cmake_parse_arguments(ADD_PYTHON_MODULE "${options}" "" "" ${ARGN})
    add_custom_command(TARGET tf_python_touchup_modules PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/tf_python/${MODULE_NAME}")
    add_custom_command(TARGET tf_python_touchup_modules PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E touch "${CMAKE_CURRENT_BINARY_DIR}/tf_python/${MODULE_NAME}/__init__.py")
    file(GLOB module_python_srcs RELATIVE ${tensorflow_source_dir}
        "${tensorflow_source_dir}/${MODULE_NAME}/*.py"
    )
    if(NOT ${ADD_PYTHON_MODULE_DONTCOPY})
        foreach(script ${module_python_srcs})
            get_filename_component(REL_DIR ${script} DIRECTORY)
            # NOTE(mrry): This rule may exclude modules that should be part of
            # the distributed PIP package
            # (e.g. tensorflow/contrib/testing/python/framework/util_test.py),
            # so we currently add explicit commands to include those files
            # later on in this script.
            if (NOT "${script}" MATCHES "_test\.py$")
	        add_custom_command(TARGET tf_python_copy_scripts_to_destination PRE_BUILD
                  COMMAND ${CMAKE_COMMAND} -E copy ${tensorflow_source_dir}/${script} ${CMAKE_CURRENT_BINARY_DIR}/tf_python/${script})
            endif()
        endforeach()
    endif()
endfunction()

add_python_module("tensorflow")
add_python_module("tensorflow/core")
add_python_module("tensorflow/core/example")
add_python_module("tensorflow/core/framework")
add_python_module("tensorflow/core/lib")
add_python_module("tensorflow/core/lib/core")
add_python_module("tensorflow/core/protobuf")
add_python_module("tensorflow/core/util")
add_python_module("tensorflow/examples")
add_python_module("tensorflow/examples/tutorials")
add_python_module("tensorflow/examples/tutorials/mnist")
add_python_module("tensorflow/python")
add_python_module("tensorflow/python/client")
add_python_module("tensorflow/python/debug")
add_python_module("tensorflow/python/debug/cli")
add_python_module("tensorflow/python/debug/examples")
add_python_module("tensorflow/python/debug/lib")
add_python_module("tensorflow/python/debug/wrappers")
add_python_module("tensorflow/python/eager")
add_python_module("tensorflow/python/estimator")
add_python_module("tensorflow/python/estimator/canned")
add_python_module("tensorflow/python/estimator/export")
add_python_module("tensorflow/python/estimator/inputs")
add_python_module("tensorflow/python/estimator/inputs/queues")
add_python_module("tensorflow/python/feature_column")
add_python_module("tensorflow/python/framework")
add_python_module("tensorflow/python/grappler")
add_python_module("tensorflow/python/keras")
add_python_module("tensorflow/python/keras/activations")
add_python_module("tensorflow/python/keras/applications")
add_python_module("tensorflow/python/keras/applications/inception_v3")
add_python_module("tensorflow/python/keras/applications/mobilenet")
add_python_module("tensorflow/python/keras/applications/resnet50")
add_python_module("tensorflow/python/keras/applications/vgg16")
add_python_module("tensorflow/python/keras/applications/vgg19")
add_python_module("tensorflow/python/keras/applications/xception")
add_python_module("tensorflow/python/keras/backend")
add_python_module("tensorflow/python/keras/callbacks")
add_python_module("tensorflow/python/keras/constraints")
add_python_module("tensorflow/python/keras/datasets")
add_python_module("tensorflow/python/keras/datasets/boston_housing")
add_python_module("tensorflow/python/keras/datasets/cifar10")
add_python_module("tensorflow/python/keras/datasets/cifar100")
add_python_module("tensorflow/python/keras/datasets/imdb")
add_python_module("tensorflow/python/keras/datasets/mnist")
add_python_module("tensorflow/python/keras/datasets/reuters")
add_python_module("tensorflow/python/keras/initializers")
add_python_module("tensorflow/python/keras/layers")
add_python_module("tensorflow/python/keras/losses")
add_python_module("tensorflow/python/keras/metrics")
add_python_module("tensorflow/python/keras/models")
add_python_module("tensorflow/python/keras/optimizers")
add_python_module("tensorflow/python/keras/preprocessing")
add_python_module("tensorflow/python/keras/preprocessing/image")
add_python_module("tensorflow/python/keras/preprocessing/sequence")
add_python_module("tensorflow/python/keras/preprocessing/text")
add_python_module("tensorflow/python/keras/regularizers")
add_python_module("tensorflow/python/keras/utils")
add_python_module("tensorflow/python/keras/wrappers")
add_python_module("tensorflow/python/keras/wrappers/scikit_learn")
add_python_module("tensorflow/python/keras/_impl")
add_python_module("tensorflow/python/keras/_impl/keras")
add_python_module("tensorflow/python/keras/_impl/keras/applications")
add_python_module("tensorflow/python/keras/_impl/keras/datasets")
add_python_module("tensorflow/python/keras/_impl/keras/engine")
add_python_module("tensorflow/python/keras/_impl/keras/layers")
add_python_module("tensorflow/python/keras/_impl/keras/preprocessing")
add_python_module("tensorflow/python/keras/_impl/keras/utils")
add_python_module("tensorflow/python/keras/_impl/keras/wrappers")
add_python_module("tensorflow/python/kernel_tests")
add_python_module("tensorflow/python/kernel_tests/distributions")
add_python_module("tensorflow/python/layers")
add_python_module("tensorflow/python/lib")
add_python_module("tensorflow/python/lib/core")
add_python_module("tensorflow/python/lib/io")
add_python_module("tensorflow/python/ops")
add_python_module("tensorflow/python/ops/distributions")
add_python_module("tensorflow/python/ops/losses")
add_python_module("tensorflow/python/platform")
add_python_module("tensorflow/python/platform/default")
add_python_module("tensorflow/python/platform/summary")
add_python_module("tensorflow/python/profiler/")
add_python_module("tensorflow/python/profiler/internal")
add_python_module("tensorflow/python/saved_model")
add_python_module("tensorflow/python/summary")
add_python_module("tensorflow/python/summary/writer")
add_python_module("tensorflow/python/tools")
add_python_module("tensorflow/python/training")
add_python_module("tensorflow/python/user_ops")
add_python_module("tensorflow/python/util")
add_python_module("tensorflow/python/util/protobuf")
add_python_module("tensorflow/tools")
add_python_module("tensorflow/tools/graph_transforms")
add_python_module("tensorflow/contrib")
add_python_module("tensorflow/contrib/android")
add_python_module("tensorflow/contrib/android/java")
add_python_module("tensorflow/contrib/android/java/org")
add_python_module("tensorflow/contrib/android/java/org/tensorflow")
add_python_module("tensorflow/contrib/android/java/org/tensorflow/contrib")
add_python_module("tensorflow/contrib/android/java/org/tensorflow/contrib/android")
add_python_module("tensorflow/contrib/android/jni")
add_python_module("tensorflow/contrib/bayesflow")
add_python_module("tensorflow/contrib/bayesflow/examples")
add_python_module("tensorflow/contrib/bayesflow/examples/reinforce_simple")
add_python_module("tensorflow/contrib/bayesflow/python")
add_python_module("tensorflow/contrib/bayesflow/python/kernel_tests")
add_python_module("tensorflow/contrib/bayesflow/python/ops")
add_python_module("tensorflow/contrib/boosted_trees")
add_python_module("tensorflow/contrib/boosted_trees/estimator_batch")
add_python_module("tensorflow/contrib/boosted_trees/ops")
add_python_module("tensorflow/contrib/boosted_trees/proto")
add_python_module("tensorflow/contrib/boosted_trees/python")
add_python_module("tensorflow/contrib/boosted_trees/python/kernel_tests")
add_python_module("tensorflow/contrib/boosted_trees/python/ops")
add_python_module("tensorflow/contrib/cloud")
add_python_module("tensorflow/contrib/cloud/kernels")
add_python_module("tensorflow/contrib/cloud/ops")
add_python_module("tensorflow/contrib/cloud/python")
add_python_module("tensorflow/contrib/cloud/python/ops")
add_python_module("tensorflow/contrib/cluster_resolver")
add_python_module("tensorflow/contrib/cluster_resolver/python")
add_python_module("tensorflow/contrib/cluster_resolver/python/training")
add_python_module("tensorflow/contrib/compiler")
add_python_module("tensorflow/contrib/copy_graph")
add_python_module("tensorflow/contrib/copy_graph/python")
add_python_module("tensorflow/contrib/copy_graph/python/util")
add_python_module("tensorflow/contrib/crf")
add_python_module("tensorflow/contrib/crf/python")
add_python_module("tensorflow/contrib/crf/python/kernel_tests")
add_python_module("tensorflow/contrib/crf/python/ops")
add_python_module("tensorflow/contrib/cudnn_rnn")
add_python_module("tensorflow/contrib/cudnn_rnn/kernels")
add_python_module("tensorflow/contrib/cudnn_rnn/ops")
add_python_module("tensorflow/contrib/cudnn_rnn/python")
add_python_module("tensorflow/contrib/cudnn_rnn/python/kernel_tests")
add_python_module("tensorflow/contrib/cudnn_rnn/python/ops")
add_python_module("tensorflow/contrib/data")
add_python_module("tensorflow/contrib/data/python")
add_python_module("tensorflow/contrib/data/python/framework")
add_python_module("tensorflow/contrib/data/python/kernel_tests")
add_python_module("tensorflow/contrib/data/python/ops")
add_python_module("tensorflow/contrib/data/python/util")
add_python_module("tensorflow/contrib/decision_trees")
add_python_module("tensorflow/contrib/decision_trees/proto")
add_python_module("tensorflow/contrib/deprecated")
add_python_module("tensorflow/contrib/distributions")
add_python_module("tensorflow/contrib/distributions/python")
add_python_module("tensorflow/contrib/distributions/python/kernel_tests")
add_python_module("tensorflow/contrib/distributions/python/ops")
add_python_module("tensorflow/contrib/distributions/python/ops/bijectors")
add_python_module("tensorflow/contrib/estimator")
add_python_module("tensorflow/contrib/estimator/python")
add_python_module("tensorflow/contrib/estimator/python/estimator")
add_python_module("tensorflow/contrib/factorization")
add_python_module("tensorflow/contrib/factorization/examples")
add_python_module("tensorflow/contrib/factorization/kernels")
add_python_module("tensorflow/contrib/factorization/ops")
add_python_module("tensorflow/contrib/factorization/python")
add_python_module("tensorflow/contrib/factorization/python/kernel_tests")
add_python_module("tensorflow/contrib/factorization/python/ops")
add_python_module("tensorflow/contrib/ffmpeg")
add_python_module("tensorflow/contrib/ffmpeg/default")
add_python_module("tensorflow/contrib/ffmpeg/testdata")
add_python_module("tensorflow/contrib/framework")
add_python_module("tensorflow/contrib/framework/kernels")
add_python_module("tensorflow/contrib/framework/ops")
add_python_module("tensorflow/contrib/framework/python")
add_python_module("tensorflow/contrib/framework/python/framework")
add_python_module("tensorflow/contrib/framework/python/ops")
add_python_module("tensorflow/contrib/gan")
add_python_module("tensorflow/contrib/gan/python")
add_python_module("tensorflow/contrib/gan/python/features")
add_python_module("tensorflow/contrib/gan/python/features/python")
add_python_module("tensorflow/contrib/gan/python/losses")
add_python_module("tensorflow/contrib/gan/python/losses/python")
add_python_module("tensorflow/contrib/graph_editor")
add_python_module("tensorflow/contrib/graph_editor/examples")
add_python_module("tensorflow/contrib/graph_editor/tests")
add_python_module("tensorflow/contrib/grid_rnn")
add_python_module("tensorflow/contrib/grid_rnn/python")
add_python_module("tensorflow/contrib/grid_rnn/python/kernel_tests")
add_python_module("tensorflow/contrib/grid_rnn/python/ops")
add_python_module("tensorflow/contrib/hooks")
add_python_module("tensorflow/contrib/image")
add_python_module("tensorflow/contrib/image/ops")
add_python_module("tensorflow/contrib/image/python")
add_python_module("tensorflow/contrib/image/python/ops")
add_python_module("tensorflow/contrib/input_pipeline")
add_python_module("tensorflow/contrib/input_pipeline/ops")
add_python_module("tensorflow/contrib/input_pipeline/python")
add_python_module("tensorflow/contrib/input_pipeline/python/ops")
add_python_module("tensorflow/contrib/integrate")
add_python_module("tensorflow/contrib/integrate/python")
add_python_module("tensorflow/contrib/integrate/python/ops")
add_python_module("tensorflow/contrib/ios_examples")
add_python_module("tensorflow/contrib/ios_examples/benchmark")
add_python_module("tensorflow/contrib/ios_examples/benchmark/benchmark.xcodeproj")
add_python_module("tensorflow/contrib/ios_examples/benchmark/data")
add_python_module("tensorflow/contrib/ios_examples/camera")
add_python_module("tensorflow/contrib/ios_examples/camera/camera_example.xcodeproj")
add_python_module("tensorflow/contrib/ios_examples/camera/en.lproj")
add_python_module("tensorflow/contrib/ios_examples/simple")
add_python_module("tensorflow/contrib/ios_examples/simple/data")
add_python_module("tensorflow/contrib/ios_examples/simple/tf_ios_makefile_example.xcodeproj")
add_python_module("tensorflow/contrib/keras")
add_python_module("tensorflow/contrib/keras/api")
add_python_module("tensorflow/contrib/keras/api/keras")
add_python_module("tensorflow/contrib/keras/api/keras/activations")
add_python_module("tensorflow/contrib/keras/api/keras/applications")
add_python_module("tensorflow/contrib/keras/api/keras/applications/inception_v3")
add_python_module("tensorflow/contrib/keras/api/keras/applications/mobilenet")
add_python_module("tensorflow/contrib/keras/api/keras/applications/resnet50")
add_python_module("tensorflow/contrib/keras/api/keras/applications/vgg16")
add_python_module("tensorflow/contrib/keras/api/keras/applications/vgg19")
add_python_module("tensorflow/contrib/keras/api/keras/applications/xception")
add_python_module("tensorflow/contrib/keras/api/keras/backend")
add_python_module("tensorflow/contrib/keras/api/keras/callbacks")
add_python_module("tensorflow/contrib/keras/api/keras/constraints")
add_python_module("tensorflow/contrib/keras/api/keras/datasets")
add_python_module("tensorflow/contrib/keras/api/keras/datasets/boston_housing")
add_python_module("tensorflow/contrib/keras/api/keras/datasets/cifar10")
add_python_module("tensorflow/contrib/keras/api/keras/datasets/cifar100")
add_python_module("tensorflow/contrib/keras/api/keras/datasets/imdb")
add_python_module("tensorflow/contrib/keras/api/keras/datasets/mnist")
add_python_module("tensorflow/contrib/keras/api/keras/datasets/reuters")
add_python_module("tensorflow/contrib/keras/api/keras/initializers")
add_python_module("tensorflow/contrib/keras/api/keras/layers")
add_python_module("tensorflow/contrib/keras/api/keras/losses")
add_python_module("tensorflow/contrib/keras/api/keras/metrics")
add_python_module("tensorflow/contrib/keras/api/keras/models")
add_python_module("tensorflow/contrib/keras/api/keras/optimizers")
add_python_module("tensorflow/contrib/keras/api/keras/preprocessing")
add_python_module("tensorflow/contrib/keras/api/keras/preprocessing/image")
add_python_module("tensorflow/contrib/keras/api/keras/preprocessing/sequence")
add_python_module("tensorflow/contrib/keras/api/keras/preprocessing/text")
add_python_module("tensorflow/contrib/keras/api/keras/regularizers")
add_python_module("tensorflow/contrib/keras/api/keras/utils")
add_python_module("tensorflow/contrib/keras/api/keras/wrappers")
add_python_module("tensorflow/contrib/keras/api/keras/wrappers/scikit_learn")
add_python_module("tensorflow/contrib/keras/python")
add_python_module("tensorflow/contrib/keras/python/keras")
add_python_module("tensorflow/contrib/keras/python/keras/applications")
add_python_module("tensorflow/contrib/keras/python/keras/datasets")
add_python_module("tensorflow/contrib/keras/python/keras/engine")
add_python_module("tensorflow/contrib/keras/python/keras/layers")
add_python_module("tensorflow/contrib/keras/python/keras/preprocessing")
add_python_module("tensorflow/contrib/keras/python/keras/utils")
add_python_module("tensorflow/contrib/keras/python/keras/wrappers")
add_python_module("tensorflow/contrib/kernel_methods")
add_python_module("tensorflow/contrib/kernel_methods/python")
add_python_module("tensorflow/contrib/kernel_methods/python/mappers")
add_python_module("tensorflow/contrib/labeled_tensor")
add_python_module("tensorflow/contrib/labeled_tensor/python")
add_python_module("tensorflow/contrib/labeled_tensor/python/ops")
add_python_module("tensorflow/contrib/layers")
add_python_module("tensorflow/contrib/layers/kernels")
add_python_module("tensorflow/contrib/layers/ops")
add_python_module("tensorflow/contrib/layers/python")
add_python_module("tensorflow/contrib/layers/python/kernel_tests")
add_python_module("tensorflow/contrib/layers/python/layers")
add_python_module("tensorflow/contrib/layers/python/ops")
add_python_module("tensorflow/contrib/learn")
add_python_module("tensorflow/contrib/learn/python")
add_python_module("tensorflow/contrib/learn/python/learn")
add_python_module("tensorflow/contrib/learn/python/learn/dataframe")
add_python_module("tensorflow/contrib/learn/python/learn/dataframe/queues")
add_python_module("tensorflow/contrib/learn/python/learn/dataframe/transforms")
add_python_module("tensorflow/contrib/learn/python/learn/datasets")
add_python_module("tensorflow/contrib/learn/python/learn/datasets/data")
add_python_module("tensorflow/contrib/learn/python/learn/estimators")
add_python_module("tensorflow/contrib/learn/python/learn/learn_io")
add_python_module("tensorflow/contrib/learn/python/learn/ops")
add_python_module("tensorflow/contrib/learn/python/learn/preprocessing")
add_python_module("tensorflow/contrib/learn/python/learn/preprocessing/tests")
add_python_module("tensorflow/contrib/learn/python/learn/tests")
add_python_module("tensorflow/contrib/learn/python/learn/tests/dataframe")
add_python_module("tensorflow/contrib/learn/python/learn/utils")
add_python_module("tensorflow/contrib/legacy_seq2seq")
add_python_module("tensorflow/contrib/legacy_seq2seq/python")
add_python_module("tensorflow/contrib/legacy_seq2seq/python/ops")
add_python_module("tensorflow/contrib/linalg")
add_python_module("tensorflow/contrib/linalg/python")
add_python_module("tensorflow/contrib/linalg/python/ops")
add_python_module("tensorflow/contrib/linalg/python/kernel_tests")
add_python_module("tensorflow/contrib/linear_optimizer")
add_python_module("tensorflow/contrib/linear_optimizer/kernels")
add_python_module("tensorflow/contrib/linear_optimizer/kernels/g3doc")
add_python_module("tensorflow/contrib/linear_optimizer/python")
add_python_module("tensorflow/contrib/linear_optimizer/python/kernel_tests")
add_python_module("tensorflow/contrib/linear_optimizer/python/ops")
add_python_module("tensorflow/contrib/lookup")
add_python_module("tensorflow/contrib/losses")
add_python_module("tensorflow/contrib/losses/python")
add_python_module("tensorflow/contrib/losses/python/losses")
add_python_module("tensorflow/contrib/makefile")
add_python_module("tensorflow/contrib/makefile/test")
add_python_module("tensorflow/contrib/memory_stats")
add_python_module("tensorflow/contrib/memory_stats/kernels")
add_python_module("tensorflow/contrib/memory_stats/ops")
add_python_module("tensorflow/contrib/memory_stats/python")
add_python_module("tensorflow/contrib/memory_stats/python/kernel_tests")
add_python_module("tensorflow/contrib/memory_stats/python/ops")
add_python_module("tensorflow/contrib/meta_graph_transform")
add_python_module("tensorflow/contrib/metrics")
add_python_module("tensorflow/contrib/metrics/kernels")
add_python_module("tensorflow/contrib/metrics/ops")
add_python_module("tensorflow/contrib/metrics/python")
add_python_module("tensorflow/contrib/metrics/python/kernel_tests")
add_python_module("tensorflow/contrib/metrics/python/metrics")
add_python_module("tensorflow/contrib/metrics/python/ops")
add_python_module("tensorflow/contrib/ndlstm")
add_python_module("tensorflow/contrib/ndlstm/python")
add_python_module("tensorflow/contrib/nn")
add_python_module("tensorflow/contrib/nn/python")
add_python_module("tensorflow/contrib/nn/python/ops")
add_python_module("tensorflow/contrib/nccl")
add_python_module("tensorflow/contrib/nccl/kernels")
add_python_module("tensorflow/contrib/nccl/ops")
add_python_module("tensorflow/contrib/nccl/python")
add_python_module("tensorflow/contrib/nccl/python/ops")
add_python_module("tensorflow/contrib/nearest_neighbor/kernels")
add_python_module("tensorflow/contrib/nearest_neighbor/ops")
add_python_module("tensorflow/contrib/nearest_neighbor/python")
add_python_module("tensorflow/contrib/nearest_neighbor/python/kernel_tests")
add_python_module("tensorflow/contrib/nearest_neighbor/python/ops")
add_python_module("tensorflow/contrib/opt")
add_python_module("tensorflow/contrib/opt/python")
add_python_module("tensorflow/contrib/opt/python/training")
add_python_module("tensorflow/contrib/pi_examples")
add_python_module("tensorflow/contrib/pi_examples/camera")
add_python_module("tensorflow/contrib/pi_examples/label_image")
add_python_module("tensorflow/contrib/pi_examples/label_image/data")
add_python_module("tensorflow/contrib/predictor")
add_python_module("tensorflow/contrib/quantization")
add_python_module("tensorflow/contrib/quantization/python")
add_python_module("tensorflow/contrib/remote_fused_graph/pylib")
add_python_module("tensorflow/contrib/remote_fused_graph/pylib/python")
add_python_module("tensorflow/contrib/remote_fused_graph/pylib/python/ops")
add_python_module("tensorflow/contrib/resampler")
add_python_module("tensorflow/contrib/resampler/kernels")
add_python_module("tensorflow/contrib/resampler/ops")
add_python_module("tensorflow/contrib/resampler/python")
add_python_module("tensorflow/contrib/resampler/python/ops")
add_python_module("tensorflow/contrib/rnn")
add_python_module("tensorflow/contrib/rnn/kernels")
add_python_module("tensorflow/contrib/rnn/ops")
add_python_module("tensorflow/contrib/rnn/python")
add_python_module("tensorflow/contrib/rnn/python/kernel_tests")
add_python_module("tensorflow/contrib/rnn/python/ops")
add_python_module("tensorflow/contrib/saved_model")
add_python_module("tensorflow/contrib/saved_model/python")
add_python_module("tensorflow/contrib/saved_model/python/saved_model")
add_python_module("tensorflow/contrib/seq2seq")
add_python_module("tensorflow/contrib/seq2seq/kernels")
add_python_module("tensorflow/contrib/seq2seq/ops")
add_python_module("tensorflow/contrib/seq2seq/python")
add_python_module("tensorflow/contrib/seq2seq/python/kernel_tests")
add_python_module("tensorflow/contrib/seq2seq/python/ops")
add_python_module("tensorflow/contrib/session_bundle")
add_python_module("tensorflow/contrib/session_bundle/example")
add_python_module("tensorflow/contrib/session_bundle/testdata")
add_python_module("tensorflow/contrib/signal")
add_python_module("tensorflow/contrib/signal/python")
add_python_module("tensorflow/contrib/signal/python/ops")
add_python_module("tensorflow/contrib/slim")
add_python_module("tensorflow/contrib/slim/python")
add_python_module("tensorflow/contrib/slim/python/slim")
add_python_module("tensorflow/contrib/slim/python/slim/data")
add_python_module("tensorflow/contrib/slim/python/slim/nets")
add_python_module("tensorflow/contrib/solvers")
add_python_module("tensorflow/contrib/solvers/python")
add_python_module("tensorflow/contrib/solvers/python/ops")
add_python_module("tensorflow/contrib/sparsemax")
add_python_module("tensorflow/contrib/sparsemax/python")
add_python_module("tensorflow/contrib/sparsemax/python/ops")
add_python_module("tensorflow/contrib/specs")
add_python_module("tensorflow/contrib/specs/python")
add_python_module("tensorflow/contrib/staging")
add_python_module("tensorflow/contrib/stat_summarizer")
add_python_module("tensorflow/contrib/stateless")
add_python_module("tensorflow/contrib/tensorboard")
add_python_module("tensorflow/contrib/tensorboard/plugins")
add_python_module("tensorflow/contrib/tensorboard/plugins/projector")
add_python_module("tensorflow/contrib/tensor_forest")
add_python_module("tensorflow/contrib/tensor_forest/client")
add_python_module("tensorflow/contrib/tensor_forest/core")
add_python_module("tensorflow/contrib/tensor_forest/core/ops")
add_python_module("tensorflow/contrib/tensor_forest/data")
add_python_module("tensorflow/contrib/tensor_forest/hybrid")
add_python_module("tensorflow/contrib/tensor_forest/hybrid/core")
add_python_module("tensorflow/contrib/tensor_forest/hybrid/core/ops")
add_python_module("tensorflow/contrib/tensor_forest/hybrid/ops")
add_python_module("tensorflow/contrib/tensor_forest/hybrid/python")
add_python_module("tensorflow/contrib/tensor_forest/hybrid/python/kernel_tests")
add_python_module("tensorflow/contrib/tensor_forest/hybrid/python/layers")
add_python_module("tensorflow/contrib/tensor_forest/hybrid/python/models")
add_python_module("tensorflow/contrib/tensor_forest/hybrid/python/ops")
add_python_module("tensorflow/contrib/tensor_forest/python")
add_python_module("tensorflow/contrib/tensor_forest/python/kernel_tests")
add_python_module("tensorflow/contrib/tensor_forest/python/ops")
add_python_module("tensorflow/contrib/testing")
add_python_module("tensorflow/contrib/testing/python")
add_python_module("tensorflow/contrib/testing/python/framework")
add_python_module("tensorflow/contrib/text")
add_python_module("tensorflow/contrib/text/kernels")
add_python_module("tensorflow/contrib/text/ops")
add_python_module("tensorflow/contrib/text/python")
add_python_module("tensorflow/contrib/text/python/ops")
add_python_module("tensorflow/contrib/tfprof")
add_python_module("tensorflow/contrib/timeseries")
add_python_module("tensorflow/contrib/timeseries/examples")
add_python_module("tensorflow/contrib/timeseries/examples/data")
add_python_module("tensorflow/contrib/timeseries/python")
add_python_module("tensorflow/contrib/timeseries/python/timeseries")
add_python_module("tensorflow/contrib/timeseries/python/timeseries/state_space_models")
add_python_module("tensorflow/contrib/tpu")
add_python_module("tensorflow/contrib/tpu/ops")
add_python_module("tensorflow/contrib/tpu/profiler")
add_python_module("tensorflow/contrib/tpu/python")
add_python_module("tensorflow/contrib/tpu/python/ops")
add_python_module("tensorflow/contrib/tpu/python/profiler")
add_python_module("tensorflow/contrib/tpu/python/tpu")
add_python_module("tensorflow/contrib/training")
add_python_module("tensorflow/contrib/training/python")
add_python_module("tensorflow/contrib/training/python/training")
add_python_module("tensorflow/contrib/util")
add_python_module("tensorflow/contrib/reduce_slice_ops")
add_python_module("tensorflow/contrib/reduce_slice_ops/kernels")
add_python_module("tensorflow/contrib/reduce_slice_ops/ops")
add_python_module("tensorflow/contrib/reduce_slice_ops/python")
add_python_module("tensorflow/contrib/reduce_slice_ops/python/kernel_tests")
add_python_module("tensorflow/contrib/reduce_slice_ops/python/ops")


########################################################
# tf_python_op_gen_main library
########################################################
set(tf_python_op_gen_main_srcs
    "${tensorflow_source_dir}/tensorflow/python/eager/python_eager_op_gen.h"
    "${tensorflow_source_dir}/tensorflow/python/eager/python_eager_op_gen.cc"
    "${tensorflow_source_dir}/tensorflow/python/framework/python_op_gen.cc"
    "${tensorflow_source_dir}/tensorflow/python/framework/python_op_gen.cc"
    "${tensorflow_source_dir}/tensorflow/python/framework/python_op_gen_main.cc"
    "${tensorflow_source_dir}/tensorflow/python/framework/python_op_gen.h"
    "${tensorflow_source_dir}/tensorflow/python/framework/python_op_gen_internal.h"
)

add_library(tf_python_op_gen_main OBJECT ${tf_python_op_gen_main_srcs})

add_dependencies(tf_python_op_gen_main tf_core_framework)

# create directory for ops generated files
set(python_ops_target_dir ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/python/ops)

set(tf_python_ops_generated_files)

set(tf_python_op_lib_names
    ${tf_op_lib_names}
    "user_ops"
)

function(GENERATE_PYTHON_OP_LIB tf_python_op_lib_name)
    set(options SHAPE_FUNCTIONS_NOT_REQUIRED)
    set(oneValueArgs DESTINATION)
    set(multiValueArgs ADDITIONAL_LIBRARIES)
    cmake_parse_arguments(GENERATE_PYTHON_OP_LIB
      "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    if(NOT DEFINED GENERATE_PYTHON_OP_LIB_DESTINATION)
      # Default destination is tf_python/tensorflow/python/ops/gen_<...>.py.
      set(GENERATE_PYTHON_OP_LIB_DESTINATION
          "${python_ops_target_dir}/gen_${tf_python_op_lib_name}.py")
    endif()
    if(GENERATE_PYTHON_OP_LIB_SHAPE_FUNCTIONS_NOT_REQUIRED)
      set(require_shape_fn 0)
    else()
      set(require_shape_fn 1)
    endif()

    # Create a C++ executable that links in the appropriate op
    # registrations and generates Python wrapper code based on the
    # registered ops.
    add_executable(${tf_python_op_lib_name}_gen_python
        $<TARGET_OBJECTS:tf_python_op_gen_main>
        $<TARGET_OBJECTS:tf_${tf_python_op_lib_name}>
        $<TARGET_OBJECTS:tf_core_lib>
        $<TARGET_OBJECTS:tf_core_framework>
        ${GENERATE_PYTHON_OP_LIB_ADDITIONAL_LIBRARIES}
    )
    target_link_libraries(${tf_python_op_lib_name}_gen_python PRIVATE
        tf_protos_cc
				tf_python_protos_cc
        ${tensorflow_EXTERNAL_LIBRARIES}
    )

    # Use the generated C++ executable to create a Python file
    # containing the wrappers.
    add_custom_command(
      OUTPUT ${GENERATE_PYTHON_OP_LIB_DESTINATION}
      COMMAND ${tf_python_op_lib_name}_gen_python @${tensorflow_source_dir}/tensorflow/python/ops/hidden_ops.txt ${require_shape_fn} > ${GENERATE_PYTHON_OP_LIB_DESTINATION}
      DEPENDS ${tf_python_op_lib_name}_gen_python
    )

    set(tf_python_ops_generated_files ${tf_python_ops_generated_files}
        ${GENERATE_PYTHON_OP_LIB_DESTINATION} PARENT_SCOPE)
endfunction()

GENERATE_PYTHON_OP_LIB("array_ops")
GENERATE_PYTHON_OP_LIB("bitwise_ops")
GENERATE_PYTHON_OP_LIB("math_ops")
GENERATE_PYTHON_OP_LIB("functional_ops")
GENERATE_PYTHON_OP_LIB("candidate_sampling_ops")
GENERATE_PYTHON_OP_LIB("checkpoint_ops")
GENERATE_PYTHON_OP_LIB("control_flow_ops"
  ADDITIONAL_LIBRARIES $<TARGET_OBJECTS:tf_no_op>)
GENERATE_PYTHON_OP_LIB("ctc_ops")
GENERATE_PYTHON_OP_LIB("data_flow_ops")
GENERATE_PYTHON_OP_LIB("dataset_ops")
GENERATE_PYTHON_OP_LIB("image_ops")
GENERATE_PYTHON_OP_LIB("io_ops")
GENERATE_PYTHON_OP_LIB("linalg_ops")
GENERATE_PYTHON_OP_LIB("logging_ops")
GENERATE_PYTHON_OP_LIB("lookup_ops")
GENERATE_PYTHON_OP_LIB("nn_ops")
GENERATE_PYTHON_OP_LIB("parsing_ops")
GENERATE_PYTHON_OP_LIB("random_ops")
GENERATE_PYTHON_OP_LIB("remote_fused_graph_ops"
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/remote_fused_graph/pylib/python/ops/gen_remote_fused_graph_ops.py)
GENERATE_PYTHON_OP_LIB("resource_variable_ops")
GENERATE_PYTHON_OP_LIB("script_ops")
GENERATE_PYTHON_OP_LIB("sdca_ops")
GENERATE_PYTHON_OP_LIB("set_ops")
GENERATE_PYTHON_OP_LIB("state_ops")
GENERATE_PYTHON_OP_LIB("sparse_ops")
GENERATE_PYTHON_OP_LIB("spectral_ops")
GENERATE_PYTHON_OP_LIB("string_ops")
GENERATE_PYTHON_OP_LIB("user_ops")
GENERATE_PYTHON_OP_LIB("training_ops"
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/python/training/gen_training_ops.py)

GENERATE_PYTHON_OP_LIB("contrib_boosted_trees_model_ops"
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/boosted_trees/python/ops/gen_model_ops.py)
GENERATE_PYTHON_OP_LIB("contrib_boosted_trees_split_handler_ops"
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/boosted_trees/python/ops/gen_split_handler_ops.py)
GENERATE_PYTHON_OP_LIB("contrib_boosted_trees_training_ops"
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/boosted_trees/python/ops/gen_training_ops.py)
GENERATE_PYTHON_OP_LIB("contrib_boosted_trees_prediction_ops"
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/boosted_trees/python/ops/gen_prediction_ops.py)
GENERATE_PYTHON_OP_LIB("contrib_boosted_trees_quantiles_ops"
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/boosted_trees/python/ops/gen_quantile_ops.py)
GENERATE_PYTHON_OP_LIB("contrib_boosted_trees_ensemble_optimzier_ops"
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/boosted_trees/python/ops/gen_ensemble_optimizer_ops.py)
GENERATE_PYTHON_OP_LIB("contrib_boosted_trees_stats_accumulator_ops"
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/boosted_trees/python/ops/gen_stats_accumulator_ops.py)
GENERATE_PYTHON_OP_LIB("contrib_cudnn_rnn_ops"
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/cudnn_rnn/ops/gen_cudnn_rnn_ops.py)
GENERATE_PYTHON_OP_LIB("contrib_factorization_clustering_ops"
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/factorization/python/ops/gen_clustering_ops.py)
GENERATE_PYTHON_OP_LIB("contrib_factorization_factorization_ops"
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/factorization/python/ops/gen_factorization_ops.py)
GENERATE_PYTHON_OP_LIB("contrib_framework_variable_ops"
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/framework/python/ops/gen_variable_ops.py)
GENERATE_PYTHON_OP_LIB("contrib_input_pipeline_ops"
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/input_pipeline/ops/gen_input_pipeline_ops.py)
GENERATE_PYTHON_OP_LIB("contrib_image_ops"
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/image/ops/gen_image_ops.py)
GENERATE_PYTHON_OP_LIB("contrib_layers_sparse_feature_cross_ops"
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/layers/ops/gen_sparse_feature_cross_op.py)
GENERATE_PYTHON_OP_LIB("contrib_memory_stats_ops"
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/memory_stats/ops/gen_memory_stats_ops.py)
GENERATE_PYTHON_OP_LIB("contrib_nccl_ops"
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/nccl/ops/gen_nccl_ops.py)
GENERATE_PYTHON_OP_LIB("contrib_nearest_neighbor_ops"
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/nearest_neighbor/ops/gen_nearest_neighbor_ops.py)
GENERATE_PYTHON_OP_LIB("contrib_resampler_ops"
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/resampler/ops/gen_resampler_ops.py)
GENERATE_PYTHON_OP_LIB("contrib_rnn_gru_ops"
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/rnn/ops/gen_gru_ops.py)
GENERATE_PYTHON_OP_LIB("contrib_rnn_lstm_ops"
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/rnn/ops/gen_lstm_ops.py)
GENERATE_PYTHON_OP_LIB("contrib_seq2seq_beam_search_ops"
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/seq2seq/ops/gen_beam_search_ops.py)
GENERATE_PYTHON_OP_LIB("contrib_tensor_forest_ops"
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/tensor_forest/python/ops/gen_tensor_forest_ops.py)
GENERATE_PYTHON_OP_LIB("contrib_tensor_forest_hybrid_ops"
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/tensor_forest/hybrid/ops/gen_training_ops.py)
GENERATE_PYTHON_OP_LIB("contrib_tensor_forest_model_ops"
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/tensor_forest/python/ops/gen_model_ops.py)
GENERATE_PYTHON_OP_LIB("contrib_tensor_forest_stats_ops"
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/tensor_forest/python/ops/gen_stats_ops.py)
GENERATE_PYTHON_OP_LIB("contrib_text_skip_gram_ops"
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/text/python/ops/gen_skip_gram_ops.py)
GENERATE_PYTHON_OP_LIB("contrib_bigquery_reader_ops"
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/cloud/python/ops/gen_bigquery_reader_ops.py)
GENERATE_PYTHON_OP_LIB("stateless_random_ops"
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/stateless/gen_stateless_random_ops.py)
GENERATE_PYTHON_OP_LIB("debug_ops"
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/python/debug/ops/gen_debug_ops.py)

add_custom_target(tf_python_ops SOURCES ${tf_python_ops_generated_files} ${PYTHON_PROTO_GENFILES})
add_dependencies(tf_python_ops tf_python_op_gen_main)


############################################################
# Build the SWIG-wrapped library for the TensorFlow runtime.
############################################################

find_package(SWIG REQUIRED)
# Generate the C++ and Python source code for the SWIG wrapper.
# NOTE(mrry): We always regenerate the SWIG wrapper, which means that we must
# always re-link the Python extension, but we don't have to track the
# individual headers on which the SWIG wrapper depends.
add_custom_command(
      OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/python/pywrap_tensorflow_internal.py"
             "${CMAKE_CURRENT_BINARY_DIR}/pywrap_tensorflow_internal.cc"
      DEPENDS tf_python_touchup_modules __force_rebuild
      COMMAND ${SWIG_EXECUTABLE}
      ARGS -python -c++
           -I${tensorflow_source_dir}
           -I${CMAKE_CURRENT_BINARY_DIR}
           -module pywrap_tensorflow_internal
           -outdir ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/python
           -o ${CMAKE_CURRENT_BINARY_DIR}/pywrap_tensorflow_internal.cc
           -globals ''
           ${tensorflow_source_dir}/tensorflow/python/tensorflow.i
      COMMENT "Running SWIG to generate Python wrappers"
      VERBATIM )

set (pywrap_tensorflow_internal_src
    "${tensorflow_source_dir}/tensorflow/core/profiler/internal/print_model_analysis.h"
    "${tensorflow_source_dir}/tensorflow/core/profiler/internal/print_model_analysis.cc"
    "${tensorflow_source_dir}/tensorflow/python/eager/pywrap_tfe.h"
    "${tensorflow_source_dir}/tensorflow/python/eager/pywrap_tfe_src.cc"
    "${tensorflow_source_dir}/tensorflow/python/client/tf_session_helper.h"
    "${tensorflow_source_dir}/tensorflow/python/client/tf_session_helper.cc"
    "${tensorflow_source_dir}/tensorflow/python/eager/python_eager_op_gen.h"
    "${tensorflow_source_dir}/tensorflow/python/eager/python_eager_op_gen.cc"
    "${tensorflow_source_dir}/tensorflow/python/framework/cpp_shape_inference.h"
    "${tensorflow_source_dir}/tensorflow/python/framework/cpp_shape_inference.cc"
    "${tensorflow_source_dir}/tensorflow/python/framework/python_op_gen.h"
    "${tensorflow_source_dir}/tensorflow/python/framework/python_op_gen.cc"
    "${tensorflow_source_dir}/tensorflow/python/lib/core/numpy.h"
    "${tensorflow_source_dir}/tensorflow/python/lib/core/numpy.cc"
    "${tensorflow_source_dir}/tensorflow/python/lib/core/ndarray_tensor.h"
    "${tensorflow_source_dir}/tensorflow/python/lib/core/ndarray_tensor.cc"
    "${tensorflow_source_dir}/tensorflow/python/lib/core/ndarray_tensor_bridge.h"
    "${tensorflow_source_dir}/tensorflow/python/lib/core/ndarray_tensor_bridge.cc"
    "${tensorflow_source_dir}/tensorflow/python/lib/core/py_func.h"
    "${tensorflow_source_dir}/tensorflow/python/lib/core/py_func.cc"
    "${tensorflow_source_dir}/tensorflow/python/lib/core/safe_ptr.h"
    "${tensorflow_source_dir}/tensorflow/python/lib/core/safe_ptr.cc"
    "${tensorflow_source_dir}/tensorflow/python/lib/io/py_record_reader.h"
    "${tensorflow_source_dir}/tensorflow/python/lib/io/py_record_reader.cc"
    "${tensorflow_source_dir}/tensorflow/python/lib/io/py_record_writer.h"
    "${tensorflow_source_dir}/tensorflow/python/lib/io/py_record_writer.cc"
    "${tensorflow_source_dir}/tensorflow/python/util/kernel_registry.h"
    "${tensorflow_source_dir}/tensorflow/python/util/kernel_registry.cc"
    "${tensorflow_source_dir}/tensorflow/cc/framework/ops.cc"
    "${tensorflow_source_dir}/tensorflow/cc/framework/scope.cc"
    "${CMAKE_CURRENT_BINARY_DIR}/pywrap_tensorflow_internal.cc"
)

if(WIN32)
    # Windows: build a static library with the same objects as tensorflow.dll.
    # This can be used to build for a standalone exe and also helps us to
    # find all symbols that need to be exported from the dll which is needed
    # to provide the tensorflow c/c++ api in tensorflow.dll.
    # From the static library we create the def file with all symbols that need to
    # be exported from tensorflow.dll. Because there is a limit of 64K sybmols
    # that can be exported, we filter the symbols with a python script to the namespaces
    # we need.
    #
    add_library(pywrap_tensorflow_internal_static STATIC
        ${pywrap_tensorflow_internal_src}
        $<TARGET_OBJECTS:tf_c>
        $<TARGET_OBJECTS:tf_c_python_api>
        $<TARGET_OBJECTS:tf_core_lib>
        $<TARGET_OBJECTS:tf_core_cpu>
        $<TARGET_OBJECTS:tf_core_framework>
        $<TARGET_OBJECTS:tf_core_profiler>
        $<TARGET_OBJECTS:tf_cc>
        $<TARGET_OBJECTS:tf_cc_ops>
        $<TARGET_OBJECTS:tf_cc_while_loop>
        $<TARGET_OBJECTS:tf_core_ops>
        $<TARGET_OBJECTS:tf_core_direct_session>
        $<TARGET_OBJECTS:tf_grappler>
        $<TARGET_OBJECTS:tf_tools_transform_graph_lib>
        $<$<BOOL:${tensorflow_ENABLE_GRPC_SUPPORT}>:$<TARGET_OBJECTS:tf_core_distributed_runtime>>
        $<TARGET_OBJECTS:tf_core_kernels>
        $<$<BOOL:${tensorflow_ENABLE_GPU}>:$<TARGET_OBJECTS:tf_core_kernels_cpu_only>>
        $<$<BOOL:${tensorflow_ENABLE_GPU}>:$<TARGET_OBJECTS:tf_stream_executor>>
    )

    target_include_directories(pywrap_tensorflow_internal_static PUBLIC
        ${PYTHON_INCLUDE_DIR}
        ${NUMPY_INCLUDE_DIR}
    )
    #target_link_libraries(pywrap_tensorflow_internal_static
    #	tf_protos_cc
    #	tf_python_protos_cc
    #)
    add_dependencies(pywrap_tensorflow_internal_static tf_protos_cc tf_python_protos_cc)
    set(pywrap_tensorflow_internal_static_dependencies
        $<TARGET_FILE:pywrap_tensorflow_internal_static>
        $<TARGET_FILE:tf_protos_cc>
        $<TARGET_FILE:tf_python_protos_cc>
    )

    set(pywrap_tensorflow_deffile "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/pywrap_tensorflow.def")
    set_source_files_properties(${pywrap_tensorflow_deffile} PROPERTIES GENERATED TRUE)

    add_custom_command(TARGET pywrap_tensorflow_internal_static POST_BUILD
        COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/create_def_file.py
            --input "${pywrap_tensorflow_internal_static_dependencies}"
            --output "${pywrap_tensorflow_deffile}"
            --target _pywrap_tensorflow_internal.pyd
    )
endif(WIN32)

# pywrap_tensorflow_internal is a shared library containing all of the
# TensorFlow runtime and the standard ops and kernels. These are installed into
# tf_python/tensorflow/python/.
add_library(pywrap_tensorflow_internal SHARED
    ${pywrap_tensorflow_internal_src}
    $<TARGET_OBJECTS:tf_c>
    $<TARGET_OBJECTS:tf_c_python_api>
    $<TARGET_OBJECTS:tf_core_lib>
    $<TARGET_OBJECTS:tf_core_cpu>
    $<TARGET_OBJECTS:tf_core_framework>
    $<TARGET_OBJECTS:tf_core_profiler>
    $<TARGET_OBJECTS:tf_cc>
    $<TARGET_OBJECTS:tf_cc_ops>
    $<TARGET_OBJECTS:tf_cc_while_loop>
    $<TARGET_OBJECTS:tf_core_ops>
    $<TARGET_OBJECTS:tf_core_direct_session>
    $<TARGET_OBJECTS:tf_grappler>
    $<TARGET_OBJECTS:tf_tools_transform_graph_lib>
    $<$<BOOL:${tensorflow_ENABLE_GRPC_SUPPORT}>:$<TARGET_OBJECTS:tf_core_distributed_runtime>>
    $<TARGET_OBJECTS:tf_core_kernels>
    $<$<BOOL:${tensorflow_ENABLE_GPU}>:$<TARGET_OBJECTS:tf_core_kernels_cpu_only>>
    $<$<BOOL:${tensorflow_ENABLE_GPU}>:$<TARGET_OBJECTS:tf_stream_executor>>
    ${pywrap_tensorflow_deffile}
)

if(WIN32)
    add_dependencies(pywrap_tensorflow_internal pywrap_tensorflow_internal_static)
endif(WIN32)

target_include_directories(pywrap_tensorflow_internal PUBLIC
    ${PYTHON_INCLUDE_DIR}
    ${NUMPY_INCLUDE_DIR}
)

target_link_libraries(pywrap_tensorflow_internal PRIVATE
    ${tf_core_gpu_kernels_lib}
    ${tensorflow_EXTERNAL_LIBRARIES}
    tf_protos_cc
    tf_python_protos_cc
    ${PYTHON_LIBRARIES}
)

if(WIN32)
    # include contrib/nearest_neighbor as .so
    #
    set(tf_nearest_neighbor_srcs
        "${tensorflow_source_dir}/tensorflow/contrib/nearest_neighbor/kernels/heap.h"
        "${tensorflow_source_dir}/tensorflow/contrib/nearest_neighbor/kernels/hyperplane_lsh_probes.h"
        "${tensorflow_source_dir}/tensorflow/contrib/nearest_neighbor/kernels/hyperplane_lsh_probes.cc"
        "${tensorflow_source_dir}/tensorflow/contrib/nearest_neighbor/ops/nearest_neighbor_ops.cc"
    )

    AddUserOps(TARGET _nearest_neighbor_ops
        SOURCES "${tf_nearest_neighbor_srcs}"
        DEPENDS pywrap_tensorflow_internal tf_python_ops
        DISTCOPY ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/nearest_neighbor/python/ops/)
endif(WIN32)

if(WIN32)
    # include contrib/rnn as .so
    #
    set(tf_gru_srcs
        "${tensorflow_source_dir}/tensorflow/contrib/rnn/kernels/blas_gemm.cc"
        "${tensorflow_source_dir}/tensorflow/contrib/rnn/kernels/blas_gemm.h"
        "${tensorflow_source_dir}/tensorflow/contrib/rnn/kernels/gru_ops.cc"
        "${tensorflow_source_dir}/tensorflow/contrib/rnn/kernels/gru_ops.h"
        "${tensorflow_source_dir}/tensorflow/contrib/rnn/ops/gru_ops.cc"
    )
    set(tf_gru_gpu_srcs
        "${tensorflow_source_dir}/tensorflow/contrib/rnn/kernels/gru_ops_gpu.cu.cc"
    )

    set(tf_lstm_srcs
        "${tensorflow_source_dir}/tensorflow/contrib/rnn/kernels/blas_gemm.cc"
        "${tensorflow_source_dir}/tensorflow/contrib/rnn/kernels/blas_gemm.h"
        "${tensorflow_source_dir}/tensorflow/contrib/rnn/kernels/lstm_ops.cc"
        "${tensorflow_source_dir}/tensorflow/contrib/rnn/kernels/lstm_ops.h"
        "${tensorflow_source_dir}/tensorflow/contrib/rnn/ops/lstm_ops.cc"
    )
    set(tf_lstm_gpu_srcs
        "${tensorflow_source_dir}/tensorflow/contrib/rnn/kernels/lstm_ops_gpu.cu.cc"
    )

    AddUserOps(TARGET _gru_ops
        SOURCES "${tf_gru_srcs}"
        GPUSOURCES ${tf_gru_gpu_srcs}
        DEPENDS pywrap_tensorflow_internal tf_python_ops
        DISTCOPY ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/rnn/python/ops/)

    AddUserOps(TARGET _lstm_ops
        SOURCES "${tf_lstm_srcs}"
        GPUSOURCES ${tf_lstm_gpu_srcs}
        DEPENDS pywrap_tensorflow_internal tf_python_ops
        DISTCOPY ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/rnn/python/ops/)
endif(WIN32)

if(WIN32)
    # include contrib/seq2seq as .so
    #
    set(tf_beam_search_srcs
        "${tensorflow_source_dir}/tensorflow/contrib/seq2seq/kernels/beam_search_ops.cc"
        "${tensorflow_source_dir}/tensorflow/contrib/seq2seq/kernels/beam_search_ops.h"
        "${tensorflow_source_dir}/tensorflow/contrib/seq2seq/ops/beam_search_ops.cc"
    )

    set(tf_beam_search_gpu_srcs
        "${tensorflow_source_dir}/tensorflow/contrib/seq2seq/kernels/beam_search_ops_gpu.cu.cc"
    )

    AddUserOps(TARGET _beam_search_ops
        SOURCES "${tf_beam_search_srcs}"
        GPUSOURCES ${tf_beam_search_gpu_srcs}
        DEPENDS pywrap_tensorflow_internal tf_python_ops
        DISTCOPY ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/seq2seq/python/ops/)
endif(WIN32)

############################################################
# Build a PIP package containing the TensorFlow runtime.
############################################################
add_custom_target(tf_python_build_pip_package)
add_dependencies(tf_python_build_pip_package
    pywrap_tensorflow_internal
    tf_python_copy_scripts_to_destination
    tf_python_touchup_modules
    tf_python_ops
    tf_extension_ops)

# Fix-up Python files that were not included by the add_python_module() macros.
add_custom_command(TARGET tf_python_build_pip_package POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy ${tensorflow_source_dir}/tensorflow/tools/pip_package/setup.py
                                   ${CMAKE_CURRENT_BINARY_DIR}/tf_python/)
# This file is unfortunately excluded by the regex that excludes *_test.py
# files, but it is imported into tf.contrib, so we add it explicitly.
add_custom_command(TARGET tf_python_copy_scripts_to_destination PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy ${tensorflow_source_dir}/tensorflow/contrib/testing/python/framework/util_test.py
                                   ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/testing/python/framework/)

if(WIN32)
  add_custom_command(TARGET tf_python_build_pip_package POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/$(Configuration)/pywrap_tensorflow_internal.dll
                                     ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/python/_pywrap_tensorflow_internal.pyd
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/$(Configuration)/pywrap_tensorflow_internal.lib
                                     ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/python/)
else()
  add_custom_command(TARGET tf_python_build_pip_package POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/libpywrap_tensorflow_internal.so
                                     ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/python/_pywrap_tensorflow_internal.so)
endif()
add_custom_command(TARGET tf_python_build_pip_package POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy ${tensorflow_source_dir}/tensorflow/tools/pip_package/README
                                   ${CMAKE_CURRENT_BINARY_DIR}/tf_python/)
add_custom_command(TARGET tf_python_build_pip_package POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy ${tensorflow_source_dir}/tensorflow/tools/pip_package/MANIFEST.in
                                   ${CMAKE_CURRENT_BINARY_DIR}/tf_python/)

# Copy datasets for tf.contrib.learn.
add_custom_command(TARGET tf_python_build_pip_package POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy ${tensorflow_source_dir}/tensorflow/contrib/learn/python/learn/datasets/data/boston_house_prices.csv
                                   ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/learn/python/learn/datasets/data/)
add_custom_command(TARGET tf_python_build_pip_package POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy ${tensorflow_source_dir}/tensorflow/contrib/learn/python/learn/datasets/data/iris.csv
                                   ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/learn/python/learn/datasets/data/)
add_custom_command(TARGET tf_python_build_pip_package POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy ${tensorflow_source_dir}/tensorflow/contrib/learn/python/learn/datasets/data/text_test.csv
                                   ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/learn/python/learn/datasets/data/)
add_custom_command(TARGET tf_python_build_pip_package POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy ${tensorflow_source_dir}/tensorflow/contrib/learn/python/learn/datasets/data/text_train.csv
                                   ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/contrib/learn/python/learn/datasets/data/)

# Create include header directory
add_custom_command(TARGET tf_python_build_pip_package PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory
  ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/include/)

# tensorflow headers
add_custom_command(TARGET tf_python_build_pip_package PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory
  ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/include/tensorflow)
add_custom_command(TARGET tf_python_build_pip_package PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory
  ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/include/tensorflow/core)
add_custom_command(TARGET tf_python_build_pip_package PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory
  ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/include/tensorflow/stream_executor)
add_custom_command(TARGET tf_python_build_pip_package POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${tensorflow_source_dir}/tensorflow/core
                                   ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/include/tensorflow/core)
add_custom_command(TARGET tf_python_build_pip_package POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_BINARY_DIR}/tensorflow/core
                                   ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/include/tensorflow/core)
add_custom_command(TARGET tf_python_build_pip_package POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${tensorflow_source_dir}/tensorflow/stream_executor
                                   ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/include/tensorflow/stream_executor)

# google protobuf headers
add_custom_command(TARGET tf_python_build_pip_package PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory
  ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/include/google)
add_custom_command(TARGET tf_python_build_pip_package POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_BINARY_DIR}/protobuf/src/protobuf/src/google
                                   ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/include/google)

# Eigen directory
add_custom_command(TARGET tf_python_build_pip_package PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory
  ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/include/Eigen)
add_custom_command(TARGET tf_python_build_pip_package POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_BINARY_DIR}/eigen/src/eigen/Eigen
                                   ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/include/Eigen)

# external directory
add_custom_command(TARGET tf_python_build_pip_package PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory
  ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/include/external)
add_custom_command(TARGET tf_python_build_pip_package PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory
  ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/include/external/eigen_archive)
add_custom_command(TARGET tf_python_build_pip_package POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_BINARY_DIR}/external/eigen_archive
                                   ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/include/external/eigen_archive)

# third_party eigen directory
add_custom_command(TARGET tf_python_build_pip_package PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory
  ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/include/third_party)
add_custom_command(TARGET tf_python_build_pip_package PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory
  ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/include/third_party/eigen3)
add_custom_command(TARGET tf_python_build_pip_package POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${tensorflow_source_dir}/third_party/eigen3
                                   ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/include/third_party/eigen3)

# unsupported Eigen directory
add_custom_command(TARGET tf_python_build_pip_package PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory
  ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/include/unsupported)
add_custom_command(TARGET tf_python_build_pip_package PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory
  ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/include/unsupported/Eigen)
add_custom_command(TARGET tf_python_build_pip_package POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_BINARY_DIR}/eigen/src/eigen/unsupported/Eigen
                                   ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/include/unsupported/Eigen)

if(${tensorflow_TF_NIGHTLY})
  if(${tensorflow_ENABLE_GPU})
    add_custom_command(TARGET tf_python_build_pip_package POST_BUILD
      COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/tf_python/setup.py bdist_wheel --project_name tf_nightly_gpu
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tf_python)
  else()
    add_custom_command(TARGET tf_python_build_pip_package POST_BUILD
      COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/tf_python/setup.py bdist_wheel --project_name tf_nightly
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tf_python)
  endif(${tensorflow_ENABLE_GPU})
else()
  if(${tensorflow_ENABLE_GPU})
    add_custom_command(TARGET tf_python_build_pip_package POST_BUILD
      COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/tf_python/setup.py bdist_wheel --project_name tensorflow_gpu
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tf_python)
  else()
    add_custom_command(TARGET tf_python_build_pip_package POST_BUILD
      COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/tf_python/setup.py bdist_wheel
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tf_python)
  endif(${tensorflow_ENABLE_GPU})
endif(${tensorflow_TF_NIGHTLY})
