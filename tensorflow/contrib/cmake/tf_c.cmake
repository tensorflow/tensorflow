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
########################################################
# tf_c_framework library
########################################################
set(tf_c_srcs
    "${tensorflow_source_dir}/tensorflow/c/c_api.cc"
    "${tensorflow_source_dir}/tensorflow/c/c_api.h"
    "${tensorflow_source_dir}/tensorflow/c/c_api_function.cc"
    "${tensorflow_source_dir}/tensorflow/c/eager/c_api.cc"
    "${tensorflow_source_dir}/tensorflow/c/eager/c_api.h"
    "${tensorflow_source_dir}/tensorflow/c/eager/tape.h"
    "${tensorflow_source_dir}/tensorflow/c/eager/runtime.cc"
    "${tensorflow_source_dir}/tensorflow/c/eager/runtime.h"
    "${tensorflow_source_dir}/tensorflow/c/checkpoint_reader.cc"
    "${tensorflow_source_dir}/tensorflow/c/checkpoint_reader.h"
    "${tensorflow_source_dir}/tensorflow/c/tf_status_helper.cc"
    "${tensorflow_source_dir}/tensorflow/c/tf_status_helper.h"
)

add_library(tf_c OBJECT ${tf_c_srcs})
add_dependencies(
  tf_c
  tf_cc_framework
  tf_cc_while_loop
  tf_core_lib
  tf_protos_cc)

if(tensorflow_BUILD_PYTHON_BINDINGS)
  add_library(tf_c_python_api OBJECT
    "${tensorflow_source_dir}/tensorflow/c/python_api.cc"
    "${tensorflow_source_dir}/tensorflow/c/python_api.h"
  )
  add_dependencies(
    tf_c_python_api
    tf_c
    tf_core_lib
    tf_core_framework
    tf_protos_cc)
endif()
