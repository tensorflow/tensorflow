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
# tf_core_direct_session library
########################################################
file(GLOB tf_core_direct_session_srcs
   "${tensorflow_source_dir}/tensorflow/core/common_runtime/direct_session.cc"
   "${tensorflow_source_dir}/tensorflow/core/common_runtime/direct_session.h"
)

file(GLOB_RECURSE tf_core_direct_session_test_srcs
    "${tensorflow_source_dir}/tensorflow/core/debug/*test*.h"
    "${tensorflow_source_dir}/tensorflow/core/debug/*test*.cc"
)

list(REMOVE_ITEM tf_core_direct_session_srcs ${tf_core_direct_session_test_srcs})

add_library(tf_core_direct_session OBJECT ${tf_core_direct_session_srcs})

add_dependencies(tf_core_direct_session tf_core_cpu)