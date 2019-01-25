# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# tf_core_eager_runtime library
########################################################
file(GLOB_RECURSE tf_core_eager_runtime_srcs
    "${tensorflow_source_dir}/tensorflow/core/common_runtime/eager/*.cc"
    "${tensorflow_source_dir}/tensorflow/core/common_runtime/eager/*.h"
)

file(GLOB_RECURSE tf_core_eager_runtime_exclude_srcs
    "${tensorflow_source_dir}/tensorflow/core/common_runtime/eager/*test*.h"
    "${tensorflow_source_dir}/tensorflow/core/common_runtime/eager/*test*.cc"
)

list(REMOVE_ITEM tf_core_eager_runtime_srcs ${tf_core_eager_runtime_exclude_srcs})

add_library(tf_core_eager_runtime OBJECT ${tf_core_eager_runtime_srcs})
add_dependencies(
	tf_core_eager_runtime 
	tf_c 
	tf_core_lib)


file(GLOB_RECURSE tf_c_eager_srcs
    "${tensorflow_source_dir}/tensorflow/c/eager/*.cc"
    "${tensorflow_source_dir}/tensorflow/c/eager/*.h"
)

file(GLOB_RECURSE tf_c_eager_exlclude_srcs
    "${tensorflow_source_dir}/tensorflow/c/eager/*test*.h"
    "${tensorflow_source_dir}/tensorflow/c/eager/*test*.cc"
)

list(REMOVE_ITEM tf_c_eager_srcs ${tf_c_eager_exlclude_srcs})

add_library(tf_c_eager OBJECT ${tf_c_eager_srcs})
add_dependencies(
  tf_c_eager
  tf_core_eager_runtime
  tf_c
  tf_cc_framework
  tf_cc_while_loop
  tf_core_lib
  tf_protos_cc)