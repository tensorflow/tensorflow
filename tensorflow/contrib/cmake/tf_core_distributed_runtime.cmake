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
# tf_core_distributed_runtime library
########################################################
file(GLOB_RECURSE tf_core_distributed_runtime_srcs
   "${tensorflow_source_dir}/tensorflow/core/distributed_runtime/*.h"
   "${tensorflow_source_dir}/tensorflow/core/distributed_runtime/*.cc"
)

file(GLOB_RECURSE tf_core_distributed_runtime_exclude_srcs
    "${tensorflow_source_dir}/tensorflow/core/distributed_runtime/server_lib.cc"  # Build in tf_core_cpu instead.
    "${tensorflow_source_dir}/tensorflow/core/distributed_runtime/*test*.h"
    "${tensorflow_source_dir}/tensorflow/core/distributed_runtime/*test*.cc"
    "${tensorflow_source_dir}/tensorflow/core/distributed_runtime/rpc/grpc_tensorflow_server.cc"
)

list(REMOVE_ITEM tf_core_distributed_runtime_srcs ${tf_core_distributed_runtime_exclude_srcs})

add_library(tf_core_distributed_runtime OBJECT ${tf_core_distributed_runtime_srcs})

add_dependencies(tf_core_distributed_runtime
    tf_core_cpu grpc
)

########################################################
# grpc_tensorflow_server executable
########################################################
set(grpc_tensorflow_server_srcs
    "${tensorflow_source_dir}/tensorflow/core/distributed_runtime/rpc/grpc_tensorflow_server.cc"
)

add_executable(grpc_tensorflow_server
    ${grpc_tensorflow_server_srcs}
    $<TARGET_OBJECTS:tf_core_lib>
    $<TARGET_OBJECTS:tf_core_cpu>
    $<TARGET_OBJECTS:tf_core_framework>
    $<TARGET_OBJECTS:tf_core_kernels>
    $<TARGET_OBJECTS:tf_cc_framework>
    $<TARGET_OBJECTS:tf_cc_ops>
    $<TARGET_OBJECTS:tf_core_ops>
    $<TARGET_OBJECTS:tf_core_direct_session>
    $<TARGET_OBJECTS:tf_core_distributed_runtime>
    $<$<BOOL:${tensorflow_ENABLE_GPU}>:$<TARGET_OBJECTS:tf_stream_executor>>
)

target_link_libraries(grpc_tensorflow_server PUBLIC
    tf_protos_cc
    ${tf_core_gpu_kernels_lib}
    ${tensorflow_EXTERNAL_LIBRARIES}
)
