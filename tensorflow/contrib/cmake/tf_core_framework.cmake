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
# RELATIVE_PROTOBUF_GENERATE_CPP function
########################################################
# A variant of PROTOBUF_GENERATE_CPP that keeps the directory hierarchy.
# ROOT_DIR must be absolute, and proto paths must be relative to ROOT_DIR.
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

if(NOT WIN32)
  function(RELATIVE_PROTOBUF_GENERATE_GRPC_CPP SRCS HDRS ROOT_DIR)
    if(NOT ARGN)
      message(SEND_ERROR "Error: RELATIVE_PROTOBUF_GENERATE_GRPC_CPP() called without any proto files")
      return()
    endif()

    set(${SRCS})
    set(${HDRS})
    foreach(FIL ${ARGN})
      set(ABS_FIL ${ROOT_DIR}/${FIL})
      get_filename_component(FIL_WE ${FIL} NAME_WE)
      get_filename_component(FIL_DIR ${ABS_FIL} PATH)
      file(RELATIVE_PATH REL_DIR ${ROOT_DIR} ${FIL_DIR})

      list(APPEND ${SRCS} "${CMAKE_CURRENT_BINARY_DIR}/${REL_DIR}/${FIL_WE}.grpc.pb.cc")
      list(APPEND ${HDRS} "${CMAKE_CURRENT_BINARY_DIR}/${REL_DIR}/${FIL_WE}.grpc.pb.h")
      list(APPEND ${SRCS} "${CMAKE_CURRENT_BINARY_DIR}/${REL_DIR}/${FIL_WE}.pb.cc")
      list(APPEND ${HDRS} "${CMAKE_CURRENT_BINARY_DIR}/${REL_DIR}/${FIL_WE}.pb.h")

      add_custom_command(
        OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${REL_DIR}/${FIL_WE}.grpc.pb.cc"
               "${CMAKE_CURRENT_BINARY_DIR}/${REL_DIR}/${FIL_WE}.grpc.pb.h"
               "${CMAKE_CURRENT_BINARY_DIR}/${REL_DIR}/${FIL_WE}.pb.cc"
               "${CMAKE_CURRENT_BINARY_DIR}/${REL_DIR}/${FIL_WE}.pb.h"
        COMMAND ${PROTOBUF_PROTOC_EXECUTABLE}
        ARGS --grpc_out ${CMAKE_CURRENT_BINARY_DIR} --cpp_out ${CMAKE_CURRENT_BINARY_DIR} --plugin protoc-gen-grpc=${GRPC_BUILD}/grpc_cpp_plugin -I ${ROOT_DIR} ${ABS_FIL} -I ${PROTOBUF_INCLUDE_DIRS}
        DEPENDS ${ABS_FIL} protobuf grpc
        COMMENT "Running C++ protocol buffer grpc compiler on ${FIL}"
        VERBATIM )
    endforeach()

    set_source_files_properties(${${SRCS}} ${${HDRS}} PROPERTIES GENERATED TRUE)
    set(${SRCS} ${${SRCS}} PARENT_SCOPE)
    set(${HDRS} ${${HDRS}} PARENT_SCOPE)
  endfunction()
endif()

function(RELATIVE_PROTOBUF_TEXT_GENERATE_CPP SRCS HDRS ROOT_DIR)
  if(NOT ARGN)
      message(SEND_ERROR "Error: RELATIVE_PROTOBUF_TEXT_GENERATE_CPP() called without any proto files")
    return()
  endif()

  set(${SRCS})
  set(${HDRS})
  foreach(FIL ${ARGN})
    set(ABS_FIL ${ROOT_DIR}/${FIL})
    get_filename_component(FIL_WE ${FIL} NAME_WE)
    get_filename_component(FIL_DIR ${ABS_FIL} PATH)
    file(RELATIVE_PATH REL_DIR ${ROOT_DIR} ${FIL_DIR})

    list(APPEND ${SRCS} "${CMAKE_CURRENT_BINARY_DIR}/${REL_DIR}/${FIL_WE}.pb_text.cc")
    list(APPEND ${HDRS} "${CMAKE_CURRENT_BINARY_DIR}/${REL_DIR}/${FIL_WE}.pb_text.h")

    add_custom_command(
      OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${REL_DIR}/${FIL_WE}.pb_text.cc"
             "${CMAKE_CURRENT_BINARY_DIR}/${REL_DIR}/${FIL_WE}.pb_text.h"
      COMMAND ${PROTO_TEXT_EXE}
      ARGS "${CMAKE_CURRENT_BINARY_DIR}/${REL_DIR}" ${REL_DIR} ${ABS_FIL} "${ROOT_DIR}/tensorflow/tools/proto_text/placeholder.txt"
      DEPENDS ${ABS_FIL} ${PROTO_TEXT_EXE}
      COMMENT "Running C++ protocol buffer text compiler (${PROTO_TEXT_EXE}) on ${FIL}"
      VERBATIM )
  endforeach()

  set_source_files_properties(${${SRCS}} ${${HDRS}} PROPERTIES GENERATED TRUE)
  set(${SRCS} ${${SRCS}} PARENT_SCOPE)
  set(${HDRS} ${${HDRS}} PARENT_SCOPE)
endfunction()

########################################################
# tf_protos_cc library
########################################################

file(GLOB_RECURSE tf_protos_cc_srcs RELATIVE ${tensorflow_source_dir}
    "${tensorflow_source_dir}/tensorflow/core/*.proto"
    "${tensorflow_source_dir}/tensorflow/compiler/xla/*.proto"
    "${tensorflow_source_dir}/tensorflow/contrib/boosted_trees/proto/*.proto"
    "${tensorflow_source_dir}/tensorflow/contrib/tpu/proto/*.proto"
)

RELATIVE_PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS
    ${tensorflow_source_dir} ${tf_protos_cc_srcs}
)


set(PROTO_TEXT_EXE "proto_text")
set(tf_proto_text_srcs
    "tensorflow/core/example/example.proto"
    "tensorflow/core/example/feature.proto"
    "tensorflow/core/framework/allocation_description.proto"
    "tensorflow/core/framework/attr_value.proto"
    "tensorflow/core/framework/cost_graph.proto"
    "tensorflow/core/framework/device_attributes.proto"
    "tensorflow/core/framework/function.proto"
    "tensorflow/core/framework/graph.proto"
    "tensorflow/core/framework/graph_transfer_info.proto"
    "tensorflow/core/framework/kernel_def.proto"
    "tensorflow/core/framework/log_memory.proto"
    "tensorflow/core/framework/node_def.proto"
    "tensorflow/core/framework/op_def.proto"
    "tensorflow/core/framework/remote_fused_graph_execute_info.proto"
    "tensorflow/core/framework/resource_handle.proto"
    "tensorflow/core/framework/step_stats.proto"
    "tensorflow/core/framework/summary.proto"
    "tensorflow/core/framework/tensor.proto"
    "tensorflow/core/framework/tensor_description.proto"
    "tensorflow/core/framework/tensor_shape.proto"
    "tensorflow/core/framework/tensor_slice.proto"
    "tensorflow/core/framework/types.proto"
    "tensorflow/core/framework/versions.proto"
    "tensorflow/core/lib/core/error_codes.proto"
    "tensorflow/core/protobuf/cluster.proto"
    "tensorflow/core/protobuf/config.proto"
    "tensorflow/core/protobuf/debug.proto"
    "tensorflow/core/protobuf/device_properties.proto"
    "tensorflow/core/protobuf/rewriter_config.proto"
    "tensorflow/core/protobuf/tensor_bundle.proto"
    "tensorflow/core/protobuf/saver.proto"
    "tensorflow/core/util/memmapped_file_system.proto"
    "tensorflow/core/util/saved_tensor_slice.proto"
)
RELATIVE_PROTOBUF_TEXT_GENERATE_CPP(PROTO_TEXT_SRCS PROTO_TEXT_HDRS
    ${tensorflow_source_dir} ${tf_proto_text_srcs}
)

if(WIN32)
  add_library(tf_protos_cc ${PROTO_SRCS} ${PROTO_HDRS})
else()
  file(GLOB_RECURSE tf_protos_grpc_cc_srcs RELATIVE ${tensorflow_source_dir}
      "${tensorflow_source_dir}/tensorflow/core/debug/*.proto"
  )
  RELATIVE_PROTOBUF_GENERATE_GRPC_CPP(PROTO_GRPC_SRCS PROTO_GRPC_HDRS
      ${tensorflow_source_dir} ${tf_protos_grpc_cc_srcs}
  )
  add_library(tf_protos_cc ${PROTO_GRPC_SRCS} ${PROTO_GRPC_HDRS} ${PROTO_SRCS} ${PROTO_HDRS})
endif()

########################################################
# tf_core_lib library
########################################################
file(GLOB_RECURSE tf_core_lib_srcs
    "${tensorflow_source_dir}/tensorflow/core/lib/*.h"
    "${tensorflow_source_dir}/tensorflow/core/lib/*.cc"
    "${tensorflow_source_dir}/tensorflow/core/public/*.h"
)

file(GLOB tf_core_platform_srcs
    "${tensorflow_source_dir}/tensorflow/core/platform/*.h"
    "${tensorflow_source_dir}/tensorflow/core/platform/*.cc"
    "${tensorflow_source_dir}/tensorflow/core/platform/default/*.h"
    "${tensorflow_source_dir}/tensorflow/core/platform/default/*.cc"
    "${tensorflow_source_dir}/tensorflow/core/framework/resource_handle.h"
    "${tensorflow_source_dir}/tensorflow/core/framework/resource_handle.cc")
if (NOT tensorflow_ENABLE_GPU)
  file(GLOB tf_core_platform_gpu_srcs
      "${tensorflow_source_dir}/tensorflow/core/platform/cuda_libdevice_path.*"
      "${tensorflow_source_dir}/tensorflow/core/platform/default/cuda_libdevice_path.*")
  list(REMOVE_ITEM tf_core_platform_srcs ${tf_core_platform_gpu_srcs})
else()
  file(GLOB tf_core_platform_srcs_exclude
      "${tensorflow_source_dir}/tensorflow/core/platform/default/device_tracer.cc")
  list(REMOVE_ITEM tf_core_platform_srcs ${tf_core_platform_srcs_exclude})
endif()

list(APPEND tf_core_lib_srcs ${tf_core_platform_srcs})

if(UNIX)
  file(GLOB tf_core_platform_posix_srcs
      "${tensorflow_source_dir}/tensorflow/core/platform/posix/*.h"
      "${tensorflow_source_dir}/tensorflow/core/platform/posix/*.cc"
  )
  list(APPEND tf_core_lib_srcs ${tf_core_platform_posix_srcs})
endif(UNIX)

if(WIN32)
  file(GLOB tf_core_platform_windows_srcs
      "${tensorflow_source_dir}/tensorflow/core/platform/windows/*.h"
      "${tensorflow_source_dir}/tensorflow/core/platform/windows/*.cc"
      "${tensorflow_source_dir}/tensorflow/core/platform/posix/error.h"
      "${tensorflow_source_dir}/tensorflow/core/platform/posix/error.cc"
  )
  list(APPEND tf_core_lib_srcs ${tf_core_platform_windows_srcs})
endif(WIN32)

if (tensorflow_ENABLE_HDFS_SUPPORT)
  list(APPEND tf_core_platform_hdfs_srcs
      "${tensorflow_source_dir}/tensorflow/core/platform/hadoop/hadoop_file_system.cc"
      "${tensorflow_source_dir}/tensorflow/core/platform/hadoop/hadoop_file_system.h"
  )
  list(APPEND tf_core_lib_srcs ${tf_core_platform_hdfs_srcs})
endif()

file(GLOB_RECURSE tf_core_lib_test_srcs
    "${tensorflow_source_dir}/tensorflow/core/lib/*test*.h"
    "${tensorflow_source_dir}/tensorflow/core/lib/*test*.cc"
    "${tensorflow_source_dir}/tensorflow/core/platform/*test*.h"
    "${tensorflow_source_dir}/tensorflow/core/platform/*test*.cc"
    "${tensorflow_source_dir}/tensorflow/core/public/*test*.h"
)
list(REMOVE_ITEM tf_core_lib_srcs ${tf_core_lib_test_srcs})

add_library(tf_core_lib OBJECT ${tf_core_lib_srcs})
add_dependencies(tf_core_lib ${tensorflow_EXTERNAL_DEPENDENCIES} tf_protos_cc)

# Tricky setup to force always rebuilding
# force_rebuild always runs forcing ${VERSION_INFO_CC} target to run
# ${VERSION_INFO_CC} would cache, but it depends on a phony never produced
# target.
# This code forces rebuild every time, not needed as version from git is fetched only once
# move to make.bat which mimicks make.sh

if (NOT WIN32)

  set(VERSION_INFO_CC ${tensorflow_source_dir}/tensorflow/core/util/version_info.cc)
  add_custom_target(force_rebuild_target ALL DEPENDS ${VERSION_INFO_CC})
  add_custom_command(OUTPUT __force_rebuild COMMAND ${CMAKE_COMMAND} -E echo)
  add_custom_command(OUTPUT
      ${VERSION_INFO_CC}
      COMMAND ${PYTHON_EXECUTABLE} ${tensorflow_source_dir}/tensorflow/tools/git/gen_git_source.py
      ARGS --raw_generate ${VERSION_INFO_CC} --source_dir ${tensorflow_source_dir} --git_tag_override=${GIT_TAG_OVERRIDE}
      DEPENDS __force_rebuild)
endif()

set(tf_version_srcs ${tensorflow_source_dir}/tensorflow/core/util/version_info.cc)

########################################################
# tf_core_framework library
########################################################
file(GLOB_RECURSE tf_core_framework_srcs
    "${tensorflow_source_dir}/tensorflow/core/framework/*.h"
    "${tensorflow_source_dir}/tensorflow/core/framework/*.cc"
    "${tensorflow_source_dir}/tensorflow/core/graph/edgeset.h"
    "${tensorflow_source_dir}/tensorflow/core/graph/edgeset.cc"
    "${tensorflow_source_dir}/tensorflow/core/graph/graph.h"
    "${tensorflow_source_dir}/tensorflow/core/graph/graph.cc"
    "${tensorflow_source_dir}/tensorflow/core/graph/graph_def_builder.h"
    "${tensorflow_source_dir}/tensorflow/core/graph/graph_def_builder.cc"
    "${tensorflow_source_dir}/tensorflow/core/graph/node_builder.h"
    "${tensorflow_source_dir}/tensorflow/core/graph/node_builder.cc"
    "${tensorflow_source_dir}/tensorflow/core/graph/tensor_id.h"
    "${tensorflow_source_dir}/tensorflow/core/graph/tensor_id.cc"
    "${tensorflow_source_dir}/tensorflow/core/graph/while_context.h"
    "${tensorflow_source_dir}/tensorflow/core/graph/while_context.cc"
    "${tensorflow_source_dir}/tensorflow/core/util/*.h"
    "${tensorflow_source_dir}/tensorflow/core/util/*.cc"
    "${tensorflow_source_dir}/tensorflow/core/common_runtime/session.cc"
    "${tensorflow_source_dir}/tensorflow/core/common_runtime/session_factory.cc"
    "${tensorflow_source_dir}/tensorflow/core/common_runtime/session_options.cc"
    "${tensorflow_source_dir}/tensorflow/contrib/tensorboard/db/*.cc"
    "${tensorflow_source_dir}/tensorflow/contrib/tensorboard/db/*.h"
    "${tensorflow_source_dir}/public/*.h"
)

file(GLOB_RECURSE tf_core_framework_exclude_srcs
    "${tensorflow_source_dir}/tensorflow/core/framework/*test*.h"
    "${tensorflow_source_dir}/tensorflow/core/framework/*test*.cc"
    "${tensorflow_source_dir}/tensorflow/core/framework/*testutil.h"
    "${tensorflow_source_dir}/tensorflow/core/framework/*testutil.cc"
    "${tensorflow_source_dir}/tensorflow/core/framework/*main.cc"
    "${tensorflow_source_dir}/tensorflow/core/framework/resource_handle.cc"
    "${tensorflow_source_dir}/tensorflow/core/util/*test*.h"
    "${tensorflow_source_dir}/tensorflow/core/util/*test*.cc"
    "${tensorflow_source_dir}/tensorflow/core/util/*main.cc"
    "${tensorflow_source_dir}/tensorflow/contrib/tensorboard/db/*test*.cc"
    "${tensorflow_source_dir}/tensorflow/contrib/tensorboard/db/loader.cc"
    "${tensorflow_source_dir}/tensorflow/contrib/tensorboard/db/vacuum.cc"
    "${tensorflow_source_dir}/tensorflow/core/framework/op_kernel_registry.h"
    "${tensorflow_source_dir}/tensorflow/core/framework/op_kernel_registry.cc"
)

########################################################
# tf_core_kernel_registry library
########################################################
file(GLOB_RECURSE tf_core_kernel_registry_srcs
    "${tensorflow_source_dir}/tensorflow/core/framework/op_kernel_registry.h"
    "${tensorflow_source_dir}/tensorflow/core/framework/op_kernel_registry.cc"
)

# TODO(jart): Why doesn't this work?
# set_source_files_properties(
#     ${tensorflow_source_dir}/tensorflow/contrib/tensorboard/db/snapfn.cc
#     PROPERTIES COMPILE_FLAGS -DSQLITE_OMIT_LOAD_EXTENSION)

list(REMOVE_ITEM tf_core_framework_srcs ${tf_core_framework_exclude_srcs})

add_library(tf_core_kernel_registry SHARED ${tf_core_kernel_registry_srcs} )
add_dependencies( tf_core_kernel_registry tf_core_lib )

add_library(tf_core_framework OBJECT
    ${tf_core_framework_srcs}
    ${tf_version_srcs}
    ${PROTO_TEXT_HDRS}
    ${PROTO_TEXT_SRCS})
add_dependencies(tf_core_framework
    tf_core_lib
    proto_text
)


