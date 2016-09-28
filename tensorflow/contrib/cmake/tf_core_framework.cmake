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

include_directories(${PROTOBUF_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
file(GLOB_RECURSE tf_protos_cc_srcs RELATIVE ${tensorflow_source_dir}
    "${tensorflow_source_dir}/tensorflow/core/*.proto"
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
    "tensorflow/core/framework/kernel_def.proto"
    "tensorflow/core/framework/log_memory.proto"
    "tensorflow/core/framework/node_def.proto"
    "tensorflow/core/framework/op_def.proto"
    "tensorflow/core/framework/step_stats.proto"
    "tensorflow/core/framework/summary.proto"
    "tensorflow/core/framework/tensor.proto"
    "tensorflow/core/framework/tensor_description.proto"
    "tensorflow/core/framework/tensor_shape.proto"
    "tensorflow/core/framework/tensor_slice.proto"
    "tensorflow/core/framework/types.proto"
    "tensorflow/core/framework/versions.proto"
    "tensorflow/core/lib/core/error_codes.proto"
    "tensorflow/core/protobuf/config.proto"
    "tensorflow/core/protobuf/tensor_bundle.proto"
    "tensorflow/core/protobuf/saver.proto"
    "tensorflow/core/util/memmapped_file_system.proto"
    "tensorflow/core/util/saved_tensor_slice.proto"
)
RELATIVE_PROTOBUF_TEXT_GENERATE_CPP(PROTO_TEXT_SRCS PROTO_TEXT_HDRS
    ${tensorflow_source_dir} ${tf_proto_text_srcs}
)

add_library(tf_protos_cc ${PROTO_SRCS} ${PROTO_HDRS})
target_include_directories(tf_protos_cc PUBLIC
     ${CMAKE_CURRENT_BINARY_DIR}
)
target_link_libraries(tf_protos_cc PUBLIC
    ${PROTOBUF_LIBRARIES}
)
# C++11
target_compile_features(tf_protos_cc PRIVATE
    cxx_rvalue_references
)

########################################################
# tf_core_lib library
########################################################
file(GLOB_RECURSE tf_core_lib_srcs
    "${tensorflow_source_dir}/tensorflow/core/lib/*.h"
    "${tensorflow_source_dir}/tensorflow/core/lib/*.cc"
    "${tensorflow_source_dir}/tensorflow/core/platform/*.h"
    "${tensorflow_source_dir}/tensorflow/core/platform/*.cc"
    "${tensorflow_source_dir}/tensorflow/core/public/*.h"
)

file(GLOB_RECURSE tf_core_lib_test_srcs
    "${tensorflow_source_dir}/tensorflow/core/lib/*test*.h"
    "${tensorflow_source_dir}/tensorflow/core/lib/*test*.cc"
    "${tensorflow_source_dir}/tensorflow/core/platform/*test*.h"
    "${tensorflow_source_dir}/tensorflow/core/platform/*test*.cc"
    "${tensorflow_source_dir}/tensorflow/core/public/*test*.h"
)

list(REMOVE_ITEM tf_core_lib_srcs ${tf_core_lib_test_srcs}) 

if(NOT tensorflow_ENABLE_SSL_SUPPORT)
  file(GLOB_RECURSE tf_core_lib_cloud_srcs
      "${tensorflow_source_dir}/tensorflow/core/platform/cloud/*.h"
      "${tensorflow_source_dir}/tensorflow/core/platform/cloud/*.cc"
  )
  list(REMOVE_ITEM tf_core_lib_srcs ${tf_core_lib_cloud_srcs})
endif()

add_library(tf_core_lib OBJECT ${tf_core_lib_srcs})
target_include_directories(tf_core_lib PUBLIC
    ${tensorflow_source_dir}
    ${gif_INCLUDE_DIR}
    ${jpeg_INCLUDE_DIR}
    ${png_INCLUDE_DIR}
    ${eigen_INCLUDE_DIRS}
    ${jsoncpp_INCLUDE_DIR}
)
target_compile_options(tf_core_lib PRIVATE
    -fno-exceptions
    -DEIGEN_AVOID_STL_ARRAY
)

# C++11
target_compile_features(tf_core_lib PRIVATE
    cxx_rvalue_references
)

add_dependencies(tf_core_lib
    gif_copy_headers_to_destination
    jpeg_copy_headers_to_destination
    png_copy_headers_to_destination
    eigen
    tf_protos_cc
    jsoncpp
    )

if(tensorflow_ENABLE_SSL_SUPPORT)
  target_include_directories(tf_core_lib PUBLIC ${boringssl_INCLUDE_DIR})
  add_dependencies(tf_core_lib boringssl)
endif()


# Tricky setup to force always rebuilding
# force_rebuild always runs forcing ${VERSION_INFO_CC} target to run
# ${VERSION_INFO_CC} would cache, but it depends on a phony never produced
# target.
set(VERSION_INFO_CC ${tensorflow_source_dir}/tensorflow/core/util/version_info.cc)
add_custom_target(force_rebuild_target ALL DEPENDS ${VERSION_INFO_CC})
add_custom_command(OUTPUT __force_rebuild COMMAND cmake -E echo)
add_custom_command(OUTPUT
    ${VERSION_INFO_CC}
    COMMAND ${tensorflow_source_dir}/tensorflow/tools/git/gen_git_source.py
    --raw_generate ${VERSION_INFO_CC}
    DEPENDS __force_rebuild)

set(tf_version_srcs ${tensorflow_source_dir}/tensorflow/core/util/version_info.cc)


########################################################
# tf_core_framework library
########################################################
file(GLOB_RECURSE tf_core_framework_srcs
    "${tensorflow_source_dir}/tensorflow/core/framework/*.h"
    "${tensorflow_source_dir}/tensorflow/core/framework/*.cc"
    "${tensorflow_source_dir}/tensorflow/core/util/*.h"
    "${tensorflow_source_dir}/tensorflow/core/util/*.cc"
    "${tensorflow_source_dir}/tensorflow/core/client/tensor_c_api.cc"
    "${tensorflow_source_dir}/tensorflow/core/common_runtime/session.cc"
    "${tensorflow_source_dir}/tensorflow/core/common_runtime/session_factory.cc"
    "${tensorflow_source_dir}/tensorflow/core/common_runtime/session_options.cc"
    "${tensorflow_source_dir}/public/*.h"
)

file(GLOB_RECURSE tf_core_framework_test_srcs
    "${tensorflow_source_dir}/tensorflow/core/framework/*test*.h"
    "${tensorflow_source_dir}/tensorflow/core/framework/*test*.cc"
    "${tensorflow_source_dir}/tensorflow/core/framework/*testutil.h"
    "${tensorflow_source_dir}/tensorflow/core/framework/*testutil.cc"
    "${tensorflow_source_dir}/tensorflow/core/framework/*main.cc"
    "${tensorflow_source_dir}/tensorflow/core/util/*test*.h"
    "${tensorflow_source_dir}/tensorflow/core/util/*test*.cc"
    "${tensorflow_source_dir}/tensorflow/core/util/*main.cc"
)

list(REMOVE_ITEM tf_core_framework_srcs ${tf_core_framework_test_srcs})

add_library(tf_core_framework OBJECT
    ${tf_core_framework_srcs}
    ${tf_version_srcs}
    ${PROTO_TEXT_HDRS}
    ${PROTO_TEXT_SRCS})
target_include_directories(tf_core_framework PUBLIC
    ${tensorflow_source_dir}
    ${eigen_INCLUDE_DIRS}
)
add_dependencies(tf_core_framework
    tf_core_lib
    proto_text
)
target_compile_options(tf_core_framework PRIVATE
    -fno-exceptions
    -DEIGEN_AVOID_STL_ARRAY
)
# C++11
target_compile_features(tf_core_framework PRIVATE
    cxx_rvalue_references
)
