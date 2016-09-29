########################################################
# tf_core_distributed_runtime library
########################################################
file(GLOB_RECURSE tf_core_distributed_runtime_srcs
   "${tensorflow_source_dir}/tensorflow/core/distributed_runtime/*.h"
   "${tensorflow_source_dir}/tensorflow/core/distributed_runtime/*.cc"
)

file(GLOB_RECURSE tf_core_distributed_runtime_exclude_srcs
    "${tensorflow_source_dir}/tensorflow/core/distributed_runtime/*test*.h"
    "${tensorflow_source_dir}/tensorflow/core/distributed_runtime/*test*.cc"
    "${tensorflow_source_dir}/tensorflow/core/distributed_runtime/rpc/grpc_tensorflow_server.cc"
)

list(REMOVE_ITEM tf_core_distributed_runtime_srcs ${tf_core_distributed_runtime_exclude_srcs})

add_library(tf_core_distributed_runtime OBJECT ${tf_core_distributed_runtime_srcs})

add_dependencies(tf_core_distributed_runtime
    tf_core_cpu grpc
)

target_include_directories(tf_core_distributed_runtime PRIVATE
   ${tensorflow_source_dir}
   ${eigen_INCLUDE_DIRS}
   ${GRPC_INCLUDE_DIRS}
)

target_compile_options(tf_core_distributed_runtime PRIVATE
   -fno-exceptions
   -DEIGEN_AVOID_STL_ARRAY
)

# C++11
target_compile_features(tf_core_distributed_runtime PRIVATE
   cxx_rvalue_references
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
)

add_dependencies(tf_core_distributed_runtime
    grpc
)

target_include_directories(grpc_tensorflow_server PUBLIC
    ${tensorflow_source_dir}
    ${eigen_INCLUDE_DIRS}
    ${GRPC_INCLUDE_DIRS}
)

find_package(ZLIB REQUIRED)

target_link_libraries(grpc_tensorflow_server PUBLIC
    ${CMAKE_THREAD_LIBS_INIT}
    ${PROTOBUF_LIBRARIES}
    ${GRPC_LIBRARIES}
    tf_protos_cc
    ${farmhash_STATIC_LIBRARIES}
    ${gif_STATIC_LIBRARIES}
    ${jpeg_STATIC_LIBRARIES}
    ${jsoncpp_STATIC_LIBRARIES}
    ${png_STATIC_LIBRARIES}
    ${ZLIB_LIBRARIES}
    ${CMAKE_DL_LIBS}
)
if(tensorflow_ENABLE_SSL_SUPPORT)
  target_link_libraries(grpc_tensorflow_server PUBLIC
      ${boringssl_STATIC_LIBRARIES})
endif()

target_compile_options(grpc_tensorflow_server PRIVATE
    -fno-exceptions
    -DEIGEN_AVOID_STL_ARRAY
)

# C++11
target_compile_features(grpc_tensorflow_server PRIVATE
    cxx_rvalue_references
)
