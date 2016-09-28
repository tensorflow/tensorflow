########################################################
# tf_core_cpu library
########################################################
file(GLOB_RECURSE tf_core_cpu_srcs
    "${tensorflow_source_dir}/tensorflow/core/common_runtime/*.h"
    "${tensorflow_source_dir}/tensorflow/core/common_runtime/*.cc"
    "${tensorflow_source_dir}/tensorflow/core/graph/*.h"
    "${tensorflow_source_dir}/tensorflow/core/graph/*.cc"
    "${tensorflow_source_dir}/tensorflow/core/public/*.h"
)

file(GLOB_RECURSE tf_core_cpu_exclude_srcs
    "${tensorflow_source_dir}/tensorflow/core/*test*.h"
    "${tensorflow_source_dir}/tensorflow/core/*test*.cc"
    "${tensorflow_source_dir}/tensorflow/core/*main.cc"
    "${tensorflow_source_dir}/tensorflow/core/common_runtime/gpu/*.cc"
    "${tensorflow_source_dir}/tensorflow/core/common_runtime/gpu_device_factory.cc"
    "${tensorflow_source_dir}/tensorflow/core/common_runtime/direct_session.cc"
    "${tensorflow_source_dir}/tensorflow/core/common_runtime/direct_session.h"
    "${tensorflow_source_dir}/tensorflow/core/common_runtime/session.cc"
    "${tensorflow_source_dir}/tensorflow/core/common_runtime/session_factory.cc"
    "${tensorflow_source_dir}/tensorflow/core/common_runtime/session_options.cc"
)

list(REMOVE_ITEM tf_core_cpu_srcs ${tf_core_cpu_exclude_srcs}) 
# We need to include stubs for the GPU tracer, which are in the exclude glob.
list(APPEND tf_core_cpu_srcs
     "${tensorflow_source_dir}/tensorflow/core/common_runtime/gpu/gpu_tracer.cc"
     "${tensorflow_source_dir}/tensorflow/core/common_runtime/gpu/gpu_tracer.h"
)

add_library(tf_core_cpu OBJECT ${tf_core_cpu_srcs})

target_include_directories(tf_core_cpu PRIVATE
    ${tensorflow_source_dir}
    ${eigen_INCLUDE_DIRS}
)

add_dependencies(tf_core_cpu
    tf_core_framework
)
#target_link_libraries(tf_core_cpu
#    ${CMAKE_THREAD_LIBS_INIT}
#    ${PROTOBUF_LIBRARIES}
#    tf_core_framework
#    tf_core_lib
#    tf_protos_cc
#)

target_compile_options(tf_core_cpu PRIVATE
    -fno-exceptions
    -DEIGEN_AVOID_STL_ARRAY
)

# C++11
target_compile_features(tf_core_cpu PRIVATE
    cxx_rvalue_references
)

