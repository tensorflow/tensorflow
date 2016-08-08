########################################################
# tf_core_direct_session library
########################################################
file(GLOB tf_core_direct_session_srcs
   "${tensorflow_source_dir}/tensorflow/core/common_runtime/direct_session.cc"
   "${tensorflow_source_dir}/tensorflow/core/common_runtime/direct_session.h"
   "${tensorflow_source_dir}/tensorflow/core/debug/*.h"
   "${tensorflow_source_dir}/tensorflow/core/debug/*.cc"
)

file(GLOB_RECURSE tf_core_direct_session_test_srcs
    "${tensorflow_source_dir}/tensorflow/core/debug/*test*.h"
    "${tensorflow_source_dir}/tensorflow/core/debug/*test*.cc"
)

list(REMOVE_ITEM tf_core_direct_session_srcs ${tf_core_direct_session_test_srcs})

add_library(tf_core_direct_session OBJECT ${tf_core_direct_session_srcs})

add_dependencies(tf_core_direct_session tf_core_cpu)

target_include_directories(tf_core_direct_session PRIVATE
   ${tensorflow_source_dir}
   ${eigen_INCLUDE_DIRS}
)

#target_link_libraries(tf_core_direct_session
#   ${CMAKE_THREAD_LIBS_INIT}
#   ${PROTOBUF_LIBRARIES}
#   tf_core_cpu
#   tf_core_framework
#   tf_core_lib
#   tf_protos_cc
#)

target_compile_options(tf_core_direct_session PRIVATE
   -fno-exceptions
   -DEIGEN_AVOID_STL_ARRAY
)

# C++11
target_compile_features(tf_core_direct_session PRIVATE
   cxx_rvalue_references
)
