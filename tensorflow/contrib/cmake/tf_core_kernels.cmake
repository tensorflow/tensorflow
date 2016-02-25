########################################################
# tf_core_kernels library
########################################################
file(GLOB_RECURSE tf_core_kernels_srcs
   "${tensorflow_source_dir}/tensorflow/core/kernels/*.h"
   "${tensorflow_source_dir}/tensorflow/core/kernels/*.cc"
)

file(GLOB_RECURSE tf_core_kernels_exclude_srcs
   "${tensorflow_source_dir}/tensorflow/core/kernels/*test*.h"
   "${tensorflow_source_dir}/tensorflow/core/kernels/*test*.cc"
   "${tensorflow_source_dir}/tensorflow/core/kernels/*testutil.h"
   "${tensorflow_source_dir}/tensorflow/core/kernels/*testutil.cc"
   "${tensorflow_source_dir}/tensorflow/core/kernels/*main.cc"
   "${tensorflow_source_dir}/tensorflow/core/kernels/*.cu.cc"
)

list(REMOVE_ITEM tf_core_kernels_srcs ${tf_core_kernels_exclude_srcs}) 

add_library(tf_core_kernels OBJECT ${tf_core_kernels_srcs})

add_dependencies(tf_core_kernels tf_core_cpu)

target_include_directories(tf_core_kernels PRIVATE
   ${tensorflow_source_dir}
   ${png_INCLUDE_DIR}
   ${eigen_INCLUDE_DIRS}
)

#target_link_libraries(tf_core_kernels
#   ${CMAKE_THREAD_LIBS_INIT}
#   ${PROTOBUF_LIBRARIES}
#   tf_core_cpu
#   tf_core_framework
#   tf_core_lib
#   tf_protos_cc
#   tf_models_word2vec_kernels
#   tf_stream_executor
#   tf_core_ops
#   tf_core_cpu
#)

#        "@gemmlowp//:eight_bit_int_gemm",

target_compile_options(tf_core_kernels PRIVATE
   -fno-exceptions
   -DEIGEN_AVOID_STL_ARRAY
)

# C++11
target_compile_features(tf_core_kernels PRIVATE
   cxx_rvalue_references
)
