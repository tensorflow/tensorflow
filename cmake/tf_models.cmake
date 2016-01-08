#cc_library(
#    name = "word2vec_ops",
#    srcs = [
#        "word2vec_ops.cc",
#    ],
#    visibility = ["//tensorflow:internal"],
#    deps = [
#        "//tensorflow/core:framework",
#    ],
#    alwayslink = 1,
#)

########################################################
# tf_models_word2vec_ops library
########################################################
file(GLOB tf_models_word2vec_ops_srcs
    "${tensorflow_source_dir}/tensorflow/models/embedding/word2vec_ops.cc"
)

add_library(tf_models_word2vec_ops OBJECT ${tf_models_word2vec_ops_srcs})

target_include_directories(tf_models_word2vec_ops PRIVATE
    ${tensorflow_source_dir}
    ${eigen_INCLUDE_DIRS}
)

add_dependencies(tf_models_word2vec_ops
    tf_core_framework
)
#target_link_libraries(tf_models_word2vec_ops
#    ${CMAKE_THREAD_LIBS_INIT}
#    ${PROTOBUF_LIBRARIES}
#    tf_core_framework
#    tf_core_lib
#    tf_protos_cc
#)

target_compile_options(tf_models_word2vec_ops PRIVATE
    -fno-exceptions
    -DEIGEN_AVOID_STL_ARRAY
)

# C++11
target_compile_features(tf_models_word2vec_ops PRIVATE
    cxx_rvalue_references
)

#cc_library(
#    name = "word2vec_kernels",
#    srcs = [
#        "word2vec_kernels.cc",
#    ],
#    visibility = ["//tensorflow:internal"],
#    deps = [
#        "//tensorflow/core",
#    ],
#    alwayslink = 1,
#)
########################################################
# tf_models_word2vec_kernels library
########################################################
file(GLOB tf_models_word2vec_kernels_srcs
    "${tensorflow_source_dir}/tensorflow/models/embedding/word2vec_kernels.cc"
)

add_library(tf_models_word2vec_kernels OBJECT ${tf_models_word2vec_kernels_srcs})

target_include_directories(tf_models_word2vec_kernels PRIVATE
    ${tensorflow_source_dir}
    ${eigen_INCLUDE_DIRS}
    ${re2_INCLUDES}
)

add_dependencies(tf_models_word2vec_ops
    tf_core_cpu
)

#target_link_libraries(tf_models_word2vec_kernels
#    ${CMAKE_THREAD_LIBS_INIT}
#    ${PROTOBUF_LIBRARIES}
#    tf_core_framework
#    tf_core_lib
#    tf_protos_cc
#    tf_core_cpu
#)

target_compile_options(tf_models_word2vec_kernels PRIVATE
    -fno-exceptions
    -DEIGEN_AVOID_STL_ARRAY
)

# C++11
target_compile_features(tf_models_word2vec_kernels PRIVATE
    cxx_rvalue_references
)
