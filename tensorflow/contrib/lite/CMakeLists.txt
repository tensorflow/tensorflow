cmake_minimum_required(VERSION 3.5)

# Project
project(tensorflow-lite C CXX)

# Set C++14 as standard for the whole project
set(CMAKE_CXX_STANDARD 14)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" )

set(TENSORFLOW_ROOT_DIR "C:/Development/tensorflow")
set(TENSORFLOW_LITE_ROOT_DIR "C:/Development/tensorflow/tensorflow/contrib/lite")
set(TENSORFLOW_LITE_KERNEL_DIR "C:/Development/tensorflow/tensorflow/contrib/lite/kernels")
set(TENSORFLOW_LITE_DOWNLOADED_DEPENDENCIES_DIR "C:/Development/tensorflow/tensorflow/contrib/lite/downloads")
set(TENSORFLOW_LITE_GEMMLOWP_DIR "C:/Development/tensorflow/tensorflow/contrib/lite/downloads/gemmlowp")
set(TENSORFLOW_LITE_EIGEN_DIR "C:/Development/tensorflow/tensorflow/contrib/lite/downloads/eigen")
set(TENSORFLOW_LITE_FLATBUFFERS_DIR "C:/Development/tensorflow/tensorflow/contrib/lite/downloads/flatbuffers/include")
set(TENSORFLOW_LITE_FARMHASH_DIR "C:/Development/tensorflow/tensorflow/contrib/lite/downloads/farmhash/src")
set(TENSORFLOW_LITE_GMOCK_DIR "C:/Development/tensorflow/tensorflow/contrib/lite/downloads/googletest/googlemock/include")
set(TENSORFLOW_LITE_GTEST_DIR "C:/Development/tensorflow/tensorflow/contrib/lite/downloads/googletest/googletest/include")

# TODO: SIMD Visual studio
add_definitions(-DGEMMLOWP_ALLOW_SLOW_SCALAR_FALLBACK
    -DNOMINMAX)


SET (TENSORFLOW_LITE_KERNEL_SOURCES
    ${TENSORFLOW_LITE_KERNEL_DIR}/activations.cc
    ${TENSORFLOW_LITE_KERNEL_DIR}/add.cc
    ${TENSORFLOW_LITE_KERNEL_DIR}/basic_rnn.cc
    ${TENSORFLOW_LITE_KERNEL_DIR}/batch_to_space_nd.cc
    ${TENSORFLOW_LITE_KERNEL_DIR}/concatenation.cc
    ${TENSORFLOW_LITE_KERNEL_DIR}/conv.cc
    ${TENSORFLOW_LITE_KERNEL_DIR}/depthwise_conv.cc
    ${TENSORFLOW_LITE_KERNEL_DIR}/embedding_lookup.cc
    ${TENSORFLOW_LITE_KERNEL_DIR}/embedding_lookup_sparse.cc
    ${TENSORFLOW_LITE_KERNEL_DIR}/fully_connected.cc
    ${TENSORFLOW_LITE_KERNEL_DIR}/gather.cc
    ${TENSORFLOW_LITE_KERNEL_DIR}/gemm_support.cc
    ${TENSORFLOW_LITE_KERNEL_DIR}/hashtable_lookup.cc
    ${TENSORFLOW_LITE_KERNEL_DIR}/internal/optimized/neon_tensor_utils.cc
    ${TENSORFLOW_LITE_KERNEL_DIR}/internal/quantization_util.cc
    ${TENSORFLOW_LITE_KERNEL_DIR}/internal/reference/portable_tensor_utils.cc
    ${TENSORFLOW_LITE_KERNEL_DIR}/internal/tensor_utils.cc
    ${TENSORFLOW_LITE_KERNEL_DIR}/kernel_util.cc
    ${TENSORFLOW_LITE_KERNEL_DIR}/l2norm.cc
    ${TENSORFLOW_LITE_KERNEL_DIR}/local_response_norm.cc
    ${TENSORFLOW_LITE_KERNEL_DIR}/lsh_projection.cc
    ${TENSORFLOW_LITE_KERNEL_DIR}/lstm.cc
    ${TENSORFLOW_LITE_KERNEL_DIR}/mean.cc
    ${TENSORFLOW_LITE_KERNEL_DIR}/mul.cc
    ${TENSORFLOW_LITE_KERNEL_DIR}/pad.cc
    ${TENSORFLOW_LITE_KERNEL_DIR}/pooling.cc
    ${TENSORFLOW_LITE_KERNEL_DIR}/register.cc
    ${TENSORFLOW_LITE_KERNEL_DIR}/reshape.cc
    ${TENSORFLOW_LITE_KERNEL_DIR}/resize_bilinear.cc
    ${TENSORFLOW_LITE_KERNEL_DIR}/skip_gram.cc
    ${TENSORFLOW_LITE_KERNEL_DIR}/space_to_batch_nd.cc
    ${TENSORFLOW_LITE_KERNEL_DIR}/space_to_depth.cc
    ${TENSORFLOW_LITE_KERNEL_DIR}/svdf.cc
    # ${TENSORFLOW_LITE_KERNEL_DIR}/test_util.cc
    ${TENSORFLOW_LITE_KERNEL_DIR}/transpose.cc
    ${TENSORFLOW_LITE_KERNEL_DIR}/unidirectional_sequence_rnn.cc
    )

add_library(tensorflow-lite STATIC
    simple_memory_arena.cc
    context.c
    error_reporter.cc
    interpreter.cc
    string_util.cc
    allocation.cc
    model.cc
    ${TENSORFLOW_LITE_KERNEL_SOURCES}
    )


include_directories(${TENSORFLOW_ROOT_DIR}
    ${TENSORFLOW_LITE_GEMMLOWP_DIR}
    ${TENSORFLOW_LITE_FLATBUFFERS_DIR}
    ${TENSORFLOW_LITE_EIGEN_DIR}
    ${TENSORFLOW_LITE_FARMHASH_DIR}
    ${TENSORFLOW_LITE_GMOCK_DIR}
    ${TENSORFLOW_LITE_GTEST_DIR}
    )
