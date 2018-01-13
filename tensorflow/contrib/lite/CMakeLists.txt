cmake_minimum_required(VERSION 3.5)

# Project
project(tensorflow-lite C CXX)

# Set C++14 as standard for the whole project
set(CMAKE_CXX_STANDARD 14)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4" )

set(TENSORFLOW_ROOT_DIR "C:/Development/tensorflow")
set(TENSORFLOW_LITE_ROOT_DIR "C:/Development/tensorflow/tensorflow/contrib/lite")
set(TENSORFLOW_LITE_DOWNLOADED_DEPENDCIES_DIR "C:/Development/tensorflow/tensorflow/contrib/lite/downloads")
set(TENSORFLOW_LITE_GEMMLOWP_DIR "C:/Development/tensorflow/tensorflow/contrib/lite/downloads/gemmlowp")

# TODO: SIMD Visual studio
add_definitions(-DGEMMLOWP_ALLOW_SLOW_SCALAR_FALLBACK
    -DNONMINMAX)


add_library(tensorflow-lite STATIC
    simple_memory_arena.cc
    context.c
    error_reporter.cc
    interpreter.cc
    )


include_directories(${TENSORFLOW_ROOT_DIR}
    ${TENSORFLOW_LITE_GEMMLOWP_DIR}
    )
