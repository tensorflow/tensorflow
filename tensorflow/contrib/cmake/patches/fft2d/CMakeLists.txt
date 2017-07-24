cmake_minimum_required(VERSION 2.8.3)

project(fft2d)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(FFT2D_SRCS
    "fftsg.c"
)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

add_library(fft2d ${FFT2D_SRCS})

install(TARGETS fft2d
  LIBRARY DESTINATION lib COMPONENT RuntimeLibraries
  ARCHIVE DESTINATION lib COMPONENT Development)
