cmake_minimum_required(VERSION 3.13.1)
include($ENV{ZEPHYR_BASE}/cmake/app/boilerplate.cmake NO_POLICY_SCOPE)
project(tf_lite_magic_wand)

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} %{CXX_FLAGS}%")
set(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} %{CC_FLAGS}%")
set(CMAKE_EXE_LINKER_FLAGS "%{LINKER_FLAGS}%")

# -fno-threadsafe-statics -- disables the mutex around initialization of local static variables
target_compile_options(app PRIVATE "-fno-threadsafe-statics")

target_sources(app PRIVATE
		%{SRCS}%
		)

target_include_directories(app PRIVATE
		%{INCLUDE_DIRS}%
                )
