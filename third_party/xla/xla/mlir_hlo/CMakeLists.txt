#
# Copyright 2020 The OpenXLA Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
cmake_minimum_required(VERSION 3.15.0)

if(POLICY CMP0068)
  cmake_policy(SET CMP0068 NEW)
  set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
endif()

if(POLICY CMP0075)
  cmake_policy(SET CMP0075 NEW)
endif()

if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

# CMP0116: Ninja generators transform `DEPFILE`s from `add_custom_command()`
# New in CMake 3.20. https://cmake.org/cmake/help/latest/policy/CMP0116.html
if(POLICY CMP0116)
  cmake_policy(SET CMP0116 OLD)
endif()

#-------------------------------------------------------------------------------
# Options and settings
#-------------------------------------------------------------------------------
option(MHLO_BUILD_EMBEDDED "Build MHLO as part of another project" OFF)
option(MHLO_ENABLE_BINDINGS_PYTHON "Enables MHLO python bindings" OFF)

#-------------------------------------------------------------------------------
# Project setup and globals
#-------------------------------------------------------------------------------
set(MHLO_EXTERNAL_PROJECT_BUILD OFF)

if(NOT (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR) AND NOT MLIR_BINARY_DIR)
  # Building as part of LLVM via the external project mechanism.
  set(MHLO_EXTERNAL_PROJECT_BUILD ON)
else()
  # Building standalone.
  project(mlir-hlo LANGUAGES CXX C)
  set(CMAKE_C_STANDARD 11)
  set(CMAKE_CXX_STANDARD 17)
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")
endif()

if(MHLO_ENABLE_BINDINGS_PYTHON AND NOT MHLO_EXTERNAL_PROJECT_BUILD)
  message(WARNING "MHLO python bindings are only supported in unified, external project builds")
endif()

# Build with ccache if the package is present
set(LLVM_CCACHE_BUILD OFF CACHE BOOL "Set to ON for a ccache enabled build")
if(LLVM_CCACHE_BUILD)
  find_program(CCACHE_PROGRAM ccache)
  if(CCACHE_PROGRAM)
      set(LLVM_CCACHE_MAXSIZE "" CACHE STRING "Size of ccache")
      set(LLVM_CCACHE_DIR "" CACHE STRING "Directory to keep ccached data")
      set(LLVM_CCACHE_PARAMS "CCACHE_CPP2=yes CCACHE_HASHDIR=yes"
          CACHE STRING "Parameters to pass through to ccache")

      set(CCACHE_PROGRAM "${LLVM_CCACHE_PARAMS} ${CCACHE_PROGRAM}")
      if (LLVM_CCACHE_MAXSIZE)
        set(CCACHE_PROGRAM "CCACHE_MAXSIZE=${LLVM_CCACHE_MAXSIZE} ${CCACHE_PROGRAM}")
      endif()
      if (LLVM_CCACHE_DIR)
        set(CCACHE_PROGRAM "CCACHE_DIR=${LLVM_CCACHE_DIR} ${CCACHE_PROGRAM}")
      endif()
      set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ${CCACHE_PROGRAM})
  else()
    message(FATAL_ERROR "Unable to find the program ccache. Set LLVM_CCACHE_BUILD to OFF")
  endif()
endif()

#-------------------------------------------------------------------------------
# MLIR/LLVM Configuration
#-------------------------------------------------------------------------------

# Find MLIR to install if we are building standalone. If building as part of
# another project, let it handle the MLIR dependency. The dependent project
# might use a bundled version of MLIR instead of installing, for instance.
if(MHLO_EXTERNAL_PROJECT_BUILD)
  message(STATUS "Building MHLO as an external LLVM project")
  set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir ) # --src-root
  set(MLIR_INCLUDE_DIR ${MLIR_MAIN_SRC_DIR}/include ) # --includedir
  set(MLIR_GENERATED_INCLUDE_DIR ${LLVM_BINARY_DIR}/tools/mlir/include)
  include_directories(SYSTEM ${MLIR_INCLUDE_DIR})
  include_directories(SYSTEM ${MLIR_GENERATED_INCLUDE_DIR})
  include_directories(SYSTEM ${MLIR_TABLEGEN_OUTPUT_DIR})

  set(BACKEND_PACKAGE_STRING "${PACKAGE_STRING}")
  list(APPEND CMAKE_MODULE_PATH "${MLIR_MAIN_SRC_DIR}/cmake/modules")
elseif(NOT MHLO_BUILD_EMBEDDED)
  message(STATUS "Building MHLO with an installed MLIR")
  find_package(MLIR REQUIRED CONFIG)
  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
else()
  message(STATUS "Building MHLO embedded in another project")
endif()

if(LLVM_ENABLE_ZLIB)
  find_package(ZLIB)
endif()

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

#-------------------------------------------------------------------------------
# StableHLO configuration
#-------------------------------------------------------------------------------

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/stablehlo)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/stablehlo)
set(LLVM_TABLEGEN_FLAGS -I ${MLIR_HLO_SOURCE_DIR}/stablehlo)

#-------------------------------------------------------------------------------
# Python configuration
#-------------------------------------------------------------------------------

if(MHLO_ENABLE_BINDINGS_PYTHON)
  include(MLIRDetectPythonEnv)
  mlir_configure_python_dev_packages()
endif()

#-------------------------------------------------------------------------------
# Directory setup
#-------------------------------------------------------------------------------

set(MLIR_HLO_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(MLIR_HLO_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(MLIR_HLO_MAIN_INCLUDE_DIR ${MLIR_HLO_SOURCE_DIR})
set(MLIR_HLO_GEN_INCLUDE_DIR ${MLIR_HLO_BINARY_DIR})
set(MLIR_HLO_TOOLS_DIR ${MLIR_HLO_BINARY_DIR}/bin)
set(MLIR_HLO_LIB_DIR ${MLIR_HLO_BINARY_DIR}/lib)
add_custom_target(check-mlir-hlo)

add_subdirectory(bindings)
add_subdirectory(deallocation)
add_subdirectory(lhlo)
add_subdirectory(mhlo)
add_subdirectory(stablehlo)
add_subdirectory(tests)
add_subdirectory(tools)
add_subdirectory(transforms)
add_subdirectory(utils)


add_subdirectory(cmake/modules)
