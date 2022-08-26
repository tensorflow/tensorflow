# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
# Copyright 2022 The StableHLO Authors.
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
option(STABLEHLO_ENABLE_BINDINGS_PYTHON "Enables StableHLO Python bindings" OFF)

#-------------------------------------------------------------------------------
# Project setup and globals
#-------------------------------------------------------------------------------
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(STABLEHLO_STANDALONE_BUILD ON)
  project(stablehlo LANGUAGES CXX C)
  set(CMAKE_C_STANDARD 11)
  set(CMAKE_CXX_STANDARD 17)
endif()

#-------------------------------------------------------------------------------
# MLIR/LLVM Configuration
#-------------------------------------------------------------------------------

if(STABLEHLO_STANDALONE_BUILD)
  message(STATUS "Building StableHLO standalone")
  find_package(MLIR REQUIRED CONFIG)
  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

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
else()
  message(STATUS "Building StableHLO as part of another project")
  set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir ) # --src-root
  set(MLIR_INCLUDE_DIR ${MLIR_MAIN_SRC_DIR}/include ) # --includedir
  set(MLIR_GENERATED_INCLUDE_DIR ${LLVM_BINARY_DIR}/tools/mlir/include)

  include_directories(SYSTEM ${MLIR_INCLUDE_DIR})
  include_directories(SYSTEM ${MLIR_GENERATED_INCLUDE_DIR})
  include_directories(SYSTEM ${MLIR_TABLEGEN_OUTPUT_DIR})
  include_directories(${CMAKE_CURRENT_SOURCE_DIR})
  include_directories(${CMAKE_CURRENT_BINARY_DIR})

  set(BACKEND_PACKAGE_STRING "${PACKAGE_STRING}")
  list(APPEND CMAKE_MODULE_PATH "${MLIR_MAIN_SRC_DIR}/cmake/modules")
endif()

#-------------------------------------------------------------------------------
# Python configuration
#-------------------------------------------------------------------------------

if(STABLEHLO_ENABLE_BINDINGS_PYTHON)
  if(STABLEHLO_STANDALONE_BUILD)
    message(WARNING "StableHLO Python bindings are not supported in standalone mode. See build_tools/README.md for instructions.")
  endif()

  include(MLIRDetectPythonEnv)
  mlir_configure_python_dev_packages()
endif()

#-------------------------------------------------------------------------------
# Directory setup
#-------------------------------------------------------------------------------

set(STABLEHLO_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(STABLEHLO_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

add_custom_target(check-stablehlo)

add_subdirectory(dialect)
add_subdirectory(integrations)
add_subdirectory(tests)
add_subdirectory(tools)
