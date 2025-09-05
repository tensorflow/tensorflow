#
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

if(TARGET eigen OR eigen_POPULATED)
  return()
endif()

include(OverridableFetchContent)

OverridableFetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  # Sync with tensorflow/third_party/eigen3/workspace.bzl
  GIT_TAG 70d8d99d0df9fd967b135efd8d12ed20fc48d007
  # It's not currently (cmake 3.17) possible to shallow clone with a GIT TAG
  # as cmake attempts to git checkout the commit hash after the clone
  # which doesn't work as it's a shallow clone hence a different commit hash.
  # https://gitlab.kitware.com/cmake/cmake/-/issues/17770
  # GIT_SHALLOW TRUE
  GIT_PROGRESS TRUE
  PREFIX "${CMAKE_BINARY_DIR}"
  SOURCE_DIR "${CMAKE_BINARY_DIR}/eigen"
  LICENSE_FILE "COPYING.MPL2"
)
OverridableFetchContent_GetProperties(eigen)
if(NOT eigen_POPULATED)
  OverridableFetchContent_Populate(eigen)
endif()

# Patch Eigen to disable Fortran compiler check for BLAS and LAPACK tests.
if(NOT EIGEN_DISABLED_FORTRAN_COMPILER_CHECK)
  file(WRITE "${eigen_SOURCE_DIR}/cmake/language_support.cmake" "
      function(workaround_9220 language language_works)
        set(\${language_works} OFF PARENT_SCOPE)
      endfunction()"
  )
endif()
# Patch Eigen to disable benchmark suite.
if(NOT EIGEN_BUILD_BTL)
  file(WRITE "${eigen_SOURCE_DIR}/bench/spbench/CMakeLists.txt" "")
endif()

# Patch Eigen to disable doc generation, as it builds C++ standalone apps with
# the host toolchain which breaks cross compiled builds.
if(NOT EIGEN_GENERATE_DOCS)
  file(WRITE "${eigen_SOURCE_DIR}/doc/CMakeLists.txt" "")
  file(WRITE "${eigen_SOURCE_DIR}/unsupported/doc/CMakeLists.txt" "")
endif()

set(EIGEN_DISABLED_FORTRAN_COMPILER_CHECK ON CACHE BOOL "Disabled Fortran")

set(EIGEN_LEAVE_TEST_IN_ALL_TARGET OFF CACHE BOOL
  "Remove tests from all target."
)
set(BUILD_TESTING OFF CACHE BOOL "Disable tests.")
set(EIGEN_TEST_CXX11 OFF CACHE BOOL "Disable tests of C++11 features.")
set(EIGEN_BUILD_BTL OFF CACHE BOOL "Disable benchmark suite.")
set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "Disable pkg-config.")
set(EIGEN_SPLIT_LARGE_TESTS OFF CACHE BOOL "Disable test splitting.")
set(EIGEN_DEFAULT_TO_ROW_MAJOR OFF CACHE BOOL
  "Disable row-major matrix storage"
)
set(EIGEN_TEST_NOQT ON CACHE BOOL "Disable Qt support in tests.")
set(EIGEN_TEST_SSE2 OFF CACHE BOOL "Disable SSE2 test.")
set(EIGEN_TEST_SSE3 OFF CACHE BOOL "Disable SSE3 test.")
set(EIGEN_TEST_SSSE3 OFF CACHE BOOL "Disable SSSE3 test.")
set(EIGEN_TEST_SSE4_1 OFF CACHE BOOL "Disable SSE4.1 test.")
set(EIGEN_TEST_SSE4_2 OFF CACHE BOOL "Disable SSE4.2 test.")
set(EIGEN_TEST_AVX OFF CACHE BOOL "Disable AVX test.")
set(EIGEN_TEST_FMA OFF CACHE BOOL "Disable FMA test.")
set(EIGEN_TEST_AVX512 OFF CACHE BOOL "Disable AVX512 test.")
set(EIGEN_TEST_F16C OFF CACHE BOOL "Disable F16C test.")
set(EIGEN_TEST_ALTIVEC OFF CACHE BOOL "Disable AltiVec test.")
set(EIGEN_TEST_VSX OFF CACHE BOOL "Disable VSX test.")
set(EIGEN_TEST_MSA OFF CACHE BOOL "Disable MSA test.")
set(EIGEN_TEST_NEON OFF CACHE BOOL "Disable NEON test.")
set(EIGEN_TEST_NEON64 OFF CACHE BOOL "Disable NEON64 test.")
set(EIGEN_TEST_Z13 OFF CACHE BOOL "Disable Z13 test.")
set(EIGEN_TEST_Z14 OFF CACHE BOOL "Disable Z14 test.")
set(EIGEN_TEST_OPENMP OFF CACHE BOOL "Disable OpenMP test.")
set(EIGEN_TEST_NO_EXPLICIT_VECTORIZATION OFF CACHE BOOL "Disable vectorization")
set(EIGEN_TEST_X87 OFF CACHE BOOL "Disable X87 instructions test")
set(EIGEN_TEST_32BIT OFF CACHE BOOL "Disable 32-bit instructions test")
set(EIGEN_TEST_NO_EXPLICIT_ALIGNMENT OFF CACHE BOOL "Disable alignment test")
set(EIGEN_TEST_NO_EXCEPTIONS OFF CACHE BOOL "Disable exceptions test")
set(EIGEN_TEST_SYCL OFF CACHE BOOL "Disable Sycl test")
set(EIGEN_SYCL_TRISYCL OFF CACHE BOOL "Disable triSYCL test")
# Make sure only MPL2.0 or more permissively licensed code is included.
add_compile_definitions(EIGEN_MPL2_ONLY)
add_subdirectory("${eigen_SOURCE_DIR}" "${eigen_BINARY_DIR}")
