#
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

cmake_minimum_required(VERSION 3.16)
project(tensorflow-lite-c C CXX)

option(TFLITE_C_BUILD_SHARED_LIBS "Build shared libraries" ON)

set(TF_SOURCE_DIR "" CACHE PATH
  "Directory that contains the TensorFlow project"
)
if (NOT TF_SOURCE_DIR)
  get_filename_component(TF_SOURCE_DIR
    "${CMAKE_CURRENT_LIST_DIR}/../../../"
    ABSOLUTE
  )
endif()

set(TFLITE_SOURCE_DIR "${TF_SOURCE_DIR}/tensorflow/lite")

add_subdirectory(
  "${TFLITE_SOURCE_DIR}"
  "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite"
  EXCLUDE_FROM_ALL
)

set(CMAKE_CXX_STANDARD 20)

if(CMAKE_SYSTEM_NAME MATCHES "Windows"
    AND (MSVC AND (CMAKE_SIZEOF_VOID_P EQUAL 4)))
  message("Disabling MSVC /O2 optimization for Win32")
  set(CompFlags
    CMAKE_CXX_FLAGS_RELEASE
    CMAKE_CXX_FLAGS_MINSIZEREL
    CMAKE_CXX_FLAGS_RELWITHDEBINFO
    CMAKE_C_FLAGS_RELEASE
    CMAKE_C_FLAGS_MINSIZEREL
    CMAKE_C_FLAGS_RELWITHDEBINFO
  )
  foreach (CompFlag ${CompFlags})
    string(REGEX REPLACE "(\/Ob. )" "" ${CompFlag} "${${CompFlag}}")
    string(REPLACE "/O2" "/O1" ${CompFlag} "${${CompFlag}}")
    list(REMOVE_DUPLICATES ${CompFlag})
    set(${CompFlag} "${${CompFlag}}" CACHE INTERNAL "")
  endforeach()
endif()

set(TFLITE_C_LIBTYPE STATIC)
if (TFLITE_C_BUILD_SHARED_LIBS)
  set(TFLITE_C_LIBTYPE SHARED)
endif()

add_library(tensorflowlite_c ${TFLITE_C_LIBTYPE}
  ${TFLITE_SOURCE_DIR}/core/c/c_api.cc
  ${TFLITE_SOURCE_DIR}/core/c/c_api_experimental.cc
  ${TFLITE_SOURCE_DIR}/core/c/common.cc
  ${TFLITE_SOURCE_DIR}/core/c/operator.cc
  ${TF_SOURCE_DIR}/tensorflow/compiler/mlir/lite/core/c/tflite_types.h
  builtin_op_data.h
  c_api.h
  c_api_experimental.h
  c_api_internal.h
  c_api_types.h
  common.h
)

if (TFLITE_C_BUILD_SHARED_LIBS)
  if (WIN32)
    target_compile_definitions(tensorflowlite_c PRIVATE TFL_COMPILE_LIBRARY)
    target_compile_definitions(tensorflow-lite PRIVATE TFL_COMPILE_LIBRARY)
  elseif (APPLE)
    target_link_options(tensorflowlite_c PRIVATE "-Wl,-exported_symbols_list,${TFLITE_SOURCE_DIR}/c/exported_symbols.lds")
  else ()
    target_link_options(tensorflowlite_c PRIVATE "-Wl,--version-script,${TFLITE_SOURCE_DIR}/c/version_script.lds")
  endif()
endif()

target_link_libraries(tensorflowlite_c
  tensorflow-lite
)

if(TFLITE_ENABLE_INSTALL)
  install(
    TARGETS tensorflowlite_c
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
  set(TFLITE_C_PUBLIC_HDRS
    builtin_op_data.h
    c_api.h
    c_api_experimental.h
    c_api_opaque.h
    c_api_types.h
    common.h
    ${TFLITE_SOURCE_DIR}/core/c/builtin_op_data.h
    ${TFLITE_SOURCE_DIR}/core/c/c_api.h
    ${TFLITE_SOURCE_DIR}/core/c/c_api_experimental.h
    ${TFLITE_SOURCE_DIR}/core/c/c_api_opaque.h
    ${TFLITE_SOURCE_DIR}/core/c/c_api_types.h
    ${TFLITE_SOURCE_DIR}/core/c/common.h
    ${TFLITE_SOURCE_DIR}/core/c/operator.h
    ${TFLITE_SOURCE_DIR}/core/async/c/types.h
    ${TFLITE_SOURCE_DIR}/builtin_ops.h
    ${TF_SOURCE_DIR}/tensorflow/compiler/mlir/lite/core/c/tflite_types.h
    ${TF_SOURCE_DIR}/tensorflow/compiler/mlir/lite/core/c/builtin_op_data.h
  )
  foreach(hdr ${TFLITE_C_PUBLIC_HDRS})
    get_filename_component(hdr_abs ${hdr} ABSOLUTE)
    get_filename_component(hdr_dir ${hdr_abs} DIRECTORY)
    file(RELATIVE_PATH rel_dir "${TF_SOURCE_DIR}/tensorflow" "${hdr_dir}")
    install(
      FILES ${hdr_abs}
      DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/tensorflow/${rel_dir}"
    )
  endforeach()

  if(DEFINED TF_MAJOR_VERSION AND DEFINED TF_MINOR_VERSION AND DEFINED TF_PATCH_VERSION)
    set(TFLITE_C_PC_VERSION "${TF_MAJOR_VERSION}.${TF_MINOR_VERSION}.${TF_PATCH_VERSION}")
  else()
    set(TFLITE_C_PC_VERSION "2.21.0")
  endif()

  configure_file(
    tensorflowlite_c.pc.in
    ${CMAKE_CURRENT_BINARY_DIR}/tensorflowlite_c.pc
    @ONLY
  )

  install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/tensorflowlite_c.pc
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
  )
endif()
