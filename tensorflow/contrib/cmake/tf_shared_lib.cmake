# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
if(WIN32)
  # Windows: build a static library with the same objects as tensorflow.dll.
  # This can be used to build for a standalone exe and also helps us to
  # find all symbols that need to be exported from the dll which is needed
  # to provide the tensorflow c/c++ api in tensorflow.dll.
  # From the static library we create the def file with all symbols that need to
  # be exported from tensorflow.dll. Because there is a limit of 64K sybmols
  # that can be exported, we filter the symbols with a python script to the namespaces
  # we need.
  #
  add_library(tensorflow_static STATIC
      $<TARGET_OBJECTS:tf_c>
      $<TARGET_OBJECTS:tf_cc>
      $<TARGET_OBJECTS:tf_cc_framework>
      $<TARGET_OBJECTS:tf_cc_ops>
      $<TARGET_OBJECTS:tf_cc_while_loop>
      $<TARGET_OBJECTS:tf_core_lib>
      $<TARGET_OBJECTS:tf_core_cpu>
      $<TARGET_OBJECTS:tf_core_framework>
      $<TARGET_OBJECTS:tf_core_ops>
      $<TARGET_OBJECTS:tf_core_direct_session>
      $<TARGET_OBJECTS:tf_tools_transform_graph_lib>
      $<$<BOOL:${tensorflow_ENABLE_GRPC_SUPPORT}>:$<TARGET_OBJECTS:tf_core_distributed_runtime>>
      $<TARGET_OBJECTS:tf_core_kernels>
      $<$<BOOL:${tensorflow_ENABLE_GPU}>:$<TARGET_OBJECTS:tf_core_kernels_cpu_only>>
      $<$<BOOL:${tensorflow_ENABLE_GPU}>:$<TARGET_OBJECTS:tf_stream_executor>>
  )

  add_dependencies(tensorflow_static tf_protos_cc)
  set(tensorflow_static_dependencies
      $<TARGET_FILE:tensorflow_static>
      $<TARGET_FILE:tf_protos_cc>
  )

  if(${CMAKE_GENERATOR} MATCHES "Visual Studio.*")
    set(tensorflow_deffile "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/tensorflow.def")
  else()
    set(tensorflow_deffile "${CMAKE_CURRENT_BINARY_DIR}/tensorflow.def")
  endif()
  set_source_files_properties(${tensorflow_deffile} PROPERTIES GENERATED TRUE)

  add_custom_command(TARGET tensorflow_static POST_BUILD
      COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/create_def_file.py
          --input "${tensorflow_static_dependencies}"
          --output "${tensorflow_deffile}"
          --target tensorflow.dll
  )
endif(WIN32)

# tensorflow is a shared library containing all of the
# TensorFlow runtime and the standard ops and kernels.
add_library(tensorflow SHARED
    $<TARGET_OBJECTS:tf_c>
    $<TARGET_OBJECTS:tf_cc>
    $<TARGET_OBJECTS:tf_cc_framework>
    $<TARGET_OBJECTS:tf_cc_ops>
    $<TARGET_OBJECTS:tf_cc_while_loop>
    $<TARGET_OBJECTS:tf_core_lib>
    $<TARGET_OBJECTS:tf_core_cpu>
    $<TARGET_OBJECTS:tf_core_framework>
    $<TARGET_OBJECTS:tf_core_ops>
    $<TARGET_OBJECTS:tf_core_direct_session>
    $<TARGET_OBJECTS:tf_tools_transform_graph_lib>
    $<$<BOOL:${tensorflow_ENABLE_GRPC_SUPPORT}>:$<TARGET_OBJECTS:tf_core_distributed_runtime>>
    $<TARGET_OBJECTS:tf_core_kernels>
    $<$<BOOL:${tensorflow_ENABLE_GPU}>:$<$<BOOL:${BOOL_WIN32}>:$<TARGET_OBJECTS:tf_core_kernels_cpu_only>>>
    $<$<BOOL:${tensorflow_ENABLE_GPU}>:$<TARGET_OBJECTS:tf_stream_executor>>
    ${tensorflow_deffile}
)

target_link_libraries(tensorflow PRIVATE
    ${tf_core_gpu_kernels_lib}
    ${tensorflow_EXTERNAL_LIBRARIES}
    tf_protos_cc
)

# There is a bug in GCC 5 resulting in undefined reference to a __cpu_model function when
# linking to the tensorflow library. Adding the following libraries fixes it.
# See issue on github: https://github.com/tensorflow/tensorflow/issues/9593
if(CMAKE_COMPILER_IS_GNUCC AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.0)
    target_link_libraries(tensorflow PRIVATE gcc_s gcc)
endif()

# Offer the user the choice of overriding the installation directories
set(INSTALL_LIB_DIR lib CACHE PATH "Installation directory for libraries")
set(INSTALL_BIN_DIR bin CACHE PATH "Installation directory for executables")
set(INSTALL_INCLUDE_DIR include CACHE PATH
  "Installation directory for header files")
if(WIN32 AND NOT CYGWIN)
  set(DEF_INSTALL_CMAKE_DIR cmake)
else()
  set(DEF_INSTALL_CMAKE_DIR lib/cmake)
endif()
set(INSTALL_CMAKE_DIR ${DEF_INSTALL_CMAKE_DIR} CACHE PATH
  "Installation directory for CMake files")

# Make relative paths absolute (needed later on)
foreach(p LIB BIN INCLUDE CMAKE)
  set(var INSTALL_${p}_DIR)
  if(NOT IS_ABSOLUTE "${${var}}")
    set(${var} "${CMAKE_INSTALL_PREFIX}/${${var}}")
  endif()
endforeach()

if(WIN32)
  add_dependencies(tensorflow tensorflow_static)
endif(WIN32)

target_include_directories(tensorflow PUBLIC 
    $<INSTALL_INTERFACE:include/>)

# Add all targets to build-tree export set
export(TARGETS tensorflow
  FILE ${PROJECT_BINARY_DIR}/TensorflowTargets.cmake)

# Export the package for use from the build-tree
export(PACKAGE Tensorflow)

# Create the TensorflowConfig.cmake and TensorflowConfigVersion files
file(RELATIVE_PATH REL_INCLUDE_DIR "${INSTALL_CMAKE_DIR}"
   "${INSTALL_INCLUDE_DIR}")
# for the build tree
set(CONF_INCLUDE_DIRS "${tensorflow_source_dir}" 
                      "${PROJECT_BINARY_DIR}"
                      "${CMAKE_CURRENT_BINARY_DIR}/protobuf/src/protobuf/src"
                      "${CMAKE_CURRENT_BINARY_DIR}/nsync/install/include" # Please if there is a better directory
                      "${CMAKE_CURRENT_BINARY_DIR}/eigen/src/eigen/Eigen/"
                      "${CMAKE_CURRENT_BINARY_DIR}/external/eigen_archive/"
                      "${tensorflow_source_dir}/third_party/eigen3/"
                      "${CMAKE_CURRENT_BINARY_DIR}/eigen/src/eigen/unsupported/Eigen/")
configure_file(TensorflowConfig.cmake.in
  "${PROJECT_BINARY_DIR}/TensorflowConfig.cmake" @ONLY)
# for the install tree, yet to be complete
set(CONF_INCLUDE_DIRS "\${TENSORFLOW_CMAKE_DIR}/${REL_INCLUDE_DIR}")
configure_file(TensorflowConfig.cmake.in
  "${PROJECT_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/TensorflowConfig.cmake" @ONLY)
# for both
configure_file(TensorflowConfigVersion.cmake.in
  "${PROJECT_BINARY_DIR}/TensorflowConfigVersion.cmake" @ONLY)

# install(TARGETS tensorflow EXPORT tensorflow_export
#         RUNTIME DESTINATION ${INSTALL_BIN_DIR}
#         LIBRARY DESTINATION ${INSTALL_LIB_DIR}
#         ARCHIVE DESTINATION ${INSTALL_LIB_DIR})

# install(EXPORT tensorflow_export
#         FILE TensorflowConfig.cmake
#         DESTINATION ${INSTALL_CMAKE_DIR})
        
install(FILES
  "${PROJECT_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/TensorflowConfig.cmake"
  "${PROJECT_BINARY_DIR}/TensorflowConfigVersion.cmake"
  DESTINATION "${INSTALL_CMAKE_DIR}" COMPONENT dev)

# install the export set for use with the install-tree
install(EXPORT TensorflowTargets 
  DESTINATION ${INSTALL_CMAKE_DIR})

install(TARGETS tensorflow EXPORT TensorflowTargets
        RUNTIME DESTINATION ${INSTALL_BIN_DIR}
        LIBRARY DESTINATION ${INSTALL_LIB_DIR}
        ARCHIVE DESTINATION ${INSTALL_LIB_DIR})

# install necessary headers
# tensorflow headers
install(DIRECTORY ${tensorflow_source_dir}/tensorflow/cc/
        DESTINATION include/tensorflow/cc
        FILES_MATCHING PATTERN "*.h")
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tensorflow/cc/
        DESTINATION include/tensorflow/cc
        FILES_MATCHING PATTERN "*.h")
install(DIRECTORY ${tensorflow_source_dir}/tensorflow/core/
        DESTINATION include/tensorflow/core
        FILES_MATCHING PATTERN "*.h")
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tensorflow/core/
        DESTINATION include/tensorflow/core
        FILES_MATCHING PATTERN "*.h")
install(DIRECTORY ${tensorflow_source_dir}/tensorflow/stream_executor/
        DESTINATION include/tensorflow/stream_executor
        FILES_MATCHING PATTERN "*.h")
# google protobuf headers
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/protobuf/src/protobuf/src/google/
        DESTINATION include/google
        FILES_MATCHING PATTERN "*.h")
# Eigen directory
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/eigen/src/eigen/Eigen/
        DESTINATION include/Eigen)
# external directory
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/external/eigen_archive/
        DESTINATION include/external/eigen_archive)
# third_party eigen directory
install(DIRECTORY ${tensorflow_source_dir}/third_party/eigen3/
        DESTINATION include/third_party/eigen3)
# unsupported Eigen directory
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/eigen/src/eigen/unsupported/Eigen/
        DESTINATION include/unsupported/Eigen)