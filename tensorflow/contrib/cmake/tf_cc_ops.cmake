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
########################################################
# tf_cc_framework library
########################################################
set(tf_cc_framework_srcs
    "${tensorflow_source_dir}/tensorflow/cc/framework/ops.h"
    "${tensorflow_source_dir}/tensorflow/cc/framework/ops.cc"
    "${tensorflow_source_dir}/tensorflow/cc/framework/scope.h"
    "${tensorflow_source_dir}/tensorflow/cc/framework/scope_internal.h"
    "${tensorflow_source_dir}/tensorflow/cc/framework/scope.cc"
)

add_library(tf_cc_framework OBJECT ${tf_cc_framework_srcs})

add_dependencies(tf_cc_framework tf_core_framework)

########################################################
# tf_cc_op_gen_main library
########################################################
set(tf_cc_op_gen_main_srcs
    "${tensorflow_source_dir}/tensorflow/cc/framework/cc_op_gen.cc"
    "${tensorflow_source_dir}/tensorflow/cc/framework/cc_op_gen_main.cc"
    "${tensorflow_source_dir}/tensorflow/cc/framework/cc_op_gen.h"
)

add_library(tf_cc_op_gen_main OBJECT ${tf_cc_op_gen_main_srcs})

add_dependencies(tf_cc_op_gen_main tf_core_framework)

########################################################
# tf_gen_op_wrapper_cc executables
########################################################

# create directory for ops generated files
set(cc_ops_target_dir ${CMAKE_CURRENT_BINARY_DIR}/tensorflow/cc/ops)

add_custom_target(create_cc_ops_header_dir
    COMMAND ${CMAKE_COMMAND} -E make_directory ${cc_ops_target_dir}
)

set(tf_cc_ops_generated_files)

set(tf_cc_op_lib_names
    ${tf_op_lib_names}
    "user_ops"
)
foreach(tf_cc_op_lib_name ${tf_cc_op_lib_names})
    # Using <TARGET_OBJECTS:...> to work around an issue where no ops were
    # registered (static initializers dropped by the linker because the ops
    # are not used explicitly in the *_gen_cc executables).
    add_executable(${tf_cc_op_lib_name}_gen_cc
        $<TARGET_OBJECTS:tf_cc_op_gen_main>
        $<TARGET_OBJECTS:tf_${tf_cc_op_lib_name}>
        $<TARGET_OBJECTS:tf_core_lib>
        $<TARGET_OBJECTS:tf_core_framework>
    )

    target_link_libraries(${tf_cc_op_lib_name}_gen_cc PRIVATE
        tf_protos_cc
        ${tensorflow_EXTERNAL_LIBRARIES}
    )

    set(cc_ops_include_internal 0)
    if(${tf_cc_op_lib_name} STREQUAL "sendrecv_ops")
        set(cc_ops_include_internal 1)
    endif()

    add_custom_command(
        OUTPUT ${cc_ops_target_dir}/${tf_cc_op_lib_name}.h
               ${cc_ops_target_dir}/${tf_cc_op_lib_name}.cc
               ${cc_ops_target_dir}/${tf_cc_op_lib_name}_internal.h
               ${cc_ops_target_dir}/${tf_cc_op_lib_name}_internal.cc
        COMMAND ${tf_cc_op_lib_name}_gen_cc ${cc_ops_target_dir}/${tf_cc_op_lib_name}.h ${cc_ops_target_dir}/${tf_cc_op_lib_name}.cc ${cc_ops_include_internal} ${tensorflow_source_dir}/tensorflow/core/api_def/base_api
        DEPENDS ${tf_cc_op_lib_name}_gen_cc create_cc_ops_header_dir
    )

    list(APPEND tf_cc_ops_generated_files ${cc_ops_target_dir}/${tf_cc_op_lib_name}.h)
    list(APPEND tf_cc_ops_generated_files ${cc_ops_target_dir}/${tf_cc_op_lib_name}.cc)
    list(APPEND tf_cc_ops_generated_files ${cc_ops_target_dir}/${tf_cc_op_lib_name}_internal.h)
    list(APPEND tf_cc_ops_generated_files ${cc_ops_target_dir}/${tf_cc_op_lib_name}_internal.cc)
endforeach()



########################################################
# tf_cc_ops library
########################################################
add_library(tf_cc_ops OBJECT
    ${tf_cc_ops_generated_files}
    "${tensorflow_source_dir}/tensorflow/cc/ops/const_op.h"
    "${tensorflow_source_dir}/tensorflow/cc/ops/const_op.cc"
    "${tensorflow_source_dir}/tensorflow/cc/ops/standard_ops.h"
)

########################################################
# tf_cc_while_loop library
########################################################
add_library(tf_cc_while_loop OBJECT
    "${tensorflow_source_dir}/tensorflow/cc/ops/while_loop.h"
    "${tensorflow_source_dir}/tensorflow/cc/ops/while_loop.cc"
)

add_dependencies(tf_cc_while_loop tf_core_framework tf_cc_ops)

########################################################
# tf_cc library
########################################################
file(GLOB_RECURSE tf_cc_srcs
    "${tensorflow_source_dir}/tensorflow/cc/client/*.h"
    "${tensorflow_source_dir}/tensorflow/cc/client/*.cc"
    "${tensorflow_source_dir}/tensorflow/cc/gradients/*.h"
    "${tensorflow_source_dir}/tensorflow/cc/gradients/*.cc"
    "${tensorflow_source_dir}/tensorflow/cc/training/*.h"
    "${tensorflow_source_dir}/tensorflow/cc/training/*.cc"
)

set(tf_cc_srcs
    ${tf_cc_srcs}
    "${tensorflow_source_dir}/tensorflow/cc/framework/grad_op_registry.h"
    "${tensorflow_source_dir}/tensorflow/cc/framework/grad_op_registry.cc"
    "${tensorflow_source_dir}/tensorflow/cc/framework/gradient_checker.h"
    "${tensorflow_source_dir}/tensorflow/cc/framework/gradient_checker.cc"
    "${tensorflow_source_dir}/tensorflow/cc/framework/gradients.h"
    "${tensorflow_source_dir}/tensorflow/cc/framework/gradients.cc"
    "${tensorflow_source_dir}/tensorflow/cc/framework/while_gradients.h"
    "${tensorflow_source_dir}/tensorflow/cc/framework/while_gradients.cc"
)

file(GLOB_RECURSE tf_cc_test_srcs
    "${tensorflow_source_dir}/tensorflow/cc/*test*.cc"
)

list(REMOVE_ITEM tf_cc_srcs ${tf_cc_test_srcs})

add_library(tf_cc OBJECT ${tf_cc_srcs})
add_dependencies(tf_cc tf_cc_framework tf_cc_ops)

if (WIN32)
  if(${CMAKE_GENERATOR} MATCHES "Visual Studio.*")
    set (pywrap_tensorflow_lib "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/pywrap_tensorflow_internal.lib")
  else()
    set (pywrap_tensorflow_lib "${CMAKE_CURRENT_BINARY_DIR}/pywrap_tensorflow_internal.lib")
  endif()
else (WIN32)
  set (pywrap_tensorflow_lib "${CMAKE_CURRENT_BINARY_DIR}/libpywrap_tensorflow_internal${CMAKE_SHARED_LIBRARY_SUFFIX}")
endif (WIN32)
add_custom_target(tf_extension_ops)

function(AddUserOps)
  cmake_parse_arguments(_AT "" "" "TARGET;SOURCES;GPUSOURCES;DEPENDS;DISTCOPY" ${ARGN})
  if (tensorflow_ENABLE_GPU AND _AT_GPUSOURCES)
    # if gpu build is enabled and we have gpu specific code,
    # hint to cmake that this needs to go to nvcc
    set (gpu_source ${_AT_GPUSOURCES})
    set (gpu_lib "${_AT_TARGET}_gpu")
    set_source_files_properties(${gpu_source} PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
    cuda_compile(gpu_lib ${gpu_source})
  endif()
  # create shared library from source and cuda obj
  add_library(${_AT_TARGET} SHARED ${_AT_SOURCES} ${gpu_lib})
  target_link_libraries(${_AT_TARGET} ${pywrap_tensorflow_lib})
  if (tensorflow_ENABLE_GPU AND _AT_GPUSOURCES)
      # some ops call out to cuda directly; need to link libs for the cuda dlls
      target_link_libraries(${_AT_TARGET} ${CUDA_LIBRARIES})
  endif()
  if (_AT_DISTCOPY)
      add_custom_command(TARGET ${_AT_TARGET} POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${_AT_TARGET}> ${_AT_DISTCOPY}/)
  endif()
  if (_AT_DEPENDS)
    add_dependencies(${_AT_TARGET} ${_AT_DEPENDS})
  endif()
  # make sure TF_COMPILE_LIBRARY is not defined for this target
  get_target_property(target_compile_flags  ${_AT_TARGET} COMPILE_FLAGS)
  if(target_compile_flags STREQUAL "target_compile_flags-NOTFOUND")
    if (WIN32)
      set(target_compile_flags "/UTF_COMPILE_LIBRARY")
    else (WIN32)
      # gcc uses UTF as default
      set(target_compile_flags "-finput-charset=UTF-8")
    endif (WIN32)
  else()
    if (WIN32)
      set(target_compile_flags "${target_compile_flags} /UTF_COMPILE_LIBRARY")
    else (WIN32)
      # gcc uses UTF as default
      set(target_compile_flags "${target_compile_flags} -finput-charset=UTF-8")
    endif (WIN32)
  endif()
  set_target_properties(${_AT_TARGET} PROPERTIES COMPILE_FLAGS ${target_compile_flags})
  add_dependencies(tf_extension_ops ${_AT_TARGET})
endfunction(AddUserOps)
