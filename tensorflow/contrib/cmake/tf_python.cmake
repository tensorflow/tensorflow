# CMake rules for generating the TensorFlow Python bindings.
#
# Known limitations:
# * Generates output in a hard-coded path ${CMAKE_CURRENT_BINARY_DIR}/tf_python.
# * No support for dynamic library loading.
# * No support for tf.contrib. (TODO(mrry): Add rules for building op libraries.)
# * No support for Python 3. (TODO(mrry): Add override for FindPythonInterp.)
#
# The _pywrap_tensorflow target builds everything.

########################################################
# Resolve installed dependencies
########################################################

# 1. Resolve the installed version of Python (for Python.h and python).
# TODO(mrry): Parameterize the build script to enable Python 3 building.
include(FindPythonInterp)
if(NOT PYTHON_INCLUDE_DIR)
  set(PYTHON_NOT_FOUND false)
  exec_program("${PYTHON_EXECUTABLE}"
    ARGS "-c 'import distutils.sysconfig; print distutils.sysconfig.get_python_inc()'"
    OUTPUT_VARIABLE PYTHON_INCLUDE_DIR
    RETURN_VALUE PYTHON_NOT_FOUND)
  if(${PYTHON_NOT_FOUND})
    message(FATAL_ERROR
            "Cannot get Python include directory. Is distutils installed?")
  endif(${PYTHON_NOT_FOUND})
endif(NOT PYTHON_INCLUDE_DIR)
FIND_PACKAGE(PythonLibs)

# 2. Resolve the installed version of NumPy (for numpy/arrayobject.h).
if(NOT NUMPY_INCLUDE_DIR)
  set(NUMPY_NOT_FOUND false)
  exec_program("${PYTHON_EXECUTABLE}"
    ARGS "-c 'import numpy; print numpy.get_include()'"
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    RETURN_VALUE NUMPY_NOT_FOUND)
  if(${NUMPY_NOT_FOUND})
    message(FATAL_ERROR
            "Cannot get NumPy include directory: Is NumPy installed?")
  endif(${NUMPY_NOT_FOUND})
endif(NOT NUMPY_INCLUDE_DIR)

# 3. Resolve the installed version of zlib (for libz.so).
find_package(ZLIB REQUIRED)


########################################################
# Build the Python directory structure.
########################################################

# TODO(mrry): Configure this to build in a directory other than tf_python/
# TODO(mrry): Assemble the Python files into a PIP package.

# tf_python_srcs contains all static .py files
file(GLOB_RECURSE tf_python_srcs RELATIVE ${tensorflow_source_dir}
    "${tensorflow_source_dir}/tensorflow/python/*.py"
)
list(APPEND tf_python_srcs "tensorflow/__init__.py")

# tf_python_copy_scripts_to_destination copies all Python files
# (including static source and generated protobuf wrappers, but *not*
# generated TensorFlow op wrappers) into tf_python/.
add_custom_target(tf_python_copy_scripts_to_destination)

# Copy static files to tf_python/.
foreach(script ${tf_python_srcs})
  get_filename_component(REL_DIR ${script} DIRECTORY)
    add_custom_command(TARGET tf_python_copy_scripts_to_destination PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${tensorflow_source_dir}/${script} ${CMAKE_CURRENT_BINARY_DIR}/tf_python/${script})
endforeach()

# Generates the Python protobuf wrappers.
# ROOT_DIR must be absolute; subsequent arguments are interpreted as
# paths of .proto files, and must be relative to ROOT_DIR.
function(RELATIVE_PROTOBUF_GENERATE_PYTHON ROOT_DIR SRCS)
  if(NOT ARGN)
    message(SEND_ERROR "Error: RELATIVE_PROTOBUF_GENERATE_PYTHON() called without any proto files")
    return()
  endif()

  set(${SRCS})
  foreach(FIL ${ARGN})
    set(ABS_FIL ${ROOT_DIR}/${FIL})
    get_filename_component(FIL_WE ${FIL} NAME_WE)
    get_filename_component(FIL_DIR ${ABS_FIL} PATH)
    file(RELATIVE_PATH REL_DIR ${ROOT_DIR} ${FIL_DIR})

    list(APPEND ${SRCS} "${CMAKE_CURRENT_BINARY_DIR}/tf_python/${REL_DIR}/${FIL_WE}_pb2.py")
    add_custom_command(
      OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/tf_python/${REL_DIR}/${FIL_WE}_pb2.py"
      COMMAND ${PROTOBUF_PROTOC_EXECUTABLE}
      ARGS --python_out  ${CMAKE_CURRENT_BINARY_DIR}/tf_python/ -I ${ROOT_DIR} -I ${PROTOBUF_INCLUDE_DIRS} ${ABS_FIL} 
      DEPENDS ${PROTOBUF_PROTOC_EXECUTABLE} protobuf
      COMMENT "Running Python protocol buffer compiler on ${FIL}"
      VERBATIM )
  endforeach()
  set(${SRCS} ${${SRCS}} PARENT_SCOPE)
endfunction()

file(GLOB_RECURSE tf_protos_python_srcs RELATIVE ${tensorflow_source_dir}
    "${tensorflow_source_dir}/tensorflow/core/*.proto"
    "${tensorflow_source_dir}/tensorflow/python/*.proto"
)
RELATIVE_PROTOBUF_GENERATE_PYTHON(
    ${tensorflow_source_dir} PYTHON_PROTO_GENFILES ${tf_protos_python_srcs}
)

# tf_python_touchup_modules adds empty __init__.py files to all
# directories containing Python code, so that Python will recognize
# them as modules.
add_custom_target(tf_python_touchup_modules
  DEPENDS tf_python_copy_scripts_to_destination
)

function(add_python_module MODULE_NAME)
    add_custom_command(TARGET tf_python_touchup_modules PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/tf_python/${MODULE_NAME}") 
    add_custom_command(TARGET tf_python_touchup_modules PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E touch "${CMAKE_CURRENT_BINARY_DIR}/tf_python/${MODULE_NAME}/__init__.py")
endfunction()

add_python_module("tensorflow")
add_python_module("tensorflow/core")
add_python_module("tensorflow/core/example")
add_python_module("tensorflow/core/framework")
add_python_module("tensorflow/core/lib")
add_python_module("tensorflow/core/lib/core")
add_python_module("tensorflow/core/protobuf")
add_python_module("tensorflow/core/util")
add_python_module("tensorflow/python")
add_python_module("tensorflow/python/client")
add_python_module("tensorflow/python/framework")
add_python_module("tensorflow/python/ops")
add_python_module("tensorflow/python/kernel_tests")
add_python_module("tensorflow/python/lib")
add_python_module("tensorflow/python/lib/core")
add_python_module("tensorflow/python/lib/core/io")
add_python_module("tensorflow/python/platform")
add_python_module("tensorflow/python/platform/default")
add_python_module("tensorflow/python/platform/summary")
add_python_module("tensorflow/python/platform/summary/impl")
add_python_module("tensorflow/python/tools")
add_python_module("tensorflow/python/training")
add_python_module("tensorflow/python/util")
add_python_module("tensorflow/python/util/protobuf")
add_python_module("tensorflow/contrib")
add_python_module("tensorflow/contrib/bayesflow")
add_python_module("tensorflow/contrib/bayesflow/python")
add_python_module("tensorflow/contrib/bayesflow/python/ops")
add_python_module("tensorflow/contrib/bayesflow/python/ops/bernoulli")
add_python_module("tensorflow/contrib/framework")
add_python_module("tensorflow/contrib/framework/python")
add_python_module("tensorflow/contrib/framework/python/framework")
add_python_module("tensorflow/contrib/layers")
add_python_module("tensorflow/contrib/layers/python")
add_python_module("tensorflow/contrib/layers/python/layers")
add_python_module("tensorflow/contrib/layers/python/ops")



########################################################
# tf_python_op_gen_main library
########################################################
set(tf_python_op_gen_main_srcs
    "${tensorflow_source_dir}/tensorflow/python/framework/python_op_gen.cc"
    "${tensorflow_source_dir}/tensorflow/python/framework/python_op_gen_main.cc"
    "${tensorflow_source_dir}/tensorflow/python/framework/python_op_gen.h"
)

add_library(tf_python_op_gen_main OBJECT ${tf_python_op_gen_main_srcs})

add_dependencies(tf_python_op_gen_main tf_core_framework)

target_include_directories(tf_python_op_gen_main PRIVATE
    ${tensorflow_source_dir}
    ${eigen_INCLUDE_DIRS}
)

target_compile_options(tf_python_op_gen_main PRIVATE
    -fno-exceptions
    -DEIGEN_AVOID_STL_ARRAY
)

# C++11
target_compile_features(tf_python_op_gen_main PRIVATE
    cxx_rvalue_references
)

# create directory for ops generated files
set(python_ops_target_dir ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/python/ops)

set(tf_python_ops_generated_files)

set(tf_python_op_lib_names
    ${tf_op_lib_names}
    "user_ops"
)

function(GENERATE_PYTHON_OP_LIB tf_python_op_lib_name)
    set(oneValueArgs DESTINATION)
    set(multiValueArgs ADDITIONAL_LIBRARIES)
    cmake_parse_arguments(GENERATE_PYTHON_OP_LIB
      "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    if(NOT DEFINED GENERATE_PYTHON_OP_LIB_DESTINATION)
      # Default destination is tf_python/tensorflow/python/ops/gen_<...>.py.
      set(GENERATE_PYTHON_OP_LIB_DESTINATION
          "${python_ops_target_dir}/gen_${tf_python_op_lib_name}.py")
    endif()

    # Create a C++ executable that links in the appropriate op
    # registrations and generates Python wrapper code based on the
    # registered ops.
    add_executable(${tf_python_op_lib_name}_gen_python
        $<TARGET_OBJECTS:tf_python_op_gen_main>
        $<TARGET_OBJECTS:tf_${tf_python_op_lib_name}>
        $<TARGET_OBJECTS:tf_core_lib>
        $<TARGET_OBJECTS:tf_core_framework>
	${GENERATE_PYTHON_OP_LIB_ADDITIONAL_LIBRARIES}
    )
    target_include_directories(${tf_python_op_lib_name}_gen_python PRIVATE
        ${tensorflow_source_dir}
        ${eigen_INCLUDE_DIRS}
    )
    target_link_libraries(${tf_python_op_lib_name}_gen_python PRIVATE
        ${CMAKE_THREAD_LIBS_INIT}
        ${PROTOBUF_LIBRARIES}
        tf_protos_cc
        ${gif_STATIC_LIBRARIES}
	${jpeg_STATIC_LIBRARIES}
        ${png_STATIC_LIBRARIES}
        ${ZLIB_LIBRARIES}
        ${jsoncpp_STATIC_LIBRARIES}
        ${CMAKE_DL_LIBS}
    )
    target_compile_options(${tf_python_op_lib_name}_gen_python PRIVATE
        -fno-exceptions
        -DEIGEN_AVOID_STL_ARRAY
        -lm
    )
    # C++11
    target_compile_features(${tf_python_op_lib_name}_gen_python PRIVATE
        cxx_rvalue_references
    )
    if(tensorflow_ENABLE_SSL_SUPPORT)
      target_link_libraries(${tf_python_op_lib_name}_gen_python PRIVATE
          ${boringssl_STATIC_LIBRARIES})
    endif()
  
    # Use the generated C++ executable to create a Python file
    # containing the wrappers.
    add_custom_command(
      OUTPUT ${GENERATE_PYTHON_OP_LIB_DESTINATION}
      COMMAND ${tf_python_op_lib_name}_gen_python @${tensorflow_source_dir}/tensorflow/python/ops/hidden_ops.txt 1 > ${GENERATE_PYTHON_OP_LIB_DESTINATION}
      DEPENDS ${tf_python_op_lib_name}_gen_python
    )
    
    set(tf_python_ops_generated_files ${tf_python_ops_generated_files}
        ${GENERATE_PYTHON_OP_LIB_DESTINATION} PARENT_SCOPE)
endfunction()

GENERATE_PYTHON_OP_LIB("array_ops")
GENERATE_PYTHON_OP_LIB("math_ops")
GENERATE_PYTHON_OP_LIB("functional_ops")
GENERATE_PYTHON_OP_LIB("candidate_sampling_ops")
GENERATE_PYTHON_OP_LIB("control_flow_ops"
  ADDITIONAL_LIBRARIES $<TARGET_OBJECTS:tf_no_op>)
GENERATE_PYTHON_OP_LIB("ctc_ops")
GENERATE_PYTHON_OP_LIB("data_flow_ops")
GENERATE_PYTHON_OP_LIB("image_ops")
GENERATE_PYTHON_OP_LIB("io_ops")
GENERATE_PYTHON_OP_LIB("linalg_ops")
GENERATE_PYTHON_OP_LIB("logging_ops")
GENERATE_PYTHON_OP_LIB("nn_ops")
GENERATE_PYTHON_OP_LIB("parsing_ops")
GENERATE_PYTHON_OP_LIB("random_ops")
GENERATE_PYTHON_OP_LIB("script_ops")
GENERATE_PYTHON_OP_LIB("state_ops")
GENERATE_PYTHON_OP_LIB("sparse_ops")
GENERATE_PYTHON_OP_LIB("string_ops")
GENERATE_PYTHON_OP_LIB("user_ops")
GENERATE_PYTHON_OP_LIB("training_ops"
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/python/training/gen_training_ops.py)

add_custom_target(tf_python_ops SOURCES ${tf_python_ops_generated_files} ${PYTHON_PROTO_GENFILES})
add_dependencies(tf_python_ops tf_python_op_gen_main)


############################################################
# Build the SWIG-wrapped library for the TensorFlow runtime.
############################################################

find_package(SWIG REQUIRED)
# Generate the C++ and Python source code for the SWIG wrapper.
add_custom_command(
      OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/python/pywrap_tensorflow.py"
             "${CMAKE_CURRENT_BINARY_DIR}/pywrap_tensorflow.cc"
      DEPENDS tf_python_touchup_modules
      COMMAND ${SWIG_EXECUTABLE}
      ARGS -python -c++
           -I${tensorflow_source_dir}
           -I${CMAKE_CURRENT_BINARY_DIR}
           -module pywrap_tensorflow
           -outdir ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/python
           -o ${CMAKE_CURRENT_BINARY_DIR}/pywrap_tensorflow.cc
           -globals ''
           ${tensorflow_source_dir}/tensorflow/python/tensorflow.i
      COMMENT "Running SWIG to generate Python wrappers"
      VERBATIM )

# pywrap_tensorflow is a shared library containing all of the TensorFlow
# runtime and the standard ops and kernels. These are installed into
# tf_python/tensorflow/python/.
# TODO(mrry): Refactor this to expose a framework library that
# facilitates `tf.load_op_library()`.
add_library(pywrap_tensorflow SHARED
    "${tensorflow_source_dir}/tensorflow/python/client/tf_session_helper.h"
    "${tensorflow_source_dir}/tensorflow/python/client/tf_session_helper.cc"
    "${tensorflow_source_dir}/tensorflow/python/framework/cpp_shape_inference.h"
    "${tensorflow_source_dir}/tensorflow/python/framework/cpp_shape_inference.cc"
    "${tensorflow_source_dir}/tensorflow/python/framework/python_op_gen.h"
    "${tensorflow_source_dir}/tensorflow/python/framework/python_op_gen.cc"
    "${tensorflow_source_dir}/tensorflow/python/lib/core/numpy.h"
    "${tensorflow_source_dir}/tensorflow/python/lib/core/numpy.cc"
    "${tensorflow_source_dir}/tensorflow/python/lib/core/py_func.h"
    "${tensorflow_source_dir}/tensorflow/python/lib/core/py_func.cc"
    "${tensorflow_source_dir}/tensorflow/python/lib/io/py_record_reader.h"
    "${tensorflow_source_dir}/tensorflow/python/lib/io/py_record_reader.cc"
    "${tensorflow_source_dir}/tensorflow/python/lib/io/py_record_writer.h"
    "${tensorflow_source_dir}/tensorflow/python/lib/io/py_record_writer.cc"
    "${tensorflow_source_dir}/tensorflow/c/c_api.cc"
    "${tensorflow_source_dir}/tensorflow/c/c_api.h"
    "${tensorflow_source_dir}/tensorflow/c/checkpoint_reader.cc"
    "${tensorflow_source_dir}/tensorflow/c/checkpoint_reader.h"
    "${tensorflow_source_dir}/tensorflow/c/tf_status_helper.cc"
    "${tensorflow_source_dir}/tensorflow/c/tf_status_helper.h"
    "${CMAKE_CURRENT_BINARY_DIR}/pywrap_tensorflow.cc"
    $<TARGET_OBJECTS:tf_core_lib>
    $<TARGET_OBJECTS:tf_core_cpu>
    $<TARGET_OBJECTS:tf_core_framework>
    $<TARGET_OBJECTS:tf_core_ops>
    $<TARGET_OBJECTS:tf_core_direct_session>
    $<TARGET_OBJECTS:tf_core_distributed_runtime>
    $<TARGET_OBJECTS:tf_core_kernels>
)
target_link_libraries(pywrap_tensorflow
    ${CMAKE_THREAD_LIBS_INIT}
    tf_protos_cc
    ${GRPC_LIBRARIES}
    ${PROTOBUF_LIBRARY}
    ${farmhash_STATIC_LIBRARIES}
    ${gif_STATIC_LIBRARIES}
    ${jpeg_STATIC_LIBRARIES}
    ${jsoncpp_STATIC_LIBRARIES}
    ${png_STATIC_LIBRARIES}
    ${ZLIB_LIBRARIES}
    ${CMAKE_DL_LIBS}
)
target_include_directories(pywrap_tensorflow PUBLIC
    ${tensorflow_source_dir}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${eigen_INCLUDE_DIRS}
    ${PYTHON_INCLUDE_DIR}
    ${NUMPY_INCLUDE_DIR}
)
# C++11
target_compile_features(pywrap_tensorflow PRIVATE
    cxx_rvalue_references
)
if(tensorflow_ENABLE_SSL_SUPPORT)
  target_link_libraries(pywrap_tensorflow ${boringssl_STATIC_LIBRARIES})
endif()



############################################################
# Build a PIP package containing the TensorFlow runtime.
############################################################
add_custom_target(tf_python_build_pip_package)
add_dependencies(tf_python_build_pip_package
    pywrap_tensorflow
    tf_python_copy_scripts_to_destination
    tf_python_touchup_modules
    tf_python_ops)
add_custom_command(TARGET tf_python_build_pip_package POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy ${tensorflow_source_dir}/tensorflow/contrib/cmake/setup.py
                                   ${CMAKE_CURRENT_BINARY_DIR}/tf_python/)
add_custom_command(TARGET tf_python_build_pip_package POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/libpywrap_tensorflow.so
                                   ${CMAKE_CURRENT_BINARY_DIR}/tf_python/tensorflow/python/_pywrap_tensorflow.so)
add_custom_command(TARGET tf_python_build_pip_package POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy ${tensorflow_source_dir}/tensorflow/tools/pip_package/README
                                   ${CMAKE_CURRENT_BINARY_DIR}/tf_python/)
add_custom_command(TARGET tf_python_build_pip_package POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy ${tensorflow_source_dir}/tensorflow/tools/pip_package/MANIFEST.in
                                   ${CMAKE_CURRENT_BINARY_DIR}/tf_python/)
add_custom_command(TARGET tf_python_build_pip_package POST_BUILD
  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/tf_python/setup.py bdist_wheel
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tf_python)
