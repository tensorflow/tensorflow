enable_testing()

#
# get a temp path for test data
#
function(GetTestRunPath VAR_NAME OBJ_NAME)
    if(WIN32)
      if(DEFINED ENV{TMP})
        set(TMPDIR "$ENV{TMP}")
      elseif(DEFINED ENV{TEMP})
        set(TMPDIR "$ENV{TEMP}")
      endif()
      string(REPLACE "\\" "/" TMPDIR ${TMPDIR})
    else()
      set(TMPDIR "$ENV{TMPDIR}")
    endif()
    if(NOT EXISTS "${TMPDIR}")
       message(FATAL_ERROR "Unable to determine a path to the temporary directory")
    endif()
    set(${VAR_NAME} "${TMPDIR}/${OBJ_NAME}" PARENT_SCOPE)
endfunction(GetTestRunPath)

#
# create test for each source
#
function(AddTests)
  cmake_parse_arguments(_AT "" "" "SOURCES;OBJECTS;LIBS;DATA;DEPENDS" ${ARGN})
  foreach(sourcefile ${_AT_SOURCES})
    string(REPLACE "${tensorflow_source_dir}/" "" exename ${sourcefile})
    string(REPLACE ".cc" "" exename ${exename})
    string(REPLACE "/" "_" exename ${exename})
    AddTest(
      TARGET ${exename}
      SOURCES ${sourcefile}
      OBJECTS ${_AT_OBJECTS}
      LIBS ${_AT_LIBS}
      DATA ${_AT_DATA}
      DEPENDS ${_AT_DEPENDS}
    )
  endforeach()
endfunction(AddTests)

#
# create once test
#
function(AddTest)
  cmake_parse_arguments(_AT "" "TARGET" "SOURCES;OBJECTS;LIBS;DATA;DEPENDS" ${ARGN})

  list(REMOVE_DUPLICATES _AT_SOURCES)
  list(REMOVE_DUPLICATES _AT_OBJECTS)
  list(REMOVE_DUPLICATES _AT_LIBS)
  if (_AT_DATA)
    list(REMOVE_DUPLICATES _AT_DATA)
  endif(_AT_DATA)
  if (_AT_DEPENDS)
    list(REMOVE_DUPLICATES _AT_DEPENDS)
  endif(_AT_DEPENDS)

  add_executable(${_AT_TARGET} ${_AT_SOURCES} ${_AT_OBJECTS})
  target_link_libraries(${_AT_TARGET} ${_AT_LIBS})

  GetTestRunPath(testdir ${_AT_TARGET})
  set(tempdir "${testdir}/tmp")
  file(REMOVE_RECURSE "${testdir}")
  file(MAKE_DIRECTORY "${testdir}")
  file(MAKE_DIRECTORY "${tempdir}")
  add_test(NAME ${_AT_TARGET} COMMAND ${_AT_TARGET} WORKING_DIRECTORY "${testdir}")
  set_tests_properties(${_AT_TARGET}
    PROPERTIES ENVIRONMENT "TEST_TMPDIR=${tempdir};TEST_SRCDIR=${testdir}"
  )

  foreach(datafile ${_AT_DATA})
    file(RELATIVE_PATH datafile_rel ${tensorflow_source_dir} ${datafile})
    add_custom_command(
      TARGET ${_AT_TARGET} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy
             "${datafile}"
             "${testdir}/${datafile_rel}"
      DEPENDS "${datafile}"
    )
  endforeach()

  if (_AT_DEPENDS)
    add_dependencies(${_AT_TARGET} ${_AT_DEPENDS} googletest)
  endif()
endfunction(AddTest)

#
# create python test for each script
#
function(AddPythonTests)
  cmake_parse_arguments(_AT "" "" "SOURCES;DATA;DEPENDS" ${ARGN})
  list(REMOVE_DUPLICATES _AT_SOURCES)
  if (_AT_DATA)
    list(REMOVE_DUPLICATES _AT_DATA)
  endif(_AT_DATA)
  if (_AT_DEPENDS)
    list(REMOVE_DUPLICATES _AT_DEPENDS)
  endif(_AT_DEPENDS)

  foreach(sourcefile ${_AT_SOURCES})
    add_test(NAME ${sourcefile} COMMAND ${PYTHON_EXECUTABLE} ${sourcefile} WORKING_DIRECTORY ${tensorflow_source_dir})
    if (_AT_DEPENDS)
      add_dependencies(${_AT_TARGET} ${_AT_DEPENDS})
    endif()
  endforeach()
endfunction(AddPythonTests)

if (tensorflow_BUILD_PYTHON_TESTS)
  #
  # python tests. This assumes that the tensorflow wheel is
  # installed on the test system.
  # TODO: we currently don't handle tests that need to have
  # some environment setup: see AddTest how to add this
  #

  # include all test
  file(GLOB_RECURSE tf_test_src_py
    "${tensorflow_source_dir}/tensorflow/python/debug/cli/*_test.py"
    "${tensorflow_source_dir}/tensorflow/python/debug/lib/*_test.py"
    "${tensorflow_source_dir}/tensorflow/python/debug/wrappers/*_test.py"
    "${tensorflow_source_dir}/tensorflow/python/kernel_tests/*.py"
    "${tensorflow_source_dir}/tensorflow/python/saved_model/*_test.py"
    "${tensorflow_source_dir}/tensorflow/python/training/*_test.py"
    "${tensorflow_source_dir}/tensorflow/tensorboard/*_test.py"
    # NOTE: tensor_forest tests in tensor_forest/hybrid/... still don't pass.
    "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/client/*_test.py"
    "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/python/*_test.py"
  )

  # exclude the onces we don't want
  set(tf_test_src_py_exclude
    # Windows does not have the curses library and uses readline.
    "${tensorflow_source_dir}/tensorflow/python/debug/cli/curses_ui_test.py"
    # generally not working
    "${tensorflow_source_dir}/tensorflow/python/kernel_tests/__init__.py"
    "${tensorflow_source_dir}/tensorflow/python/kernel_tests/benchmark_test.py"
    "${tensorflow_source_dir}/tensorflow/python/kernel_tests/resource_variable_ops_test.py"
    "${tensorflow_source_dir}/tensorflow/python/saved_model/saved_model_test.py"
  )
  if (WIN32)
    set(tf_test_src_py_exclude
      ${tf_test_src_py_exclude}
      # generally excluded
      "${tensorflow_source_dir}/tensorflow/python/kernel_tests/__init__.py"

      # TODO: failing tests.
      # Nothing critical in here but should get this list down to []
      # The failing list is grouped by failure source
      # stl on windows handles overflows different
      "${tensorflow_source_dir}/tensorflow/python/kernel_tests/as_string_op_test.py"
      "${tensorflow_source_dir}/tensorflow/python/kernel_tests/cast_op_test.py"
      "${tensorflow_source_dir}/tensorflow/python/kernel_tests/string_to_number_op_test.py"
      "${tensorflow_source_dir}/tensorflow/python/kernel_tests/clip_ops_test.py"
      # misc
      "${tensorflow_source_dir}/tensorflow/python/kernel_tests/variable_scope_test.py"
      "${tensorflow_source_dir}/tensorflow/python/kernel_tests/reshape_op_test.py"
      "${tensorflow_source_dir}/tensorflow/tensorboard/backend/server_test.py"
      "${tensorflow_source_dir}/tensorflow/python/kernel_tests/diag_op_test.py"  # Silently failing with GPU kernel disabled.
      # int32/int64 mixup
      "${tensorflow_source_dir}/tensorflow/python/kernel_tests/functional_ops_test.py"
      "${tensorflow_source_dir}/tensorflow/python/kernel_tests/py_func_test.py"
      # training tests
      "${tensorflow_source_dir}/tensorflow/python/training/basic_session_run_hooks_test.py"  # Needs tf.contrib fix.
      "${tensorflow_source_dir}/tensorflow/python/training/evaluation_test.py"  # Needs tf.contrib fix.
      "${tensorflow_source_dir}/tensorflow/python/training/localhost_cluster_performance_test.py"  # Needs portpicker.
      "${tensorflow_source_dir}/tensorflow/python/training/monitored_session_test.py"  # Needs tf.contrib fix.
      "${tensorflow_source_dir}/tensorflow/python/training/saver_large_variable_test.py"  # Overflow error.
      "${tensorflow_source_dir}/tensorflow/python/training/supervisor_test.py"  # Flaky I/O error on rename.
      "${tensorflow_source_dir}/tensorflow/python/training/sync_replicas_optimizer_test.py"  # Needs portpicker.
      "${tensorflow_source_dir}/tensorflow/python/kernel_tests/array_ops_test.py"  # depends on python/framework/test_ops
      # Broken TensorBoard tests due to different paths in windows
      "${tensorflow_source_dir}/tensorflow/tensorboard/backend/application_test.py"
      "${tensorflow_source_dir}/tensorflow/tensorboard/lib/python/http_util_test.py"
      # Broken tensorboard test due to cmake issues.
      "${tensorflow_source_dir}/tensorflow/tensorboard/plugins/debugger/plugin_test.py"
      # tensor_forest tests (also note that we exclude the hybrid tests for now)
      "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/python/kernel_tests/count_extremely_random_stats_op_test.py"  # Results in wrong order.
      "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/python/kernel_tests/sample_inputs_op_test.py"  # Results in wrong order.
      "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/python/kernel_tests/scatter_add_ndim_op_test.py"  # Bad placement.
      "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/python/topn_test.py"  # Results inaccurate
    )
  endif()
  list(REMOVE_ITEM tf_test_src_py ${tf_test_src_py_exclude})

  AddPythonTests(
    SOURCES ${tf_test_src_py}
  )
endif(tensorflow_BUILD_PYTHON_TESTS)

if (tensorflow_BUILD_CC_TESTS)
  #
  # cc unit tests. Be aware that by default we include 250+ tests which
  # will take time and space to build.
  # If you want to cut this down, for example to a specific test, modify
  # tf_test_src_simple to your needs
  #

  include_directories(${googletest_INCLUDE_DIRS})

  # cc tests wrapper
  set(tf_src_testlib
    "${tensorflow_source_dir}/tensorflow/cc/framework/testutil.cc"
    "${tensorflow_source_dir}/tensorflow/cc/gradients/grad_testutil.cc"
    "${tensorflow_source_dir}/tensorflow/core/common_runtime/kernel_benchmark_testlib.cc"
    "${tensorflow_source_dir}/tensorflow/core/framework/function_testlib.cc"
    "${tensorflow_source_dir}/tensorflow/core/framework/shape_inference_testutil.cc"
    "${tensorflow_source_dir}/tensorflow/core/framework/tensor_testutil.cc"
    "${tensorflow_source_dir}/tensorflow/core/graph/testlib.cc"
    "${tensorflow_source_dir}/tensorflow/core/platform/test.cc"
    "${tensorflow_source_dir}/tensorflow/core/platform/test_main.cc"
    "${tensorflow_source_dir}/tensorflow/core/platform/default/test_benchmark.cc"
    "${tensorflow_source_dir}/tensorflow/c/c_api.cc"
    "${tensorflow_source_dir}/tensorflow/c/checkpoint_reader.cc"
    "${tensorflow_source_dir}/tensorflow/c/tf_status_helper.cc"
  )

  if(WIN32)
     set(tf_src_testlib
       ${tf_src_testlib}
       "${tensorflow_source_dir}/tensorflow/core/platform/windows/test.cc"
     )
  else()
     set(tf_src_testlib
       ${tf_src_testlib}
       "${tensorflow_source_dir}/tensorflow/core/platform/posix/test.cc"
     )
  endif()

  # include all test
  file(GLOB_RECURSE tf_test_src_simple
    "${tensorflow_source_dir}/tensorflow/cc/*_test.cc"
    "${tensorflow_source_dir}/tensorflow/python/*_test.cc"
    "${tensorflow_source_dir}/tensorflow/core/*_test.cc"
    "${tensorflow_source_dir}/tensorflow/user_ops/*_test.cc"
    "${tensorflow_source_dir}/tensorflow/contrib/rnn/*_test.cc"
  )

  # exclude the ones we don't want
  set(tf_test_src_simple_exclude
    # generally not working
    "${tensorflow_source_dir}/tensorflow/cc/client/client_session_test.cc"
    "${tensorflow_source_dir}/tensorflow/cc/framework/gradients_test.cc"
    "${tensorflow_source_dir}/tensorflow/core/distributed_runtime/call_options_test.cc"
    "${tensorflow_source_dir}/tensorflow/core/distributed_runtime/tensor_coding_test.cc"
    "${tensorflow_source_dir}/tensorflow/core/kernels/hexagon/graph_transferer_test.cc"
    "${tensorflow_source_dir}/tensorflow/core/kernels/hexagon/quantized_matmul_op_for_hexagon_test.cc"
  )

  if (NOT tensorflow_ENABLE_GPU)
    # exclude gpu tests if we are not buildig for gpu
    set(tf_test_src_simple_exclude
      ${tf_test_src_simple_exclude}
      "${tensorflow_source_dir}/tensorflow/core/common_runtime/gpu/gpu_allocator_retry_test.cc"
      "${tensorflow_source_dir}/tensorflow/core/common_runtime/gpu/gpu_bfc_allocator_test.cc"
      "${tensorflow_source_dir}/tensorflow/core/common_runtime/gpu/gpu_debug_allocator_test.cc"
      "${tensorflow_source_dir}/tensorflow/core/common_runtime/gpu/gpu_event_mgr_test.cc"
      "${tensorflow_source_dir}/tensorflow/core/common_runtime/gpu/gpu_stream_util_test.cc"
      "${tensorflow_source_dir}/tensorflow/core/common_runtime/gpu/pool_allocator_test.cc"
    )
  endif()

  if (WIN32)
    set(tf_test_src_simple_exclude
      ${tf_test_src_simple_exclude}
      # generally excluded
      "${tensorflow_source_dir}/tensorflow/contrib/ffmpeg/default/ffmpeg_lib_test.cc"
      "${tensorflow_source_dir}/tensorflow/cc/framework/cc_ops_test.cc" # test_op.h missing

      # TODO: test failing
      "${tensorflow_source_dir}/tensorflow/core/common_runtime/simple_placer_test.cc"
      "${tensorflow_source_dir}/tensorflow/core/debug/debug_gateway_test.cc" # hangs
      "${tensorflow_source_dir}/tensorflow/core/distributed_runtime/executor_test.cc"
      "${tensorflow_source_dir}/tensorflow/core/kernels/quantized_reshape_op_test.cc"
      "${tensorflow_source_dir}/tensorflow/core/kernels/requantization_range_op_test.cc"
      "${tensorflow_source_dir}/tensorflow/core/kernels/requantize_op_test.cc"
      "${tensorflow_source_dir}/tensorflow/core/kernels/quantize_op_test.cc"
      "${tensorflow_source_dir}/tensorflow/core/lib/strings/str_util_test.cc"
      "${tensorflow_source_dir}/tensorflow/core/lib/strings/numbers_test.cc"
      "${tensorflow_source_dir}/tensorflow/core/lib/monitoring/collection_registry_test.cc"
      "${tensorflow_source_dir}/tensorflow/core/platform/file_system_test.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/cudnn_rnn/cudnn_rnn_ops_test.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/rnn/ops/gru_ops_test.cc" # status 5
      "${tensorflow_source_dir}/tensorflow/contrib/rnn/ops/lstm_ops_test.cc" # status 5

      # TODO: not compiling 
      "${tensorflow_source_dir}/tensorflow/core/kernels/quantization_utils_test.cc"
      "${tensorflow_source_dir}/tensorflow/core/kernels/quantize_and_dequantize_op_test.cc"
      "${tensorflow_source_dir}/tensorflow/core/kernels/quantize_down_and_shrink_range_op_test.cc"
      "${tensorflow_source_dir}/tensorflow/core/kernels/debug_ops_test.cc"
      "${tensorflow_source_dir}/tensorflow/core/kernels/quantized_activation_ops_test.cc"
      "${tensorflow_source_dir}/tensorflow/core/kernels/quantized_bias_add_op_test.cc"
      "${tensorflow_source_dir}/tensorflow/core/kernels/quantized_concat_op_test.cc"
      "${tensorflow_source_dir}/tensorflow/core/kernels/quantized_conv_ops_test.cc"
      "${tensorflow_source_dir}/tensorflow/core/kernels/quantized_matmul_op_test.cc"
      "${tensorflow_source_dir}/tensorflow/core/kernels/quantized_pooling_ops_test.cc"
      "${tensorflow_source_dir}/tensorflow/core/kernels/quantized_batch_norm_op_test.cc"
      "${tensorflow_source_dir}/tensorflow/core/kernels/cloud/bigquery_table_accessor_test.cc"
      "${tensorflow_source_dir}/tensorflow/core/platform/cloud/gcs_file_system_test.cc"
      "${tensorflow_source_dir}/tensorflow/core/platform/cloud/google_auth_provider_test.cc"
      "${tensorflow_source_dir}/tensorflow/core/platform/cloud/http_request_test.cc"
      "${tensorflow_source_dir}/tensorflow/core/platform/cloud/oauth_client_test.cc"
      "${tensorflow_source_dir}/tensorflow/core/platform/cloud/retrying_file_system_test.cc"
      "${tensorflow_source_dir}/tensorflow/core/platform/cloud/time_util_test.cc"
      "${tensorflow_source_dir}/tensorflow/core/platform/hadoop/hadoop_file_system_test.cc"
      "${tensorflow_source_dir}/tensorflow/core/platform/profile_utils/cpu_utils_test.cc"
      "${tensorflow_source_dir}/tensorflow/core/platform/subprocess_test.cc"
      "${tensorflow_source_dir}/tensorflow/core/common_runtime/gpu/gpu_allocator_retry_test.cc"
      "${tensorflow_source_dir}/tensorflow/core/common_runtime/gpu/gpu_debug_allocator_test.cc"
      "${tensorflow_source_dir}/tensorflow/core/distributed_runtime/master_test.cc"
      "${tensorflow_source_dir}/tensorflow/core/distributed_runtime/remote_device_test.cc"
      "${tensorflow_source_dir}/tensorflow/core/distributed_runtime/rpc/grpc_channel_test.cc"
      "${tensorflow_source_dir}/tensorflow/core/distributed_runtime/rpc/grpc_session_test.cc"
      "${tensorflow_source_dir}/tensorflow/core/distributed_runtime/rpc/grpc_tensor_coding_test.cc"
      "${tensorflow_source_dir}/tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr_test.cc"
      "${tensorflow_source_dir}/tensorflow/core/distributed_runtime/master_test.cc"
      "${tensorflow_source_dir}/tensorflow/core/example/example_parser_configuration_test.cc"
      "${tensorflow_source_dir}/tensorflow/core/example/feature_util_test.cc"
      "${tensorflow_source_dir}/tensorflow/core/util/reporter_test.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/factorization/kernels/clustering_ops_test.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/session_bundle/bundle_shim_test.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/session_bundle/bundle_test.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/session_bundle/signature_test.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/core/ops/training_ops_test.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/core/ops/tree_utils_test.cc"
    )
  endif()

  # Tests for saved_model require data, so need to treat them separately.
  file(GLOB tf_cc_saved_model_test_srcs
    "${tensorflow_source_dir}/tensorflow/cc/saved_model/*_test.cc"
  )

  list(REMOVE_ITEM tf_test_src_simple
    ${tf_test_src_simple_exclude}
    ${tf_cc_saved_model_test_srcs}
  )

  set(tf_test_lib tf_test_lib)
  add_library(${tf_test_lib} STATIC ${tf_src_testlib})

  # this is giving to much objects and libraries to the linker but
  # it makes this script much easier. So for now we do it this way.
  set(tf_obj_test
    $<TARGET_OBJECTS:tf_core_lib>
    $<TARGET_OBJECTS:tf_core_cpu>
    $<TARGET_OBJECTS:tf_core_framework>
    $<TARGET_OBJECTS:tf_core_kernels>
    $<TARGET_OBJECTS:tf_cc>
    $<TARGET_OBJECTS:tf_cc_framework>
    $<TARGET_OBJECTS:tf_cc_ops>
    $<TARGET_OBJECTS:tf_core_ops>
    $<TARGET_OBJECTS:tf_core_direct_session>
    $<$<BOOL:${tensorflow_ENABLE_GPU}>:$<TARGET_OBJECTS:tf_stream_executor>>
  )

  set(tf_test_libs
    tf_protos_cc
    tf_test_lib
    ${tf_core_gpu_kernels_lib}
    ${googletest_STATIC_LIBRARIES}
    ${tensorflow_EXTERNAL_LIBRARIES}
  )

  # All tests that require no data.
  AddTests(
    SOURCES ${tf_test_src_simple}
    OBJECTS ${tf_obj_test}
    LIBS ${tf_test_libs}
  )

  # Tests for tensorflow/cc/saved_model.
  file(GLOB_RECURSE tf_cc_saved_model_test_data
    "${tensorflow_source_dir}/tensorflow/cc/saved_model/testdata/*"
  )

  AddTests(
    SOURCES ${tf_cc_saved_model_test_srcs}
    DATA ${tf_cc_saved_model_test_data}
    OBJECTS ${tf_obj_test}
    LIBS ${tf_test_libs}
  )

endif(tensorflow_BUILD_CC_TESTS)
