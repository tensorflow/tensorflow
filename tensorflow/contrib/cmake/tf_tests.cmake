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
  set_tests_properties(${_AT_TARGET} PROPERTIES TIMEOUT "600")

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
    set_tests_properties(${sourcefile} PROPERTIES TIMEOUT "600")
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
  if (WIN32)
    file(GLOB_RECURSE tf_test_rnn_src_py
      "${tensorflow_source_dir}/tensorflow/contrib/rnn/python/kernel_tests/*_test.py"
    )
  endif()

  file(GLOB_RECURSE tf_test_src_py
    ${tf_test_rnn_src_py}
    "${tensorflow_source_dir}/tensorflow/python/debug/cli/*_test.py"
    "${tensorflow_source_dir}/tensorflow/python/debug/lib/*_test.py"
    "${tensorflow_source_dir}/tensorflow/python/debug/wrappers/*_test.py"
    "${tensorflow_source_dir}/tensorflow/contrib/estimator/python/estimator/*_test.py"
    "${tensorflow_source_dir}/tensorflow/python/kernel_tests/*.py"
    "${tensorflow_source_dir}/tensorflow/python/meta_graph_transform/*_test.py"
    "${tensorflow_source_dir}/tensorflow/python/platform/build_info_test.py"
    "${tensorflow_source_dir}/tensorflow/python/profiler/*_test.py"
    "${tensorflow_source_dir}/tensorflow/python/profiler/internal/*_test.py"
    "${tensorflow_source_dir}/tensorflow/python/saved_model/*_test.py"
    "${tensorflow_source_dir}/tensorflow/python/training/*_test.py"
    "${tensorflow_source_dir}/tensorflow/contrib/data/*_test.py"
    "${tensorflow_source_dir}/tensorflow/contrib/factorization/*_test.py"
    "${tensorflow_source_dir}/tensorflow/contrib/image/*_test.py"
    "${tensorflow_source_dir}/tensorflow/python/keras/_impl/keras/*_test.py"
    "${tensorflow_source_dir}/tensorflow/contrib/periodic_resample/python/kernel_tests/*_test.py"
    "${tensorflow_source_dir}/tensorflow/contrib/nearest_neighbor/python/kernel_tests/*_test.py"
    "${tensorflow_source_dir}/tensorflow/contrib/seq2seq/python/kernel_tests/*_test.py"
    "${tensorflow_source_dir}/tensorflow/contrib/stateless/python/kernel_tests/*_test.py"
    # NOTE: tensor_forest tests in tensor_forest/hybrid/... still don't pass.
    "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/client/*_test.py"
    "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/python/*_test.py"
  )

  if (tensorflow_BUILD_MORE_PYTHON_TESTS)
    # Adding other major packages
    file(GLOB_RECURSE tf_test_src_py
      ${tf_test_src_py}
      "${tensorflow_source_dir}/tensorflow/contrib/legacy_seq2seq/*_test.py"
      "${tensorflow_source_dir}/tensorflow/contrib/linalg/*_test.py"
      "${tensorflow_source_dir}/tensorflow/contrib/graph_editor/*_test.py"
      "${tensorflow_source_dir}/tensorflow/contrib/bayesflow/*_test.py"
      "${tensorflow_source_dir}/tensorflow/contrib/framework/*_test.py"
      "${tensorflow_source_dir}/tensorflow/contrib/distributions/*_test.py"
      "${tensorflow_source_dir}/tensorflow/contrib/learn/*_test.py"
    )
  endif()

  # exclude the ones we don't want
  set(tf_test_src_py_exclude
    # Not a test.
    "${tensorflow_source_dir}/tensorflow/python/kernel_tests/__init__.py"
    # Flaky because of port collisions.
    "${tensorflow_source_dir}/tensorflow/python/training/localhost_cluster_performance_test.py"
    # generally not working
    "${tensorflow_source_dir}/tensorflow/python/profiler/pprof_profiler_test.py"
    # flaky test
    "${tensorflow_source_dir}/tensorflow/python/profiler/internal/run_metadata_test.py"
    # Fails because uses data dependencies with bazel
    "${tensorflow_source_dir}/tensorflow/python/saved_model/saved_model_test.py"
    # requires scipy
    "${tensorflow_source_dir}/tensorflow/contrib/keras/python/keras/preprocessing/*_test.py"
    "${tensorflow_source_dir}/tensorflow/contrib/tfprof/python/tools/tfprof/pprof_profiler_test.py"
    # Takes very long to run without sharding (defined in bazel build file).
    "${tensorflow_source_dir}/tensorflow/python/kernel_tests/cwise_ops_test.py"
    # Loading resources in contrib doesn't seem to work on Windows
    "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/client/random_forest_test.py"
    "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/python/tensor_forest_test.py"
    # dask need fix
    "${tensorflow_source_dir}/tensorflow/contrib/learn/python/learn/learn_io/generator_io_test.py"
    "${tensorflow_source_dir}/tensorflow/contrib/learn/python/learn/learn_io/graph_io_test.py"
    # Test is flaky on Windows GPU builds (b/38283730).
    "${tensorflow_source_dir}/tensorflow/contrib/factorization/python/ops/gmm_test.py"
  )
  if (WIN32)
    set(tf_test_src_py_exclude
      ${tf_test_src_py_exclude}
      # TODO: failing tests.
      # Nothing critical in here but should get this list down to []
      # The failing list is grouped by failure source
      # Python source line inspection tests are flaky on Windows (b/36375074).
      "${tensorflow_source_dir}/tensorflow/python/debug/cli/analyzer_cli_test.py"
      "${tensorflow_source_dir}/tensorflow/python/debug/cli/profile_analyzer_cli_test.py"
      # Windows does not have the curses library and uses readline.
      "${tensorflow_source_dir}/tensorflow/python/debug/cli/curses_ui_test.py"
      # TFDBG grpc:// mode is not yet available on Windows.
      "${tensorflow_source_dir}/tensorflow/python/debug/lib/dist_session_debug_grpc_test.py"
      "${tensorflow_source_dir}/tensorflow/python/debug/lib/session_debug_grpc_test.py"
      # stl on windows handles overflows different
      "${tensorflow_source_dir}/tensorflow/python/kernel_tests/as_string_op_test.py"
      "${tensorflow_source_dir}/tensorflow/python/kernel_tests/string_to_number_op_test.py"
      "${tensorflow_source_dir}/tensorflow/python/kernel_tests/clip_ops_test.py"
      "${tensorflow_source_dir}/tensorflow/python/kernel_tests/tensor_array_ops_test.py"  # Needs portpicker.
      # Numerical issues, calculations off.
      "${tensorflow_source_dir}/tensorflow/python/kernel_tests/concat_op_test.py"
      "${tensorflow_source_dir}/tensorflow/contrib/factorization/python/ops/wals_test.py"
      "${tensorflow_source_dir}/tensorflow/contrib/periodic_resample/python/kernel_tests/periodic_resample_op_test.py"
      "${tensorflow_source_dir}/tensorflow/python/keras/_impl/keras/utils/data_utils_test.py"
      # Float division by zero
      "${tensorflow_source_dir}/tensorflow/python/kernel_tests/benchmark_test.py"
      # Flaky, for unknown reasons. Cannot reproduce in terminal. Revisit once we can get stack traces.
      "${tensorflow_source_dir}/tensorflow/python/kernel_tests/batch_matmul_op_test.py"
      # Flaky because of local cluster creation.
      "${tensorflow_source_dir}/tensorflow/python/training/sync_replicas_optimizer_test.py"
      "${tensorflow_source_dir}/tensorflow/python/debug/lib/session_debug_grpc_test.py"
      "${tensorflow_source_dir}tensorflow/python/training/localhost_cluster_performance_test.py"
      "${tensorflow_source_dir}/tensorflow/python/kernel_tests/iterator_ops_cluster_test.py"
      "${tensorflow_source_dir}/tensorflow/python/kernel_tests/functional_ops_test.py"
      "${tensorflow_source_dir}/tensorflow/contrib/data/python/kernel_tests/iterator_ops_cluster_test.py"
      # Type error in testRemoteIteratorUsingRemoteCallOpDirectSessionGPUCPU.
      "${tensorflow_source_dir}/tensorflow/python/kernel_tests/iterator_ops_test.py"
      "${tensorflow_source_dir}/tensorflow/python/kernel_tests/self_adjoint_eig_op_test.py"
      "${tensorflow_source_dir}/tensorflow/contrib/data/python/kernel_tests/iterator_ops_test.py"
      # IteratorGetMax OutOfRangeError
      "${tensorflow_source_dir}/tensorflow/contrib/data/python/kernel_tests/batch_dataset_op_test.py"
      # Depends on gemmlowp -> pthread
      "${tensorflow_source_dir}/tensorflow/python/kernel_tests/neon_depthwise_conv_op_test.py"
      # int32/int64 mixup
      "${tensorflow_source_dir}/tensorflow/python/kernel_tests/cast_op_test.py"
      "${tensorflow_source_dir}/tensorflow/python/kernel_tests/variable_scope_test.py"
      "${tensorflow_source_dir}/tensorflow/python/kernel_tests/functional_ops_test.py"
      "${tensorflow_source_dir}/tensorflow/python/kernel_tests/py_func_test.py"
      # Windows file management related issues.
      "${tensorflow_source_dir}/tensorflow/python/training/evaluation_test.py"
      # training tests
      "${tensorflow_source_dir}/tensorflow/python/training/basic_session_run_hooks_test.py"  # Needs tf.contrib fix.
      "${tensorflow_source_dir}/tensorflow/python/training/quantize_training_test.py"  # Needs quantization ops to be included in windows.
      "${tensorflow_source_dir}/tensorflow/python/training/supervisor_test.py"  # Flaky I/O error on rename.
      "${tensorflow_source_dir}/tensorflow/python/training/server_lib_test.py"  # Test occasionally deadlocks.
      "${tensorflow_source_dir}/tensorflow/python/debug/lib/session_debug_multi_gpu_test.py"  # Fails on multiple GPUs.
      "${tensorflow_source_dir}/tensorflow/python/kernel_tests/concat_op_test.py"  # numerical issues
      "${tensorflow_source_dir}/tensorflow/python/kernel_tests/linalg_grad_test.py"  # cudaSolver handle creation fails.
      "${tensorflow_source_dir}/tensorflow/python/kernel_tests/array_ops_test.py"  # depends on python/framework/test_ops
      # Dataset tests
      "${tensorflow_source_dir}/tensorflow/python/kernel_tests/dataset_constructor_op_test.py"  # Segfaults on windows
      "${tensorflow_source_dir}/tensorflow/contrib/data/python/kernel_tests/dataset_constructor_op_test.py"  # Segfaults on Windows.
      "${tensorflow_source_dir}/tensorflow/python/kernel_tests/iterator_ops_cluster_test.py"
      # Broken tensorboard test due to cmake issues.
      "${tensorflow_source_dir}/tensorflow/contrib/data/python/kernel_tests/iterator_ops_cluster_test.py"  # Needs portpicker
      "${tensorflow_source_dir}/tensorflow/contrib/data/python/kernel_tests/sloppy_transformation_dataset_op_test.py"  # b/65430561
      # tensor_forest tests (also note that we exclude the hybrid tests for now)
      "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/python/kernel_tests/count_extremely_random_stats_op_test.py"  # Results in wrong order.
      "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/python/kernel_tests/sample_inputs_op_test.py"  # Results in wrong order.
      "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/python/kernel_tests/scatter_add_ndim_op_test.py"  # Bad placement.
      "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/python/topn_test.py"  # Results inaccurate
      "${tensorflow_source_dir}/tensorflow/python/ops/cloud/bigquery_reader_ops_test.py"  # No libcurl support
      # Dask.Dataframe bugs on Window Build
      "${tensorflow_source_dir}/tensorflow/contrib/learn/python/learn/tests/dataframe/tensorflow_dataframe_test.py"
      "${tensorflow_source_dir}/tensorflow/contrib/learn/python/learn/learn_io/data_feeder_test.py"
      "${tensorflow_source_dir}/tensorflow/contrib/learn/python/learn/learn_io/io_test.py"
      "${tensorflow_source_dir}/tensorflow/contrib/learn/python/learn/graph_actions_test.py"
      # Need extra build
      "${tensorflow_source_dir}/tensorflow/contrib/distributions/python/kernel_tests/conditional_distribution_test.py"
      "${tensorflow_source_dir}/tensorflow/contrib/distributions/python/kernel_tests/conditional_transformed_distribution_test.py"
      "${tensorflow_source_dir}/tensorflow/contrib/distributions/python/kernel_tests/estimator_test.py"
      "${tensorflow_source_dir}/tensorflow/python/kernel_tests/array_ops_test.py"  # depends on python/framework/test_ops
      "${tensorflow_source_dir}/tensorflow/python/kernel_tests/depthtospace_op_test.py"  # QuantizeV2
      "${tensorflow_source_dir}/tensorflow/python/kernel_tests/spacetodepth_op_test.py"  # QuantizeV2
      # Windows Path
      "${tensorflow_source_dir}/tensorflow/contrib/framework/python/ops/checkpoint_ops_test.py" #TODO: Fix path
      "${tensorflow_source_dir}/tensorflow/contrib/factorization/python/ops/kmeans_test.py"
      "${tensorflow_source_dir}/tensorflow/contrib/learn/python/learn/estimators/kmeans_test.py"
      # Numpy upgrade needed?
      "${tensorflow_source_dir}/tensorflow/contrib/distributions/python/kernel_tests/bijectors/sinh_arcsinh_test.py"
      # Test should only be run manually
      "${tensorflow_source_dir}/tensorflow/python/kernel_tests/reduction_ops_test_big.py"
      "${tensorflow_source_dir}/tensorflow/python/kernel_tests/svd_op_test.py"
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
    "${tensorflow_source_dir}/tensorflow/contrib/nearest_neighbor/*_test.cc"
    "${tensorflow_source_dir}/tensorflow/contrib/rnn/*_test.cc"
  )

  # exclude the ones we don't want
  set(tf_test_src_simple_exclude
    # generally not working
    "${tensorflow_source_dir}/tensorflow/cc/client/client_session_test.cc"
    "${tensorflow_source_dir}/tensorflow/cc/framework/gradients_test.cc"
    "${tensorflow_source_dir}/tensorflow/core/distributed_runtime/call_options_test.cc"
    "${tensorflow_source_dir}/tensorflow/core/distributed_runtime/tensor_coding_test.cc"
    "${tensorflow_source_dir}/tensorflow/core/kernels/remote_fused_graph_rewriter_transform_test.cc"
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

  file(GLOB tf_core_profiler_test_srcs
    "${tensorflow_source_dir}/tensorflow/core/profiler/internal/*_test.cc"
    "${tensorflow_source_dir}/tensorflow/core/profiler/internal/advisor/*_test.cc"
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

  file(GLOB_RECURSE tf_core_profiler_test_data
    "${tensorflow_source_dir}/tensorflow/core/profiler/testdata/*"
  )

  AddTests(
    SOURCES ${tf_core_profiler_test_srcs}
    DATA ${tf_core_profiler_test_data}
    OBJECTS ${tf_obj_test}
    LIBS ${tf_test_libs}
  )

endif(tensorflow_BUILD_CC_TESTS)
