# Description:
#  Tools for building the TensorFlow pip package.

load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")
load("@local_config_syslibs//:build_defs.bzl", "if_not_system_lib")
load(
    "@local_tsl//third_party/py:py_import.bzl",
    "py_import",
)
load(
    "@local_tsl//third_party/py:py_manylinux_compliance_test.bzl",
    "verify_manylinux_compliance_test",
)
load("@local_xla//xla/tsl/mkl:build_defs.bzl", "if_enable_mkl", "if_mkl", "if_mkl_ml")
load("//tensorflow:tensorflow.bzl", "if_wheel_dependency", "if_with_tpu_support", "transitive_hdrs")
load("//tensorflow:tf_version.bzl", "TF_SEMANTIC_VERSION_SUFFIX", "TF_VERSION")
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "if_pywrap",
    "tf_additional_license_deps",
    "tf_cuda_tests_tags",
    "tf_exec_properties",
)
load("//tensorflow/tools/pip_package/utils:data_deps.bzl", "collect_data_files")
load("//tensorflow/tools/pip_package/utils:py_deps.bzl", "transitive_py_deps")
load("//tensorflow/tools/pip_package/utils:tf_wheel.bzl", "tf_wheel", "tf_wheel_dep")

package(default_visibility = ["//visibility:public"])

# This returns a list of headers of all public header libraries (e.g.,
# framework, lib), and all of the transitive dependencies of those
# public headers.  Not all of the headers returned by the filegroup
# are public (e.g., internal headers that are included by public
# headers), but the internal headers need to be packaged in the
# pip_package for the public headers to be properly included.
#
# Public headers are therefore defined by those that are both:
#
# 1) "publicly visible" as defined by bazel
# 2) Have documentation.
#
# This matches the policy of "public" for our python API.
transitive_hdrs(
    name = "included_headers",
    deps = [
        "//tensorflow/compiler/tf2xla:xla_compiled_cpu_function",
        "//tensorflow/compiler/mlir:mlir_graph_optimization_pass",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:stream_executor",
        "//tensorflow/cc/saved_model:loader",
        "//tensorflow/cc/saved_model:reader",
        "//tensorflow/cc/saved_model:bundle_v2",
        "//tensorflow/c/experimental/filesystem:filesystem_interface",
        "//tensorflow/c/experimental/next_pluggable_device:c_api_hdrs",
        "//tensorflow/c/experimental/pluggable_profiler:pluggable_profiler_hdrs",
        "//tensorflow/c/experimental/stream_executor:stream_executor_hdrs",
        "//tensorflow/c/experimental/grappler:grappler_hdrs",
        "//tensorflow/c:kernels_hdrs",
        "//tensorflow/c:ops_hdrs",
        "@local_xla//xla/pjrt/c:pjrt_c_api_hdrs",
        "//tensorflow/core/common_runtime/next_pluggable_device/c:plugin_c_api_hdrs",
        # TODO(rostam): Revisit these after cc_shared_library everywhere
        "//tensorflow/cc:client_session",
        "//tensorflow/cc:cc_ops",
        # WARNING: None of the C/C++ code under python/ has any API guarantees, and TF team
        # reserves the right to change APIs and other header-level interfaces.  If your custom
        # op uses these headers, it may break when users upgrade their version of tensorflow.
        # NOTE(ebrevdo): None of the python symbols are exported by libtensorflow_framework.so.
        # Code that relies on these headers should dynamically link to
        # _pywrap_tensorflow_internal.so as well.
        "//tensorflow/python/grappler:model_analyzer_lib",
        "//tensorflow/python/lib/core:py_exception_registry",
        "//tensorflow/python/lib/core:pybind11_proto",
        "//tensorflow/python/lib/core:pybind11_absl",
        "//tensorflow/python/lib/core:pybind11_lib",
        "//tensorflow/python/lib/core:pybind11_status",
        "//tensorflow/python/lib/core:py_func_lib",
        "//tensorflow/python/lib/core:py_seq_tensor",
        "//tensorflow/python/lib/core:py_util",
        "//tensorflow/python/util:cpp_python_util",
        "//tensorflow/python/util:kernel_registry",
        "//tensorflow/python/framework:python_op_gen",
        "//tensorflow/python/client:tf_session_helper",
    ] + if_cuda([
        "@local_config_cuda//cuda:cuda_headers",
    ]),
)

filegroup(
    name = "xla_cmake",
    srcs = ["xla_build/CMakeLists.txt"],
)

filegroup(
    name = "licenses",
    srcs = [
        "//:LICENSE",
        "//third_party/icu/data:LICENSE",
        "@XNNPACK//:LICENSE",
        "@com_google_absl//:LICENSE",
        "@com_google_protobuf//:LICENSE",
        "@curl//:COPYING",
        "@ducc//:LICENSE",
        "@flatbuffers//:LICENSE",
        "@gemmlowp//:LICENSE",
        "@libjpeg_turbo//:LICENSE.md",
        "@llvm-project//llvm:LICENSE.TXT",
        "@llvm-project//mlir:LICENSE.TXT",
        "@local_config_tensorrt//:LICENSE",
        "@ml_dtypes//:LICENSE",
        "@org_brotli//:LICENSE",
        "@pasta//:LICENSE",
        "@png//:LICENSE",
        "@ruy//:LICENSE",
        "@six_archive//:LICENSE",
        "@snappy//:COPYING",
        "@stablehlo//:LICENSE",
        "@tf_runtime//:LICENSE",
    ] + select({
        "//tensorflow:android": [],
        "//tensorflow:ios": [],
        "//tensorflow:linux_s390x": [],
        "//tensorflow:windows": [],
        "//conditions:default": [
            "@com_github_googlecloudplatform_google_cloud_cpp//:LICENSE",
        ],
    }) + select({
        "@local_xla//xla/tsl:with_numa_support": [
            "@hwloc//:COPYING",
        ],
        "//conditions:default": [],
    }) + if_cuda([
        "@cub_archive//:LICENSE.TXT",
        "@local_config_nccl//:LICENSE",
    ]) + if_mkl([
        "//third_party/mkl_dnn:LICENSE",
    ]) + if_enable_mkl(["@local_xla//xla/tsl/mkl:LICENSE"]) + if_not_system_lib(
        "absl_py",
        [
            "@absl_py//:LICENSE",
        ],
    ) + tf_additional_license_deps(),
)

# Add dynamic kernel dso files here.
DYNAMIC_LOADED_KERNELS = [
    "//tensorflow/core/kernels:libtfkernel_sobol_op.so",
]

COMMON_PIP_DEPS = [
    "//tensorflow/compiler/mlir/stablehlo:stablehlo",
    "//tensorflow/compiler/mlir/tensorflow:gen_mlir_passthrough_op_py",
    "//tensorflow/core/function/trace_type:serialization_test_proto_py",
    "//tensorflow/core/function/trace_type:serialization",
    "//tensorflow/dtensor/python/tests:multi_client_test_util",
    "//tensorflow/dtensor/python/tests:test_util",
    "//tensorflow/lite/python:tflite_convert",
    "//tensorflow/lite/tools:visualize",
    "//tensorflow/python/autograph/core:test_lib",
    "//tensorflow/python/autograph/impl/testing:pybind_for_testing",
    "//tensorflow/python/autograph/pyct/testing:basic_definitions",
    "//tensorflow/python/autograph/pyct/testing:decorators",
    "//tensorflow/python/autograph/pyct/common_transformers:common_transformers",
    "//tensorflow/python/autograph/pyct/static_analysis:type_inference",
    "//tensorflow/python/autograph/utils:context_managers",
    "//tensorflow/python/autograph/utils:tensor_list",
    "//tensorflow/python/compiler:compiler",
    "//tensorflow/python/ops:cond_v2",
    "//tensorflow/python:distributed_framework_test_lib",
    "//tensorflow/python/distribute:distribute_test_lib_pip",
    "//tensorflow/python/training/experimental:loss_scale",
    "//tensorflow/python/training/experimental:loss_scale_optimizer",
    "//tensorflow/python/data/benchmarks:benchmark_base",
    "//tensorflow/python/data/experimental/kernel_tests/service:multi_process_cluster",
    "//tensorflow/python/data/experimental/kernel_tests/service:test_base",
    "//tensorflow/python/data/experimental/ops:testing",
    "//tensorflow/python/data/experimental/service:server_lib",
    "//tensorflow/python/data/kernel_tests:checkpoint_test_base",
    "//tensorflow/python/data/kernel_tests:test_base",
    "//tensorflow/python/data/kernel_tests:tf_record_test_base",
    "//tensorflow/python/ops/ragged:ragged_tensor_test_ops",
    "//tensorflow/python/debug:debug_pip",
    "//tensorflow/python/distribute:combinations",
    "//tensorflow/python/distribute/coordinator:fault_tolerance_test_base",
    "//tensorflow/python/distribute/failure_handling:check_preemption_py",
    "//tensorflow/python/distribute/failure_handling:failure_handling_lib",
    "//tensorflow/python/distribute/failure_handling:failure_handling_util",
    "//tensorflow/python/distribute:multi_process_runner",
    "//tensorflow/python/eager:eager_pip",
    "//tensorflow/python/keras:combinations",
    "//tensorflow/python/keras/mixed_precision:test_util",
    "//tensorflow/python/keras/utils:dataset_creator",
    "//tensorflow/python/kernel_tests/nn_ops:bias_op_base",
    "//tensorflow/python/kernel_tests/nn_ops:cudnn_deterministic_base",
    "//tensorflow/python/kernel_tests/nn_ops:depthwise_conv_op_base",
    "//tensorflow/python/kernel_tests/nn_ops:xent_op_test_base",
    "//tensorflow/python/kernel_tests/random:util",
    "//tensorflow/python/kernel_tests/signal:test_util",
    "//tensorflow/python/kernel_tests/sparse_ops:sparse_xent_op_test_base",
    "//tensorflow/python/lib:__init__",
    "//tensorflow/python/ops/parallel_for:test_util",
    "//tensorflow/python/ops/structured:structured_tensor_dynamic",
    "//tensorflow/python/platform:resource_loader",
    "//tensorflow/python/profiler:trace",
    "//tensorflow/python/saved_model:saved_model",
    "//tensorflow/python/summary:__init__",
    "//tensorflow/python/summary:plugin_asset",
    "//tensorflow/python/summary:summary_iterator",
    "//tensorflow/python/summary:summary_py",
    "//tensorflow/python/tools:tools_pip",
    "//tensorflow/python/tools/api/generator:create_python_api",
    "//tensorflow/python/tools/api/generator2/extractor:extractor",
    "//tensorflow/python/tools/api/generator2/generator:generator",
    "//tensorflow/python/tpu",
    "//tensorflow/python/util:deprecated_module",
    "//tensorflow/python/util:deprecated_module_new",
    "//tensorflow/python/util:example_parser_configuration",
    "//tensorflow/python/util:function_utils",
    "//tensorflow/python/util:keyword_args",
    "//tensorflow/python/util:lock_util",
    "//tensorflow/python/util:module_wrapper",
    "//tensorflow/python/util:serialization",
    "//tensorflow/python/ops:image_grad_test_base",
    "//tensorflow/python/framework:memory_checker",
    "//tensorflow/python/framework:test_ops",
    "//tensorflow/python/ops:while_v2",
    "//tensorflow/tools/common:public_api",
    "//tensorflow/tools/common:test_module1",
    "//tensorflow/tools/common:traverse",
    "//tensorflow/tools/docs:tf_doctest_lib",
    "@local_tsl//tsl/profiler/protobuf:trace_events_proto_py",
    "//tensorflow/python/distribute:parameter_server_strategy_v2",
    "//tensorflow/python/distribute/coordinator:cluster_coordinator",
    "//tensorflow/python/distribute/coordinator:remote_eager_lib",
    "//tensorflow/python/distribute/coordinator:metric_utils",
    "//tensorflow/python/distribute/experimental/rpc:rpc_ops",
    "//tensorflow/python/util:pywrap_xla_ops",
    "//tensorflow:tensorflow_py",
    "//tensorflow/tools/compatibility:tf_upgrade_v2",
] + if_with_tpu_support([
    "//tensorflow/python/tools:grpc_tpu_worker",
    "//tensorflow/python/tools:grpc_tpu_worker_service",
])

py_binary(
    name = "build_pip_package_py",
    srcs = [
        "build_pip_package.py",
    ],
    main = "build_pip_package.py",
    deps = [
        "//tensorflow/tools/pip_package/utils:py_utils",
        "@pypi_setuptools//:pkg",
        "@pypi_wheel//:pkg",
    ],
)

collect_data_files(
    name = "cc_deps",
    deps = [
        #`tensorflow_py` is the main target to collect all required `data` dependencies
        "//tensorflow:tensorflow_py",
        # utils below are not part of tensorflow_py DAG
        "//tensorflow/compiler/mlir/stablehlo:stablehlo_extension",
        "//tensorflow/python/autograph/impl/testing:pybind_for_testing",
        "//tensorflow/python/framework:memory_checker",
        "//tensorflow/python/util:pywrap_xla_ops",
    ],
)

transitive_py_deps(
    name = "py_deps",
    deps = COMMON_PIP_DEPS,
)

genrule(
    name = "setup_py",
    srcs = ["setup.py.tpl"],
    outs = ["setup.py"],
    cmd = """sed -E "s/_VERSION = '0.0.0'/_VERSION = '{wheel_version}{wheel_version_suffix}'/" \
$(location setup.py.tpl) > $@;""".format(
        wheel_version = TF_VERSION,
        wheel_version_suffix = TF_SEMANTIC_VERSION_SUFFIX,
    ),
)

py_binary(
    name = "setup_py_binary",
    srcs = [":setup_py"],
    data = [
        "//:AUTHORS",
        "//:LICENSE",
    ],
    main = "setup.py",
)

tf_wheel(
    name = "wheel",
    headers = [
        ":included_headers",
        ":licenses",
        "//tensorflow/core:protos_all_proto_srcs",
    ],
    platform_name = select({
        "@platforms//os:osx": "macosx",
        "@platforms//os:macos": "macosx",
        "@platforms//os:windows": "win",
        "@platforms//os:linux": "linux",
    }),
    platform_tag = select({
        "@platforms//cpu:aarch64": "arm64",
        "@platforms//cpu:arm64": "arm64",
        "@platforms//cpu:x86_64": "x86_64",
        "@platforms//cpu:ppc": "ppc64le",
    }),
    source_files = [
        "MANIFEST.in",
        "//tensorflow/tools/pip_package:THIRD_PARTY_NOTICES.txt",
        ":setup_py",
        ":cc_deps",
        ":py_deps",
        "//tensorflow:tf_python_api_gen_v2",
    ] + select({
        "//tensorflow:dynamic_loaded_kernels": if_pywrap(
            if_false = DYNAMIC_LOADED_KERNELS,
            if_true = [],
        ),
        "//conditions:default": [],
    }) + select({
        "//conditions:default": if_pywrap(
            if_false = [
                "//tensorflow:tensorflow_cc",
                "//tensorflow:tensorflow_framework",
            ],
            if_true = [
                "//tensorflow/python:pywrap_tensorflow_binaries",
            ],
        ),
        "//tensorflow:windows": if_pywrap(
            if_false = [
                "//tensorflow/python:pywrap_tensorflow_import_lib_file",
            ],
            if_true = [],
        ),
    }) + if_mkl_ml(["@local_xla//xla/tsl/mkl:intel_binary_blob"]),
    xla_aot_compiled = [
        "//tensorflow/compiler/tf2xla:xla_compiled_cpu_runtime_srcs",
        ":xla_cmake",
    ],
)

genrule(
    name = "empty_test",
    outs = ["empty_test.py"],
    cmd = "echo '' > $@",
)

py_test(
    name = "prebuilt_wheel_import_api_packages_test_cpu",
    srcs = if_wheel_dependency(
        ["import_api_packages_test.py"],
        [":empty_test"],
    ),
    main = if_wheel_dependency("import_api_packages_test.py", "empty_test.py"),
    tags = [
        "cpu",
        "manual",
        "windows_excluded",
    ],
    deps = if_wheel_dependency(tf_wheel_dep()),
)

py_test(
    name = "prebuilt_wheel_import_api_packages_test_gpu",
    srcs = if_wheel_dependency(
        ["import_api_packages_test.py"],
        [":empty_test"],
    ),
    exec_properties = if_cuda(
        tf_exec_properties({"tags": tf_cuda_tests_tags()}),
        {},
    ),
    main = if_wheel_dependency("import_api_packages_test.py", "empty_test.py"),
    tags = [
        "gpu",
        "manual",
        "windows_excluded",
    ],
    deps = if_wheel_dependency(tf_wheel_dep()),
)

py_test(
    name = "import_api_packages_test_cpu",
    srcs = ["import_api_packages_test.py"],
    main = "import_api_packages_test.py",
    tags = [
        "cpu",
        "manual",
        "windows_excluded",
    ],
    deps = [
        ":tf_py_import",
    ],
)

py_test(
    name = "import_api_packages_test_gpu",
    srcs = ["import_api_packages_test.py"],
    exec_properties = if_cuda(
        tf_exec_properties({"tags": tf_cuda_tests_tags()}),
        {},
    ),
    main = "import_api_packages_test.py",
    tags = [
        "gpu",
        "manual",
        "windows_excluded",
    ],
    deps = [
        ":tf_py_import",
    ],
)

verify_manylinux_compliance_test(
    name = "manylinux_compliance_test",
    aarch64_compliance_tag = "manylinux_2_17_aarch64",
    test_tags = [
        "mac_excluded",
        "windows_excluded",
    ],
    wheel = ":wheel",
    x86_64_compliance_tag = "manylinux_2_17_x86_64",
)

py_import(
    name = "tf_py_import",
    wheel = ":wheel",
    wheel_deps = if_cuda([
        "@pypi_nvidia_cublas_cu12//:whl",
        "@pypi_nvidia_cuda_cupti_cu12//:whl",
        "@pypi_nvidia_cuda_nvrtc_cu12//:whl",
        "@pypi_nvidia_cuda_runtime_cu12//:whl",
        "@pypi_nvidia_cudnn_cu12//:whl",
        "@pypi_nvidia_cufft_cu12//:whl",
        "@pypi_nvidia_curand_cu12//:whl",
        "@pypi_nvidia_cusolver_cu12//:whl",
        "@pypi_nvidia_cusparse_cu12//:whl",
        "@pypi_nvidia_nccl_cu12//:whl",
        "@pypi_nvidia_nvjitlink_cu12//:whl",
    ]),
    deps = [
        "@pypi_absl_py//:pkg",
        "@pypi_astunparse//:pkg",
        "@pypi_flatbuffers//:pkg",
        "@pypi_gast//:pkg",
        "@pypi_ml_dtypes//:pkg",
        "@pypi_numpy//:pkg",
        "@pypi_opt_einsum//:pkg",
        "@pypi_packaging//:pkg",
        "@pypi_protobuf//:pkg",
        "@pypi_requests//:pkg",
        "@pypi_termcolor//:pkg",
        "@pypi_typing_extensions//:pkg",
        "@pypi_wrapt//:pkg",
    ],
)
