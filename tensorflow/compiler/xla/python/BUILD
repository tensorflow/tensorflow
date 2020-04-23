load("//tensorflow/core/platform:build_config.bzl", "pyx_library")
load("//tensorflow/compiler/xla:xla.bzl", "xla_py_test_deps")
load("//tensorflow:tensorflow.bzl", "py_test", "tf_cc_test")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "pybind_extension")

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],  # Apache 2.0
)

py_library(
    name = "xla_client",
    srcs = ["xla_client.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [":xla_extension"],
)

pyx_library(
    name = "custom_call_for_test",
    testonly = True,
    srcs = ["custom_call_for_test.pyx"],
)

py_library(
    name = "xla_client_test",
    testonly = 1,
    srcs = ["xla_client_test.py"],
    srcs_version = "PY3",
    deps = [
        ":custom_call_for_test",
        ":xla_client",
        ":xla_extension",
        "@absl_py//absl/flags",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "xla_client_test_cpu",
    srcs = ["xla_client_test.py"],
    args = ["--backend=cpu"],
    main = "xla_client_test.py",
    python_version = "PY3",
    srcs_version = "PY3",
    tags = ["no_oss"],  # TODO(phawkins): This test passes, but requires --config=monolithic.
    deps = [
        ":custom_call_for_test",
        ":xla_client",
        ":xla_extension",
        "@absl_py//absl/flags",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ] + xla_py_test_deps(),
)

py_test(
    name = "xla_client_test_gpu",
    srcs = ["xla_client_test.py"],
    args = ["--backend=gpu"],
    main = "xla_client_test.py",
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_oss",
        "requires-gpu-nvidia",
    ],  # TODO(phawkins): This test passes, but requires --config=monolithic.
    deps = [
        ":xla_client",
        ":xla_extension",
        "@absl_py//absl/flags",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ] + xla_py_test_deps(),
)

cc_library(
    name = "worker_thread",
    srcs = ["worker_thread.cc"],
    hdrs = ["worker_thread.h"],
    deps = [
        "//tensorflow/core:lib",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "types",
    srcs = ["types.cc"],
    hdrs = ["types.h"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":bfloat16",
        ":local_client",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core:lib",
        "//third_party/py/numpy:headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/types:optional",
        "@pybind11",
    ],
)

cc_library(
    name = "event_pool",
    srcs = ["event_pool.cc"],
    hdrs = ["event_pool.h"],
    deps = [
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/core:lib",
        "//tensorflow/core:stream_executor",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "semaphore",
    srcs = ["semaphore.cc"],
    hdrs = ["semaphore.h"],
    deps = [
        "//tensorflow/compiler/xla:types",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/synchronization",
    ],
)

tf_cc_test(
    name = "semaphore_test",
    srcs = ["semaphore_test.cc"],
    deps = [
        ":semaphore",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/core:lib",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "shared_device_buffer",
    srcs = ["shared_device_buffer.cc"],
    hdrs = ["shared_device_buffer.h"],
    deps = [
        ":event_pool",
        ":local_device_state",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla/service:shaped_buffer",
        "//tensorflow/compiler/xla/service:transfer_manager",
        "//tensorflow/core:lib",
        "//tensorflow/stream_executor:device_memory",
        "//tensorflow/stream_executor:device_memory_allocator",
        "//tensorflow/stream_executor:event",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/synchronization",
    ],
)

tf_cc_test(
    name = "shared_device_buffer_test",
    srcs = ["shared_device_buffer_test.cc"],
    deps = [
        ":shared_device_buffer",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla/client:client_library",
        "//tensorflow/compiler/xla/service:cpu_plugin",
        "//tensorflow/core:test_main",
        "//tensorflow/stream_executor:device_memory",
        "//tensorflow/stream_executor:device_memory_allocator",
    ],
)

cc_library(
    name = "local_device_state",
    srcs = ["local_device_state.cc"],
    hdrs = ["local_device_state.h"],
    deps = [
        ":event_pool",
        ":semaphore",
        ":worker_thread",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/core:lib",
        "//tensorflow/core:stream_executor",
        "//tensorflow/stream_executor:event",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "local_client",
    srcs = ["local_client.cc"],
    hdrs = ["local_client.h"],
    visibility = ["//tensorflow/compiler/xla:friends"],
    deps = [
        ":event_pool",
        ":local_device_state",
        ":shared_device_buffer",
        "//tensorflow/compiler/xla:executable_run_options",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:executable_build_options",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/python/distributed:protocol_proto_cc",
        "//tensorflow/compiler/xla/service:computation_placer",
        "//tensorflow/compiler/xla/service:executable",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:maybe_owning_device_memory",
        "//tensorflow/compiler/xla/service:shaped_buffer",
        "//tensorflow/compiler/xla/service/gpu:gpu_executable_run_options",
        "//tensorflow/core:allocator",
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/stream_executor:event",
        "//tensorflow/stream_executor:stream",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "python_ref_manager",
    srcs = ["python_ref_manager.cc"],
    hdrs = ["python_ref_manager.h"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@pybind11",
    ],
)

cc_library(
    name = "bfloat16",
    srcs = ["bfloat16.cc"],
    hdrs = ["bfloat16.h"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core/lib/bfloat16",
        "//tensorflow/core/platform:logging",
        "//third_party/py/numpy:headers",
        "//third_party/python_runtime:headers",  # buildcleaner: keep
        "@com_google_absl//absl/strings",
        "@pybind11",
    ],
)

py_test(
    name = "bfloat16_test",
    srcs = ["bfloat16_test.py"],
    main = "bfloat16_test.py",
    python_version = "PY3",
    tags = ["no_oss"],
    deps = [
        ":xla_client",
        ":xla_extension",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ] + xla_py_test_deps(),
)

cc_library(
    name = "dlpack",
    srcs = ["dlpack.cc"],
    hdrs = ["dlpack.h"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":local_client",
        ":shared_device_buffer",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/stream_executor:device_memory",
        "//tensorflow/stream_executor:platform",
        "//tensorflow/stream_executor/cuda:cuda_platform_id",
        "//tensorflow/stream_executor/host:host_platform_id",
        "//third_party/python_runtime:headers",  # buildcleaner: keep
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@dlpack",
        "@pybind11",
    ],
)

cc_library(
    name = "cpu_device",
    srcs = ["cpu_device.cc"],
    hdrs = ["cpu_device.h"],
    deps = [
        ":local_client",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/client:client_library",
        "//tensorflow/compiler/xla/service:platform_util",
    ],
)

cc_library(
    name = "nvidia_gpu_device",
    srcs = ["nvidia_gpu_device.cc"],
    hdrs = ["nvidia_gpu_device.h"],
    copts = if_cuda(["-DNCCL_ENABLED=1"]),
    deps = [
        ":local_client",
        "//tensorflow/compiler/xla/service/gpu:gpu_executable_run_options",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/client:client_library",
        "//tensorflow/compiler/xla/python/distributed:client",
        "//tensorflow/compiler/xla/service:platform_util",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core/common_runtime:bfc_allocator",
        "//tensorflow/core/common_runtime/gpu:gpu_mem_allocator",
        "//tensorflow/stream_executor:tf_allocator_adapter",
    ] + if_cuda(["@local_config_nccl//:nccl"]),
)

config_setting(
    name = "enable_gpu",
    values = {"define": "xla_python_enable_gpu=true"},
)

pybind_extension(
    name = "xla_extension",
    srcs = [
        "xla.cc",
    ],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    module_name = "xla_extension",
    deps = [
        ":bfloat16",
        ":cpu_device",
        ":dlpack",
        ":local_client",
        ":nvidia_gpu_device",
        ":shared_device_buffer",
        ":python_ref_manager",
        ":types",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "//third_party/py/numpy:headers",
        "@pybind11",
        "//third_party/python_runtime:headers",  # buildcleaner: keep
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:client_library",
        "//tensorflow/compiler/xla/client:executable_build_options",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/client/lib:comparators",
        "//tensorflow/compiler/xla/client/lib:math",
        "//tensorflow/compiler/xla/client/lib:qr",
        "//tensorflow/compiler/xla/client/lib:self_adjoint_eig",
        "//tensorflow/compiler/xla/client/lib:sorting",
        "//tensorflow/compiler/xla/client/lib:svd",
        "//tensorflow/compiler/xla/python/distributed",
        "//tensorflow/compiler/xla/python/distributed:client",
        "//tensorflow/compiler/xla/python/distributed:service",
        "//tensorflow/compiler/xla/service:computation_placer",
        "//tensorflow/compiler/xla/service:custom_call_target_registry",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/service:hlo_graph_dumper",
        "//tensorflow/compiler/xla/service:name_uniquer",
        "//tensorflow/compiler/xla/service:platform_util",
        "//tensorflow/compiler/xla/service:shaped_buffer",
        "//tensorflow/compiler/xla/service:transfer_manager",
        "//tensorflow/compiler/xla/service:cpu_plugin",
        "//tensorflow/core:lib",
        # Do NOT remove this dependency. The XLA Python extension must not
        # depend on any part of TensorFlow at runtime, **including**
        # libtensorflow_framework.so. The XLA module is deployed self-contained
        # without any TF dependencies as "jaxlib" on Pypi, and "jaxlib" does
        # not require Tensorflow.
        "//tensorflow/core:lib_internal_impl",  # buildcleaner: keep
        "//tensorflow/core/profiler/lib:profiler_backends",
        "//tensorflow/core/profiler/lib:profiler_session",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/core/profiler/rpc:profiler_server",
        "//tensorflow/stream_executor:device_memory_allocator",
        "//tensorflow/stream_executor:platform",
    ] + select({
        ":enable_gpu": ["//tensorflow/compiler/xla/service:gpu_plugin"],
        "//conditions:default": [],
    }),
)

tf_cc_test(
    name = "gpu_multistream_test",
    srcs = ["gpu_multistream_test.cc"],
    tags = [
        # TODO(phawkins): figure out TF test infra such that this only runs under GPU.
        "no_oss",
        "requires-gpu-nvidia",
    ],
    deps = [
        ":local_client",
        ":nvidia_gpu_device",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla/client:executable_build_options",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/service:gpu_plugin",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/core:lib",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:random",
    ],
)
