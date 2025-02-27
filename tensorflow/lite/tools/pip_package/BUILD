load("//tensorflow/lite/tools/pip_package/utils:py_manylinux_compliance_test.bzl", "verify_manylinux_compliance_test")
load("//tensorflow/lite/tools/pip_package/utils:py_wheel.bzl", "py_wheel")

package(
    default_visibility = ["//visibility:private"],
)

MANYLINUX_X86_64_TAG = "manylinux_2_17_x86_64"

genrule(
    name = "setup_py",
    srcs = ["//tensorflow/lite/tools/pip_package:setup_with_binary.py"],
    outs = ["setup.py"],
    cmd = "cat $< > $@",
)

py_wheel(
    name = "litert_wheel",
    srcs = [
        "//tensorflow/lite/experimental/genai:pywrap_genai_ops.so",
        "//tensorflow/lite/profiling/proto:model_runtime_info_py",
        "//tensorflow/lite/profiling/proto:profiling_info_py",
        "//tensorflow/lite/python:interpreter",
        "//tensorflow/lite/python:schema_py",
        "//tensorflow/lite/python/interpreter_wrapper:_pywrap_tensorflow_interpreter_wrapper.so",
        "//tensorflow/lite/python/metrics:metrics_interface",
        "//tensorflow/lite/python/metrics:metrics_portable.py",
    ],
    platform_name = MANYLINUX_X86_64_TAG,
    setup_py = ":setup_py",
    version = "1.1.2",
)

verify_manylinux_compliance_test(
    name = "manylinux_compliance_test",
    aarch64_compliance_tag = "manylinux_2_17_aarch64",
    ppc64le_compliance_tag = "manylinux_2_17_ppc64le",
    wheel = ":litert_wheel",
    x86_64_compliance_tag = MANYLINUX_X86_64_TAG,
)
