load("@bazel_skylib//rules:common_settings.bzl", "string_flag")
load("//tensorflow/lite:litert.bzl", "MANYLINUX_AARCH64_TAG", "MANYLINUX_PPC64LE_TAG", "MANYLINUX_X86_64_TAG", "WHEEL_VERSION")
load("//tensorflow/lite/tools/pip_package/utils:py_manylinux_compliance_test.bzl", "verify_manylinux_compliance_test")
load("//tensorflow/lite/tools/pip_package/utils:py_wheel.bzl", "py_wheel")

package(
    default_visibility = ["//visibility:private"],
)

string_flag(
    name = "nightly_iso_date",
    build_setting_default = "",
)

genrule(
    name = "setup_py",
    srcs = ["//tensorflow/lite/tools/pip_package:setup_with_binary.py"],
    outs = ["setup.py"],
    cmd = "cat $< > $@",
)

genrule(
    name = "fixed_pydeps",
    srcs = [
        "//tensorflow/lite/python/metrics:metrics_portable.py",
        "//tensorflow/lite/python:interpreter",
    ],
    outs = [
        "metrics_portable.py",
        "interpreter.py",
    ],
    cmd = "for f in $(SRCS); do sed -e 's/tflite_runtime/ai_edge_litert/g' $$f > $(RULEDIR)/$$(basename $$f); done",
)

# Rename the namespace in the generated proto files to ai_edge_litert.
# This is required to maintain dependency between the two protos.
genrule(
    name = "fixed_protos",
    srcs = ["//tensorflow/lite/profiling/proto:model_runtime_info_py"],
    outs = ["model_runtime_info_pb2.py"],
    cmd = "sed -e 's/tflite\\.profiling\\.proto/ai_edge_litert/g' $< > $@",
)

py_wheel(
    name = "litert_wheel",
    srcs = [
        ":fixed_protos",
        ":fixed_pydeps",
        "//tensorflow/lite/experimental/genai:pywrap_genai_ops.so",
        "//tensorflow/lite/profiling/proto:profiling_info_py",
        "//tensorflow/lite/python:schema_py",
        "//tensorflow/lite/python/interpreter_wrapper:_pywrap_tensorflow_interpreter_wrapper.so",
        "//tensorflow/lite/python/metrics:metrics_interface",
    ],
    nightly_suffix = ":nightly_iso_date",
    platform_name = MANYLINUX_X86_64_TAG,
    project_name = "ai_edge_litert",
    setup_py = ":setup_py",
    version = WHEEL_VERSION,
)

verify_manylinux_compliance_test(
    name = "manylinux_compliance_test",
    aarch64_compliance_tag = MANYLINUX_AARCH64_TAG,
    ppc64le_compliance_tag = MANYLINUX_PPC64LE_TAG,
    wheel = ":litert_wheel",
    x86_64_compliance_tag = MANYLINUX_X86_64_TAG,
)
