load("@rules_cc//cc:cc_library.bzl", "cc_library")
load(
    "//xla:xla.default.bzl",
    "xla_py_proto_library",
)
load("//xla/tsl/platform:build_config.bzl", "tf_proto_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
)

tf_proto_library(
    name = "hlo_isolation_proto",
    srcs = ["hlo_isolation.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_protobuf//:any_proto",
    ],
)

xla_py_proto_library(
    name = "hlo_isolation_py_pb2",
    visibility = ["//visibility:public"],
    deps = [":hlo_isolation_proto"],
)

cc_library(
    name = "hlo_isolation_api",
    srcs = ["hlo_isolation_api.cc"],
    hdrs = ["hlo_isolation_api.h"],
    deps = [
        ":hlo_isolation_proto_cc",
        "//xla:comparison_util",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_comparison",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass_pipeline",
        "//xla/hlo/transforms:defuser",
        "//xla/hlo/transforms:despecializer",
        "//xla/hlo/transforms/simplifiers:hlo_memory_scheduler",
        "//xla/service:hlo_runner_interface",
        "//xla/tests:test_utils",
        "//xla/tools:hlo_decomposer_lib",
        "//xla/tools:hlo_module_loader",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_googlesource_code_re2//:re2",
        "@tsl//tsl/platform:path",
    ],
)
