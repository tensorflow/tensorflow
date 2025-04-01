load("@bazel_skylib//rules:build_test.bzl", "build_test")
load("//xla:lit.bzl", "enforce_glob", "lit_test_suite")
load("//xla:py_strict.bzl", "py_strict_test")
load("//xla/tsl:tsl.bzl", "if_oss")
load("//xla/tsl:tsl.default.bzl", "filegroup")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//xla:internal"],
    licenses = ["notice"],
)

# Filegroup used to collect source files for dependency checking.
filegroup(
    name = "c_srcs",
    data = glob([
        "**/*.cc",
        "**/*.h",
    ]),
    visibility = ["//xla:internal"],
)

lit_test_suite(
    name = "hlo_opt_tests",
    srcs = enforce_glob(
        [
            # go/keep-sorted start
            "hlo_opt_all_passes_smoke_test.hlo",
            "hlo_opt_emit_proto.hlo",
            "hlo_opt_hlo_protobinary_to_hlotext.hlo",
            "hlo_opt_nonexistent_pass_failure.hlo",
            "hlo_opt_run_multiple_passes.hlo",
            "hlo_opt_run_single_pass.hlo",
            # go/keep-sorted end
        ],
        include = [
            "hlo_opt_*.hlo",
        ],
    ),
    cfg = "//xla:lit.cfg.py",
    data = [
        "hlo_opt_hlo_protobinary.pb",
    ],
    tags_override = {
        "hlo_opt_emit_proto.hlo": if_oss(["not_run:arm"]),  # TODO(b/394180263)
    },
    tools = [
        "//xla/hlo/tools:hlo-opt",
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//llvm:not",
    ],
)

py_strict_test(
    name = "generate_hlo_test_checks_test",
    srcs = ["generate_hlo_test_checks_test.py"],
    data = [
        "generate_hlo_test_checks_test_input.hlo",
        "generate_hlo_test_checks_test_output.hlo",
        "//xla/hlo/tools:hlo-opt",
    ],
    deps = [
        "//xla/hlo/tools:generate_hlo_test_checks_lib",
        "@absl_py//absl/testing:absltest",
    ],
)

# Whereas `:generate_hlo_test_checks_test` checks that the test-generation tool
# behaves as expected, this test verifies that the expected reference output
# from that test is itself a passing test.
lit_test_suite(
    name = "generate_hlo_test_checks_meta_test",
    srcs = ["generate_hlo_test_checks_test_output.hlo"],
    cfg = "//xla:lit.cfg.py",
    tools = [
        "//xla/hlo/tools:hlo-opt",
        "@llvm-project//llvm:FileCheck",
    ],
)

cc_library(
    name = "hlo_opt_test_only_passes",
    srcs = ["hlo_opt_test_only_passes.cc"],
    hdrs = ["hlo_opt_test_only_passes.h"],
    deps = [
        "//xla:shape_util",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/builder/lib:math",
        "//xla/hlo/builder/lib:matrix",
        "//xla/hlo/builder/lib:prng",
        "//xla/hlo/builder/lib:tridiagonal",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/transforms/simplifiers:algebraic_simplifier",
        "//xla/service:hlo_module_config",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

build_test(
    name = "hex_floats_to_packed_literal_build_test",
    targets = [
        "//xla/hlo/tools:hex_floats_to_packed_literal",
    ],
)

build_test(
    name = "convert_computation_build_test",
    targets = [
        "//xla/hlo/tools:convert_computation",
    ],
)

build_test(
    name = "show_text_literal_build_test",
    targets = [
        "//xla/hlo/tools:show_text_literal",
    ],
)

build_test(
    name = "hlo_proto_to_json_build_test",
    targets = [
        "//xla/hlo/tools:hlo_proto_to_json",
    ],
)

build_test(
    name = "hlo_module_metadata_processor_build_test",
    targets = [
        "//xla/hlo/tools:hlo_module_metadata_processor",
    ],
)

build_test(
    name = "show_literal_build_test",
    targets = [
        "//xla/hlo/tools:show_literal",
    ],
)
