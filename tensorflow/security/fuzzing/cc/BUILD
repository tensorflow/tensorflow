# Fuzzing TensorFlow.

load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_proto_library",
)
load(
    "//tensorflow/security/fuzzing:tf_fuzzing.bzl",
    "tf_cc_fuzz_test",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

tf_cc_fuzz_test(
    name = "status_fuzz",
    srcs = ["status_fuzz.cc"],
    tags = ["no_oss"],
    deps = [
        ":fuzz_domains",
        "//tensorflow/core/platform:status",
        "@com_google_absl//absl/status",
    ],
)

tf_cc_fuzz_test(
    name = "arg_def_case_fuzz",
    srcs = ["arg_def_case_fuzz.cc"],
    tags = ["no_oss"],
    deps = [
        "//tensorflow/core/platform:str_util",
        "//tensorflow/core/platform:stringpiece",
    ],
)

tf_cc_fuzz_test(
    name = "base64_fuzz",
    srcs = ["base64_fuzz.cc"],
    tags = ["no_oss"],
    deps = [
        "//tensorflow/core/platform:base64",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:stringpiece",
        "@com_google_absl//absl/status",
    ],
)

tf_cc_fuzz_test(
    name = "bfloat16_fuzz",
    srcs = ["bfloat16_fuzz.cc"],
    tags = ["no_oss"],  # b/175698644
    deps = [
        "//tensorflow/core:portable_gif_internal",
        "//tensorflow/core/framework:bfloat16",
    ],
)

tf_cc_fuzz_test(
    name = "checkpoint_reader_fuzz",
    srcs = ["checkpoint_reader_fuzz.cc"],
    data = glob(["checkpoint_reader_testdata/*"]),
    tags = ["no_oss"],
    deps = [
        ":checkpoint_reader_fuzz_input_proto_cc",
        "//tensorflow/c:checkpoint_reader",
        "//tensorflow/c:tf_status",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/platform:resource_loader",
        "//tensorflow/core/util:saved_tensor_slice_proto_cc",
        "@local_xla//xla/tsl/platform:status",
    ],
)

tf_proto_library(
    name = "checkpoint_reader_fuzz_input_proto",
    srcs = ["checkpoint_reader_fuzz_input.proto"],
    make_default_target_header_only = True,
    protodeps = [
        "//tensorflow/core/util:saved_tensor_slice_proto",
    ],
)

tf_cc_fuzz_test(
    name = "cleanpath_fuzz",
    srcs = ["cleanpath_fuzz.cc"],
    tags = ["no_oss"],
    deps = [
        "//tensorflow/core/platform:path",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_fuzz_test(
    name = "consume_leading_digits_fuzz",
    srcs = ["consume_leading_digits_fuzz.cc"],
    tags = ["no_oss"],
    deps = [
        "//tensorflow/core:portable_gif_internal",
        "//tensorflow/core/platform:str_util",
        "//tensorflow/core/platform:stringpiece",
    ],
)

tf_cc_fuzz_test(
    name = "joinpath_fuzz",
    srcs = ["joinpath_fuzz.cc"],
    tags = ["no_oss"],
    deps = [
        "//tensorflow/core/platform:path",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_fuzz_test(
    name = "status_group_fuzz",
    srcs = ["status_group_fuzz.cc"],
    tags = ["no_oss"],
    deps = [
        ":fuzz_domains",
        "//tensorflow/core/platform:status",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "fuzz_domains",
    testonly = True,
    hdrs = ["fuzz_domains.h"],
    deps = [
        "//tensorflow/core/platform:status",
        "@com_google_fuzztest//fuzztest",
    ],
)

tf_cc_fuzz_test(
    name = "stringprintf_fuzz",
    srcs = ["stringprintf_fuzz.cc"],
    tags = ["no_oss"],
    deps = [
        "//tensorflow/core/platform:stringprintf",
    ],
)

tf_cc_fuzz_test(
    name = "string_replace_fuzz",
    srcs = ["string_replace_fuzz.cc"],
    tags = ["no_oss"],
    deps = [
        "//tensorflow/core/platform:str_util",
        "//tensorflow/core/platform:stringpiece",
    ],
)

tf_cc_fuzz_test(
    name = "tstring_fuzz",
    srcs = ["tstring_fuzz.cc"],
    tags = ["no_oss"],
    deps = [
        "//tensorflow/core/platform:tstring",
    ],
)

tf_cc_fuzz_test(
    name = "AreAttrValuesEqual_fuzz",
    srcs = ["AreAttrValuesEqual_fuzz.cc"],
    tags = ["no_oss"],  # b/175698644
    deps = [
        "//tensorflow/core/framework:attr_value_proto_cc",
        "//tensorflow/core/framework:attr_value_util",
    ],
)

tf_cc_fuzz_test(
    name = "ParseAttrValue_fuzz",
    srcs = ["ParseAttrValue_fuzz.cc"],
    tags = ["no_oss"],  # b/175698644
    deps = [
        "//tensorflow/core/framework:attr_value_proto_cc",
        "//tensorflow/core/framework:attr_value_util",
        "//tensorflow/core/platform:stringpiece",
    ],
)

tf_cc_fuzz_test(
    name = "parseURI_fuzz",
    srcs = ["parseURI_fuzz.cc"],
    tags = ["no_oss"],
    deps = [
        "//tensorflow/core/platform:path",
        "//tensorflow/core/platform:stringpiece",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "fuzz_session",
    testonly = 1,
    hdrs = ["fuzz_session.h"],
    tags = ["no_oss"],
    visibility = [
        "//tensorflow/cc/framework/fuzzing:__subpackages__",
        "//tensorflow/security/fuzzing:__subpackages__",
    ],
    deps = [
        "//tensorflow/cc:scope",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:session_options",
        "//tensorflow/core/common_runtime:direct_session_internal",
        "//tensorflow/core/framework:tensor",
        "//tensorflow/core/platform:status",
        "@com_google_fuzztest//fuzztest",
    ],
)

tf_cc_fuzz_test(
    name = "end_to_end_fuzz",
    srcs = ["end_to_end_fuzz.cc"],
    deps = [
        "//tensorflow/cc/saved_model:constants",
        "//tensorflow/cc/saved_model:loader",
        "//tensorflow/cc/saved_model:tag_constants",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core/framework:tensor",
        "//tensorflow/core/platform:status",
        "//tensorflow/security/fuzzing/cc/core/framework:datatype_domains",
        "//tensorflow/security/fuzzing/cc/core/framework:tensor_domains",
        "//tensorflow/security/fuzzing/cc/core/framework:tensor_shape_domains",
        "@com_google_absl//absl/status",
        "@local_xla//xla/tsl/platform:env",
        "@local_xla//xla/tsl/platform:status",
    ],
)

tf_cc_fuzz_test(
    name = "text_literal_reader_fuzz",
    srcs = ["text_literal_reader_fuzz.cc"],
    deps = [
        "@local_xla//xla:text_literal_reader",
        "@local_xla//xla/hlo/parser:hlo_parser",
        "@local_xla//xla/tsl/platform:env",
    ],
)
