load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([
        "//learning/brain/mlir:tensorflow_friends",
        "//learning/brain/mlir:xla_friends",
    ]),
    licenses = ["notice"],
)

cc_library(
    name = "attribute_importer",
    hdrs = ["attribute_importer.h"],
    deprecation = "This library is deprecated. Use //third_party/tensorflow/compiler/xla/hlo/translate/hlo_to_mhlo:attribute_importer instead.",
    deps = [
        "//xla/hlo/translate/hlo_to_mhlo:attribute_importer",
    ],
)

cc_library(
    name = "hlo_function_importer",
    hdrs = ["hlo_function_importer.h"],
    deprecation = "This library is deprecated. Use //third_party/tensorflow/compiler/xla/hlo/translate/hlo_to_mhlo:hlo_function_importer instead.",
    deps = [
        "//xla/hlo/translate/hlo_to_mhlo:hlo_function_importer",
    ],
)

cc_library(
    name = "hlo_module_importer",
    hdrs = [
        "hlo_module_importer.h",
    ],
    deprecation = "This library is deprecated. Use //third_party/tensorflow/compiler/xla/hlo/translate/hlo_to_mhlo:hlo_module_importer instead.",
    deps = [
        "//xla/hlo/translate/hlo_to_mhlo:hlo_module_importer",
    ],
)

cc_library(
    name = "hlo_to_mlir_hlo",
    hdrs = ["hlo_to_mlir_hlo.h"],
    deprecation = "This library is deprecated. Use //third_party/tensorflow/compiler/xla/hlo/translate/hlo_to_mhlo:hlo_to_mlir_hlo instead.",
    deps = [
        "//xla/hlo/translate/hlo_to_mhlo:hlo_to_mlir_hlo",
    ],
)

cc_library(
    name = "hlo_utils",
    hdrs = ["hlo_utils.h"],
    deprecation = "This library is deprecated. Use //third_party/tensorflow/compiler/xla/hlo/translate/hlo_to_mhlo:hlo_utils instead.",
    includes = ["include"],
    deps = [
        "//xla/hlo/translate/hlo_to_mhlo:hlo_utils",
    ],
)

cc_library(
    name = "translate",
    hdrs = ["translate.h"],
    deprecation = "This library is deprecated. Use //third_party/tensorflow/compiler/xla/hlo/translate/hlo_to_mhlo:translate instead.",
    deps = [
        "//xla/hlo/translate/hlo_to_mhlo:translate",
    ],
)

cc_library(
    name = "translate_registration",
    testonly = True,
    deprecation = "This library is deprecated. Use //third_party/tensorflow/compiler/xla/hlo/translate/hlo_to_mhlo:translate_registration instead.",
    deps = [
        "//xla/hlo/translate/hlo_to_mhlo:translate_registration",
    ],
    alwayslink = 1,
)
