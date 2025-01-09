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
    name = "attribute_exporter",
    hdrs = ["attribute_exporter.h"],
    deprecation = "This library is deprecated. Use //third_party/tensorflow/compiler/xla/hlo/translate/mhlo_to_hlo:attribute_exporter instead.",
    deps = [
        "//xla/hlo/translate/mhlo_to_hlo:attribute_exporter",
    ],
)

cc_library(
    name = "layout_util",
    hdrs = ["layout_util.h"],
    deprecation = "This library is deprecated. Use //third_party/tensorflow/compiler/xla/hlo/translate/mhlo_to_hlo:layout_util instead.",
    deps = [
        "//xla/hlo/translate/mhlo_to_hlo:layout_util",
    ],
)

cc_library(
    name = "location_exporter",
    hdrs = ["location_exporter.h"],
    deprecation = "This library is deprecated. Use //third_party/tensorflow/compiler/xla/hlo/translate/mhlo_to_hlo:location_exporter instead.",
    deps = [
        "//xla/hlo/translate/mhlo_to_hlo:location_exporter",
    ],
)

alias(
    name = "module_attributes_exporter",
    actual = "//xla/hlo/translate/mhlo_to_hlo:module_attributes_exporter",
    deprecation = "This library is deprecated. Use //third_party/tensorflow/compiler/xla/hlo/translate/mhlo_to_hlo:module_attributes_exporter instead.",
)

alias(
    name = "stack_frame_index_builder",
    actual = "//xla/hlo/translate/mhlo_to_hlo:stack_frame_index_builder",
    deprecation = "This library is deprecated. Use //third_party/tensorflow/compiler/xla/hlo/translate/mhlo_to_hlo:stack_frame_index_builder instead.",
)

cc_library(
    name = "mlir_hlo_to_hlo",
    hdrs = ["mlir_hlo_to_hlo.h"],
    deprecation = "This library is deprecated. Use //third_party/tensorflow/compiler/xla/hlo/translate/mhlo_to_hlo:mlir_hlo_to_hlo instead.",
    deps = [
        "//xla/hlo/translate/mhlo_to_hlo:mlir_hlo_to_hlo",
    ],
)

cc_library(
    name = "translate",
    hdrs = ["translate.h"],
    deprecation = "This library is deprecated. Use //third_party/tensorflow/compiler/xla/hlo/translate/mhlo_to_hlo:translate instead.",
    deps = [
        "//xla/hlo/translate/mhlo_to_hlo:translate",
    ],
)

cc_library(
    name = "translate_registration",
    testonly = True,
    deprecation = "This library is deprecated. Use //third_party/tensorflow/compiler/xla/hlo/translate/mhlo_to_hlo:translate_registration instead.",
    deps = [
        "//xla/hlo/translate/mhlo_to_hlo:translate_registration",
    ],
    alwayslink = 1,
)

cc_library(
    name = "type_to_shape",
    hdrs = ["type_to_shape.h"],
    deprecation = "This library is deprecated. Use //third_party/tensorflow/compiler/xla/hlo/translate/mhlo_to_hlo:type_to_shape instead.",
    deps = [
        "//xla/hlo/translate/mhlo_to_hlo:type_to_shape",
    ],
)
