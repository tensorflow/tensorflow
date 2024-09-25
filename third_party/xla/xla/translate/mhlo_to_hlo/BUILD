load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("//xla/tsl:tsl.bzl", "internal_visibility")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([
        "//learning/brain/mlir:tensorflow_friends",
        "//learning/brain/mlir:xla_friends",
    ]),
    licenses = ["notice"],
)

# Deprecated, use //third_party/tensorflow/compiler/xla/hlo/translate/mhlo_to_hlo:attribute_exporter
# instead.
cc_library(
    name = "attribute_exporter",
    hdrs = ["attribute_exporter.h"],
    deps = [
        "//xla/hlo/translate/mhlo_to_hlo:attribute_exporter",
    ],
)

# Deprecated, use //third_party/tensorflow/compiler/xla/hlo/translate/mhlo_to_hlo:layout_util
# instead.
cc_library(
    name = "layout_util",
    hdrs = ["layout_util.h"],
    deps = [
        "//xla/hlo/translate/mhlo_to_hlo:layout_util",
    ],
)

# Deprecated, use //third_party/tensorflow/compiler/xla/hlo/translate/mhlo_to_hlo:location_exporter
# instead.
cc_library(
    name = "location_exporter",
    hdrs = ["location_exporter.h"],
    deps = [
        "//xla/hlo/translate/mhlo_to_hlo:location_exporter",
    ],
)

# Deprecated, use //third_party/tensorflow/compiler/xla/hlo/translate/mhlo_to_hlo:module_attributes_exporter
# instead.
alias(
    name = "module_attributes_exporter",
    actual = "//xla/hlo/translate/mhlo_to_hlo:module_attributes_exporter",
)

# Deprecated, use //third_party/tensorflow/compiler/xla/hlo/translate/mhlo_to_hlo:stack_frame_index_builder
# instead.
alias(
    name = "stack_frame_index_builder",
    actual = "//xla/hlo/translate/mhlo_to_hlo:stack_frame_index_builder",
)

# Deprecated, use //third_party/tensorflow/compiler/xla/hlo/translate/mhlo_to_hlo:mlir_hlo_to_hlo
# instead.
cc_library(
    name = "mlir_hlo_to_hlo",
    hdrs = ["mlir_hlo_to_hlo.h"],
    deps = [
        "//xla/hlo/translate/mhlo_to_hlo:mlir_hlo_to_hlo",
    ],
)

# Deprecated, use //third_party/tensorflow/compiler/xla/hlo/translate/mhlo_to_hlo:translate
# instead.
cc_library(
    name = "translate",
    hdrs = ["translate.h"],
    deps = [
        "//xla/hlo/translate/mhlo_to_hlo:translate",
    ],
)

# Deprecated, use //third_party/tensorflow/compiler/xla/hlo/translate/mhlo_to_hlo:translate_registration
# instead.
cc_library(
    name = "translate_registration",
    testonly = True,
    deps = [
        "//xla/hlo/translate/mhlo_to_hlo:translate_registration",
    ],
    alwayslink = 1,
)

# Deprecated, use //third_party/tensorflow/compiler/xla/hlo/translate/mhlo_to_hlo:type_to_shape
# instead.
cc_library(
    name = "type_to_shape",
    hdrs = ["type_to_shape.h"],
    deps = [
        "//xla/hlo/translate/mhlo_to_hlo:type_to_shape",
    ],
)
