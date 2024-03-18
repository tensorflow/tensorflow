package_group(
    name = "friends",
    packages = [
        # copybara:uncomment_begin(google-only)
        # "//platforms/xla/service/cpu/...",
        # "//third_party/mlir_edge/tpgen/...",
        # # TODO(ezhulenev): Clean up dependencies that are leftovers from Autofusion project.
        # "@tf_runtime//...",
        # "//third_party/tf_runtime_google/...",
        # copybara:uncomment_end(google-only)
        "//tensorflow/compiler/mlir/tfrt/...",
        "//xla/mlir/...",
        "//xla/runtime/...",
        # TODO(ezhulenev): All targets depending on mlir must be under xla/mlir folder
        "//xla/service/cpu/...",
        "//xla/service/gpu/...",
        "//third_party/py/enzyme_ad/...",
    ],
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)
