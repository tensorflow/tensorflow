# MLIR related utilities.

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "status_scoped_diagnostic_handler",
    srcs = ["status_scoped_diagnostic_handler.cc"],
    hdrs = [
        "status_scoped_diagnostic_handler.h",
    ],
    deps = [
        "@com_google_absl//absl/status",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@local_tsl//tsl/platform:logging",
    ],
)
