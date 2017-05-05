package(default_visibility = ["//tensorflow:internal"])

load("@io_bazel_rules_closure//closure:defs.bzl", "webfiles")

licenses(["notice"])  # Apache 2.0

# bazel run //third_party/tensorflow/tensorboard/components/tf_dashboard_common/demo
webfiles(
    name = "demo",
    srcs = [
        "tf-categorizer-demo.html",
        "tf-collapsable-pane-demo.html",
        "tf-multi-checkbox-demo.html",
        "tf-regex-group-demo.html",
    ],
    path = "/tf-dashboard-common/demo",
    deps = [
        "//tensorflow/tensorboard/components/tf_color_scale",
        "//tensorflow/tensorboard/components/tf_dashboard_common",
        "//tensorflow/tensorboard/components/tf_imports:d3",
        "@org_polymer_iron_flex_layout",
        "@org_polymer_paper_styles",
        "@org_polymer_webcomponentsjs",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(["**"]),
    tags = ["notsan"],
)
