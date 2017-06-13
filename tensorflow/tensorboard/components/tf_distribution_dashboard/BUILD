package(default_visibility = ["//tensorflow/tensorboard:internal"])

load("//tensorflow/tensorboard/defs:web.bzl", "ts_web_library")

licenses(["notice"])  # Apache 2.0

ts_web_library(
    name = "tf_distribution_dashboard",
    srcs = ["tf-distribution-dashboard.html"],
    path = "/tf-distribution-dashboard",
    deps = [
        "//tensorflow/tensorboard/components/tf_backend",
        "//tensorflow/tensorboard/components/tf_color_scale",
        "//tensorflow/tensorboard/components/tf_dashboard_common",
        "//tensorflow/tensorboard/components/tf_imports:lodash",
        "//tensorflow/tensorboard/components/tf_imports:polymer",
        "//tensorflow/tensorboard/components/vz_distribution_chart",
        "@org_polymer_iron_collapse",
        "@org_polymer_paper_icon_button",
        "@org_polymer_paper_styles",
    ],
)

ts_web_library(
    name = "demo",
    srcs = ["index.html"] + glob(["data/**"]),
    path = "/tf-distribution-dashboard",
    deps = [
        ":tf_distribution_dashboard",
        "//tensorflow/tensorboard/components/tf_imports:d3",
        "//tensorflow/tensorboard/components/tf_imports:webcomponentsjs",
        "@org_polymer_iron_demo_helpers",
        "@org_polymer_paper_styles",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(["**"]),
    tags = ["notsan"],
)
