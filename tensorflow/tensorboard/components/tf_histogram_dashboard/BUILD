package(default_visibility = ["//tensorflow/tensorboard:internal"])

load("//tensorflow/tensorboard/defs:web.bzl", "ts_web_library")

licenses(["notice"])  # Apache 2.0

ts_web_library(
    name = "tf_histogram_dashboard",
    srcs = ["tf-histogram-dashboard.html"],
    path = "/tf-histogram-dashboard",
    deps = [
        "//tensorflow/tensorboard/components/tf_backend",
        "//tensorflow/tensorboard/components/tf_color_scale",
        "//tensorflow/tensorboard/components/tf_dashboard_common",
        "//tensorflow/tensorboard/components/tf_imports:lodash",
        "//tensorflow/tensorboard/components/tf_imports:polymer",
        "//tensorflow/tensorboard/components/vz_histogram_timeseries",
        "@org_polymer_iron_collapse",
        "@org_polymer_paper_icon_button",
        "@org_polymer_paper_styles",
    ],
)

ts_web_library(
    name = "demo",
    srcs = ["index.html"] + glob(["data/**"]),
    path = "/tf-histogram-dashboard",
    deps = [
        ":tf_histogram_dashboard",
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
