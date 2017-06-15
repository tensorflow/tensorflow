package(default_visibility = ["//tensorflow/tensorboard:internal"])

load("//tensorflow/tensorboard/defs:web.bzl", "ts_web_library")

licenses(["notice"])  # Apache 2.0

ts_web_library(
    name = "vz_projector",
    srcs = [
        "analyticsLogger.ts",
        "bundle.html",
        "data.ts",
        "data-provider.ts",
        "data-provider-demo.ts",
        "data-provider-proto.ts",
        "data-provider-server.ts",
        "external.d.ts",
        "knn.ts",
        "label.ts",
        "logging.ts",
        "projectorEventContext.ts",
        "projectorScatterPlotAdapter.ts",
        "renderContext.ts",
        "scatterPlot.ts",
        "scatterPlotRectangleSelector.ts",
        "scatterPlotVisualizer.ts",
        "scatterPlotVisualizer3DLabels.ts",
        "scatterPlotVisualizerCanvasLabels.ts",
        "scatterPlotVisualizerPolylines.ts",
        "scatterPlotVisualizerSprites.ts",
        "styles.html",
        "util.ts",
        "vector.ts",
        "vz-projector.html",
        "vz-projector.ts",
        "vz-projector-app.html",
        "vz-projector-bookmark-panel.html",
        "vz-projector-bookmark-panel.ts",
        "vz-projector-colab.html",
        "vz-projector-dashboard.html",
        "vz-projector-data-panel.html",
        "vz-projector-data-panel.ts",
        "vz-projector-input.html",
        "vz-projector-input.ts",
        "vz-projector-inspector-panel.html",
        "vz-projector-inspector-panel.ts",
        "vz-projector-legend.html",
        "vz-projector-legend.ts",
        "vz-projector-metadata-card.html",
        "vz-projector-metadata-card.ts",
        "vz-projector-projections-panel.html",
        "vz-projector-projections-panel.ts",
        "vz-projector-util.ts",
    ],
    path = "/vz-projector",
    visibility = ["//visibility:public"],
    deps = [
        ":bh_tsne",
        ":heap",
        ":sptree",
        "//tensorflow/tensorboard/components/tf_dashboard_common",
        "//tensorflow/tensorboard/components/tf_imports:d3",
        "//tensorflow/tensorboard/components/tf_imports:numericjs",
        "//tensorflow/tensorboard/components/tf_imports:polymer",
        "//tensorflow/tensorboard/components/tf_imports:threejs",
        "//tensorflow/tensorboard/components/tf_imports:weblas",
        "@org_polymer_iron_collapse",
        "@org_polymer_iron_icons",
        "@org_polymer_paper_button",
        "@org_polymer_paper_checkbox",
        "@org_polymer_paper_dialog",
        "@org_polymer_paper_dialog_scrollable",
        "@org_polymer_paper_dropdown_menu",
        "@org_polymer_paper_icon_button",
        "@org_polymer_paper_input",
        "@org_polymer_paper_item",
        "@org_polymer_paper_listbox",
        "@org_polymer_paper_slider",
        "@org_polymer_paper_spinner",
        "@org_polymer_paper_styles",
        "@org_polymer_paper_toast",
        "@org_polymer_paper_toggle_button",
        "@org_polymer_paper_tooltip",
    ],
)

ts_web_library(
    name = "heap",
    srcs = ["heap.ts"],
    path = "/vz-projector",
)

ts_web_library(
    name = "sptree",
    srcs = ["sptree.ts"],
    path = "/vz-projector",
)

ts_web_library(
    name = "bh_tsne",
    srcs = ["bh_tsne.ts"],
    path = "/vz-projector",
    deps = [":sptree"],
)

filegroup(
    name = "all_files",
    srcs = glob(["**"]),
    tags = ["notsan"],
)
