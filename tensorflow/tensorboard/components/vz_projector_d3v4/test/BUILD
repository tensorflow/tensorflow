package(
    default_testonly = True,
    default_visibility = ["//tensorflow:internal"],
)

load("@io_bazel_rules_closure//closure:defs.bzl", "web_library")
load("//tensorflow/tensorboard:hacks.bzl", "tensorboard_typescript_bundle")
load("//tensorflow/tensorboard:defs.bzl", "tensorboard_typescript_genrule")

licenses(["notice"])  # Apache 2.0

web_library(
    name = "test",
    srcs = [
        "bundle.js",
        "tests.html",
    ],
    path = "/vz-projector/test",
    deps = [
        "//tensorflow/tensorboard/components/vz_projector_d3v4",
        "@org_npmjs_registry_web_component_tester",
        "@org_polymer",
        "@org_polymer_webcomponentsjs",
    ],
)

tensorboard_typescript_genrule(
    name = "ts",
    srcs = ["bundle.ts"],
    typings = [
        "@org_definitelytyped//:chai.d.ts",
        "@org_definitelytyped//:mocha.d.ts",
        "@org_definitelytyped//:polymer.d.ts",
        "@org_definitelytyped//:three.d.ts",
        "@org_definitelytyped//:webcomponents.js.d.ts",
        "//tensorflow/tensorboard/components/tf_imports_d3v4:d3.d.ts",
        "//tensorflow/tensorboard/components/tf_imports_d3v4:plottable.d.ts",
        "//tensorflow/tensorboard/components/vz_projector_d3v4:bundle.d.ts",
    ],
)

tensorboard_typescript_bundle(
    name = "bundle",
    out = "bundle.ts",
    namespace_srcs = {
        "VZ.Projector.Test": [
            "assert.ts",
            "sptree_test.ts",
            "data_test.ts",
            "data-provider_test.ts",
            "util_test.ts",

            # TODO(smilkov): Migrate these away from jasmine.
            # "scatterPlotRectangleSelector_test.ts",
            # "vz-projector-projections-panel_test.ts",
        ],
    },
    namespace_symbol_aliases = {
        "VZ.Projector.Test": {
            "BoundingBox": "VZ.Projector.ScatterPlotRectangleSelector.BoundingBox",
            "DataPoint": "VZ.Projector.Data.DataPoint",
            "DataSet": "VZ.Projector.Data.DataSet",
            "ProjectionsPanel": "VZ.Projector.ProjectorProjectionsPanel.ProjectionsPanel",
            "SPTree": "VZ.Projector.SPTree.SPTree",
            "ScatterPlotRectangleSelector": "VZ.Projector.ScatterPlotRectangleSelector.ScatterPlotRectangleSelector",
            "SpriteAndMetadataInfo": "VZ.Projector.Data.SpriteAndMetadataInfo",
            "State": "VZ.Projector.Data.State",
            "State": "VZ.Projector.Data.State",
            "data_provider": "VZ.Projector.DataProvider",
            "stateGetAccessorDimensions": "VZ.Projector.Data.stateGetAccessorDimensions",
            "util": "VZ.Projector.Util",
        },
    },
)

filegroup(
    name = "all_files",
    testonly = 0,
    srcs = glob(["**"]),
    tags = ["notsan"],
)
