package(default_visibility = [":projector_group"])

load(
    "//tensorflow/tensorboard:defs.bzl",
    "tensorboard_karma_web_test_suite",
    "tensorboard_ts_config",
    "tensorboard_ts_declaration",
    "tensorboard_ts_development_sources",
    "tensorboard_ts_library",
    "tensorboard_webcomponent_library",
)

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "all_files",
    srcs = glob(
        ["**"],
        exclude = [
            "OWNERS",
            "tsconfig.json",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)

# Visibility group for all clients of projector.
package_group(
    name = "projector_group",
    packages = [
        "//apps/labs/towerbridge/...",
        "//experimental/bigpicture/projector/...",
        "//java/com/google/apps/labs/towerbridge/...",
        "//learning/vis/projector/...",
        "//tensorflow/tensorboard/...",
    ],
)

tensorboard_ts_declaration(
    name = "external",
    srcs = ["external.d.ts"],
)

tensorboard_ts_library(
    name = "ts_lib",
    srcs = glob(
        ["*.ts"],
        exclude = [
            "*.d.ts",
            "*_test.ts",
            "bh_tsne.ts",
            "sptree.ts",
        ],
    ),
    runtime_deps = [
        "//third_party/javascript/numericjs",
        "//third_party/javascript/threejs/r77:threejs",
        "//third_party/javascript/threejs/r77/examples/js/controls:orbitcontrols",
        "//third_party/javascript/weblas",
    ],
    deps = [
        ":external",
        ":tsne_ts_lib",
        "//third_party/javascript/node_modules/typescript:es2015.promise",
        "//third_party/javascript/typings/d3_v4:bundle",
        "//third_party/javascript/typings/polymer:polymer_without_externs",
        "//third_party/javascript/typings/threejs:three",
        "//third_party/javascript/typings/webcomponents_js",
    ],
)

tensorboard_ts_library(
    name = "tsne_ts_lib",
    srcs = [
        "bh_tsne.ts",
        "sptree.ts",
    ],
)

_PROJECTOR_LIB_DEPS = [
    "//third_party/javascript/polymer/v1/iron-collapse:lib",
    "//third_party/javascript/polymer/v1/iron-icons:lib",
    "//third_party/javascript/polymer/v1/paper-button:lib",
    "//third_party/javascript/polymer/v1/paper-checkbox:lib",
    "//third_party/javascript/polymer/v1/paper-dialog:lib",
    "//third_party/javascript/polymer/v1/paper-dialog-scrollable:lib",
    "//third_party/javascript/polymer/v1/paper-dropdown-menu:lib",
    "//third_party/javascript/polymer/v1/paper-icon-button:lib",
    "//third_party/javascript/polymer/v1/paper-input:lib",
    "//third_party/javascript/polymer/v1/paper-item:lib",
    "//third_party/javascript/polymer/v1/paper-listbox:lib",
    "//third_party/javascript/polymer/v1/paper-slider:lib",
    "//third_party/javascript/polymer/v1/paper-spinner:lib",
    "//third_party/javascript/polymer/v1/paper-toast:lib",
    "//third_party/javascript/polymer/v1/paper-toggle-button:lib",
    "//third_party/javascript/polymer/v1/paper-tooltip:lib",
    "//third_party/javascript/polymer/v1/polymer:lib",
]

_PROJECTOR_DESTDIR = "vz-projector"

_PROJECTOR_LIB_TS_LIB_DEPS = [
    ":ts_lib",
    ":tsne_ts_lib",
]

# Standalone embedding projector demos should depend on this target. We
# exclude the HTML file for the dashboard itself. Demos do not need that
# HTML file. This was introduced because standalone demos as of today
# have an additional Closure pass that uses a compilation configuration
# stricter than that of TensorBoard.
tensorboard_webcomponent_library(
    name = "lib",
    srcs = glob(
        ["*.html"],
        exclude = ["vz-projector-dashboard.html"],
    ),
    ts_lib_deps = _PROJECTOR_LIB_TS_LIB_DEPS,
    destdir = _PROJECTOR_DESTDIR,
    deps = _PROJECTOR_LIB_DEPS,
)

# TensorBoard, however, should depend on this target, which includes
# the HTML file for the dashboard.
tensorboard_webcomponent_library(
    name = "lib_for_tensorboard",
    srcs = glob(["*.html"]),
    ts_lib_deps = _PROJECTOR_LIB_TS_LIB_DEPS,
    destdir = _PROJECTOR_DESTDIR,
    deps = _PROJECTOR_LIB_DEPS,
)

### Tests ###

tensorboard_ts_library(
    name = "ts_test",
    testonly = 1,
    srcs = glob(["*_test.ts"]),
    runtime_deps = [
        "//third_party/javascript/polymer/v1/polymer:lib_all_js",
    ],
    deps = [
        ":ts_lib",
        ":tsne_ts_lib",
        "//third_party/javascript/typings/chai",
        "//third_party/javascript/typings/jasmine:jasmine_without_externs",
        "//third_party/javascript/typings/mocha",
    ],
)

tensorboard_ts_development_sources(
    name = "dev_sources_for_test",
    testonly = 1,
    runtime_deps = [
        "//third_party/javascript/chai",
        "//third_party/javascript/mocha",
    ],
    deps = [
        ":ts_test",
    ],
)

# To run locally, run :all_tests_local
tensorboard_karma_web_test_suite(
    name = "all_tests",
    size = "medium",
    browsers = ["//testing/web/browsers:chrome-linux"],
    manifest = ":dev_sources_for_test",
)

# Generate a TypeScript IDE project by running this target.
tensorboard_ts_config(
    name = "tsconfig",
    deps = [
        ":ts_lib",
        ":ts_test",
        ":tsne_ts_lib",
    ],
)
