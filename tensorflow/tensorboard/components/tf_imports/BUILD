package(default_visibility = ["//tensorflow/tensorboard:internal"])

load("//tensorflow/tensorboard/defs:hacks.bzl", "tensorboard_typescript_bundle")
load("//tensorflow/tensorboard/defs:web.bzl", "ts_web_library")

licenses(["notice"])  # Apache 2.0

ts_web_library(
    name = "webcomponentsjs",
    srcs = ["@org_definitelytyped//:webcomponents.js.d.ts"],
    path = "/webcomponentsjs",
    visibility = ["//visibility:public"],
    exports = ["@org_polymer_webcomponentsjs"],
)

ts_web_library(
    name = "polymer",
    srcs = ["@org_definitelytyped//:polymer.d.ts"],
    path = "/polymer",
    visibility = ["//visibility:public"],
    exports = ["@org_polymer"],
    deps = [":webcomponentsjs"],
)

ts_web_library(
    name = "lodash",
    srcs = [
        "lodash.html",
        "@org_definitelytyped//:lodash.d.ts",
    ],
    path = "/tf-imports",
    visibility = ["//visibility:public"],
    deps = ["@com_lodash"],
)

ts_web_library(
    name = "threejs",
    srcs = [
        "threejs.html",
        "@org_definitelytyped//:three.d.ts",
        "@org_threejs//:OrbitControls.js",
        "@org_threejs//:three.js",
    ],
    path = "/tf-imports",
    visibility = ["//visibility:public"],
)

ts_web_library(
    name = "numericjs",
    srcs = [
        "numericjs.html",
        "@com_numericjs//:numeric.js",
    ],
    path = "/tf-imports",
    visibility = ["//visibility:public"],
)

ts_web_library(
    name = "weblas",
    srcs = [
        "weblas.html",
        "@io_github_waylonflinn_weblas//:weblas.js",
    ],
    path = "/tf-imports",
    visibility = ["//visibility:public"],
)

ts_web_library(
    name = "graphlib",
    srcs = [
        "graphlib.html",
        "@io_github_cpettitt_graphlib//:graphlib.core.js",
    ],
    path = "/tf-imports",
    visibility = ["//visibility:public"],
    deps = [":lodash"],
)

ts_web_library(
    name = "dagre",
    srcs = [
        "dagre.html",
        "@io_github_cpettitt_dagre//:dagre.core.js",
    ],
    path = "/tf-imports",
    visibility = ["//visibility:public"],
    deps = [
        ":graphlib",
        ":lodash",
    ],
)

ts_web_library(
    name = "d3",
    srcs = [
        "d3.d.ts",
        "d3.html",
        "@org_d3js//:d3.min.js",
    ],
    path = "/tf-imports",
    visibility = ["//visibility:public"],
)

ts_web_library(
    name = "plottable",
    srcs = [
        "plottable.d.ts",
        "plottable.html",
    ],
    path = "/tf-imports",
    visibility = ["//visibility:public"],
    deps = [
        ":d3",
        ":plottable_js_css",
    ],
)

ts_web_library(
    name = "plottable_js_css",
    srcs = [
        "@com_palantir_plottable//:package/plottable.css",
        "@com_palantir_plottable//:package/plottable.js",
    ],
    path = "/tf-imports",
    strip_prefix = "package",
    visibility = ["//visibility:private"],
)

ts_web_library(
    name = "web_component_tester",
    testonly = 1,
    visibility = ["//visibility:public"],
    exports = [
        ":chai_typings",
        ":mocha_typings",
        ":sinon_typings",
        "@org_npmjs_registry_web_component_tester",
    ],
)

ts_web_library(
    name = "chai_typings",
    testonly = 1,
    srcs = ["@org_definitelytyped//:chai.d.ts"],
    path = "/chai",
    visibility = ["//visibility:private"],
)

ts_web_library(
    name = "mocha_typings",
    testonly = 1,
    srcs = ["@org_definitelytyped//:mocha.d.ts"],
    path = "/mocha",
    visibility = ["//visibility:private"],
)

ts_web_library(
    name = "sinon_typings",
    testonly = 1,
    srcs = ["@org_definitelytyped//:sinon.d.ts"],
    path = "/sinonjs",
    visibility = ["//visibility:private"],
)

# Generate single TypeScript typings file for d3.js with no ES6 imports.
#
# The DefinitelyTyped definition of d3 v4 was written under the assumption that
# we want to use d3 in a modularized way. We don't want to do that because its
# import statements use NodeJS namespaces, and the Web Compiler only supports
# W3C, ECMA, and IETF standards.
tensorboard_typescript_bundle(
    name = "d3_typings",
    out = "d3.d.ts",
    namespace_srcs = {"d3": [
        "d3-transition.d.ts",
        "@org_definitelytyped_types_d3_path//:index.d.ts",
        "@org_definitelytyped_types_d3_time//:index.d.ts",
        "@org_definitelytyped_types_d3_dsv//:index.d.ts",
        "@org_definitelytyped_types_d3_color//:index.d.ts",
        "@org_definitelytyped_types_d3_selection//:index.d.ts",
        "@org_definitelytyped_types_d3_shape//:index.d.ts",
        "@org_definitelytyped_types_d3_scale//:index.d.ts",
        "@org_definitelytyped_types_d3_request//:index.d.ts",
        "@org_definitelytyped_types_d3_interpolate//:index.d.ts",
        "@org_definitelytyped_types_d3_drag//:index.d.ts",
        "@org_definitelytyped_types_d3_brush//:index.d.ts",
        "@org_definitelytyped_types_d3_axis//:index.d.ts",
        "@org_definitelytyped_types_d3_zoom//:index.d.ts",
        "@org_definitelytyped_types_d3_array//:index.d.ts",
        "@org_definitelytyped_types_d3_chord//:index.d.ts",
        "@org_definitelytyped_types_d3_collection//:index.d.ts",
        "@org_definitelytyped_types_d3_dispatch//:index.d.ts",
        "@org_definitelytyped_types_d3_ease//:index.d.ts",
        "@org_definitelytyped_types_d3_force//:index.d.ts",
        "@org_definitelytyped_types_d3_format//:index.d.ts",
        "@org_definitelytyped_types_d3_hierarchy//:index.d.ts",
        "@org_definitelytyped_types_d3_polygon//:index.d.ts",
        "@org_definitelytyped_types_d3_quadtree//:index.d.ts",
        "@org_definitelytyped_types_d3_queue//:index.d.ts",
        "@org_definitelytyped_types_d3_random//:index.d.ts",
        "@org_definitelytyped_types_d3_timer//:index.d.ts",
        "@org_definitelytyped_types_d3_voronoi//:index.d.ts",
    ]},
)

# It would be nice if Plottable released a .d.ts file for plottable.js like
# they did for previous versions.
tensorboard_typescript_bundle(
    name = "plottable_typings",
    out = "plottable.d.ts",
    namespace_srcs = {
        "Plottable": [
            "@com_palantir_plottable//:package/build/src/core/dataset.d.ts",
            "@com_palantir_plottable//:package/build/src/core/interfaces.d.ts",
            "@com_palantir_plottable//:package/build/src/core/version.d.ts",
        ],
        "Plottable.Animators": [
            "@com_palantir_plottable//:package/build/src/animators/animator.d.ts",
            "@com_palantir_plottable//:package/build/src/animators/easingAnimator.d.ts",
            "@com_palantir_plottable//:package/build/src/animators/nullAnimator.d.ts",
        ],
        "Plottable.Axes": [
            "@com_palantir_plottable//:package/build/src/axes/axis.d.ts",
            "@com_palantir_plottable//:package/build/src/axes/categoryAxis.d.ts",
            "@com_palantir_plottable//:package/build/src/axes/numericAxis.d.ts",
            "@com_palantir_plottable//:package/build/src/axes/timeAxis.d.ts",
        ],
        "Plottable.Components": [
            "@com_palantir_plottable//:package/build/src/components/component.d.ts",
            "@com_palantir_plottable//:package/build/src/components/componentContainer.d.ts",
            "@com_palantir_plottable//:package/build/src/components/dragBoxLayer.d.ts",
            "@com_palantir_plottable//:package/build/src/components/dragLineLayer.d.ts",
            "@com_palantir_plottable//:package/build/src/components/gridlines.d.ts",
            "@com_palantir_plottable//:package/build/src/components/group.d.ts",
            "@com_palantir_plottable//:package/build/src/components/guideLineLayer.d.ts",
            "@com_palantir_plottable//:package/build/src/components/interpolatedColorLegend.d.ts",
            "@com_palantir_plottable//:package/build/src/components/label.d.ts",
            "@com_palantir_plottable//:package/build/src/components/legend.d.ts",
            "@com_palantir_plottable//:package/build/src/components/plotGroup.d.ts",
            "@com_palantir_plottable//:package/build/src/components/selectionBoxLayer.d.ts",
            "@com_palantir_plottable//:package/build/src/components/table.d.ts",
            "@com_palantir_plottable//:package/build/src/components/xDragBoxLayer.d.ts",
            "@com_palantir_plottable//:package/build/src/components/yDragBoxLayer.d.ts",
        ],
        "Plottable.Configs": [
            "@com_palantir_plottable//:package/build/src/core/config.d.ts",
        ],
        "Plottable.Formatters": [
            "@com_palantir_plottable//:package/build/src/core/formatters.d.ts",
        ],
        "Plottable.RenderController": [
            "@com_palantir_plottable//:package/build/src/core/renderController.d.ts",
        ],
        "Plottable.RenderPolicies": [
            "@com_palantir_plottable//:package/build/src/core/renderPolicy.d.ts",
        ],
        "Plottable.SymbolFactories": [
            "@com_palantir_plottable//:package/build/src/core/symbolFactories.d.ts",
        ],
        "Plottable.Dispatchers": [
            "@com_palantir_plottable//:package/build/src/dispatchers/dispatcher.d.ts",
            "@com_palantir_plottable//:package/build/src/dispatchers/keyDispatcher.d.ts",
            "@com_palantir_plottable//:package/build/src/dispatchers/mouseDispatcher.d.ts",
            "@com_palantir_plottable//:package/build/src/dispatchers/touchDispatcher.d.ts",
        ],
        "Plottable.Drawers": [
            "@com_palantir_plottable//:package/build/src/drawers/arcDrawer.d.ts",
            "@com_palantir_plottable//:package/build/src/drawers/arcOutlineDrawer.d.ts",
            "@com_palantir_plottable//:package/build/src/drawers/areaDrawer.d.ts",
            "@com_palantir_plottable//:package/build/src/drawers/canvasBuffer.d.ts",
            "@com_palantir_plottable//:package/build/src/drawers/canvasDrawer.d.ts",
            "@com_palantir_plottable//:package/build/src/drawers/drawStep.d.ts",
            "@com_palantir_plottable//:package/build/src/drawers/drawer.d.ts",
            "@com_palantir_plottable//:package/build/src/drawers/lineDrawer.d.ts",
            "@com_palantir_plottable//:package/build/src/drawers/rectangleDrawer.d.ts",
            "@com_palantir_plottable//:package/build/src/drawers/segmentDrawer.d.ts",
            "@com_palantir_plottable//:package/build/src/drawers/svgDrawer.d.ts",
            "@com_palantir_plottable//:package/build/src/drawers/symbolDrawer.d.ts",
        ],
        "Plottable.Interactions": [
            "@com_palantir_plottable//:package/build/src/interactions/clickInteraction.d.ts",
            "@com_palantir_plottable//:package/build/src/interactions/dragInteraction.d.ts",
            "@com_palantir_plottable//:package/build/src/interactions/interaction.d.ts",
            "@com_palantir_plottable//:package/build/src/interactions/keyInteraction.d.ts",
            "@com_palantir_plottable//:package/build/src/interactions/panZoomInteraction.d.ts",
            "@com_palantir_plottable//:package/build/src/interactions/pointerInteraction.d.ts",
        ],
        "Plottable.Plots": [
            "@com_palantir_plottable//:package/build/src/plots/areaPlot.d.ts",
            "@com_palantir_plottable//:package/build/src/plots/barPlot.d.ts",
            "@com_palantir_plottable//:package/build/src/plots/clusteredBarPlot.d.ts",
            "@com_palantir_plottable//:package/build/src/plots/commons.d.ts",
            "@com_palantir_plottable//:package/build/src/plots/linePlot.d.ts",
            "@com_palantir_plottable//:package/build/src/plots/piePlot.d.ts",
            "@com_palantir_plottable//:package/build/src/plots/plot.d.ts",
            "@com_palantir_plottable//:package/build/src/plots/rectanglePlot.d.ts",
            "@com_palantir_plottable//:package/build/src/plots/scatterPlot.d.ts",
            "@com_palantir_plottable//:package/build/src/plots/segmentPlot.d.ts",
            "@com_palantir_plottable//:package/build/src/plots/stackedAreaPlot.d.ts",
            "@com_palantir_plottable//:package/build/src/plots/stackedBarPlot.d.ts",
            "@com_palantir_plottable//:package/build/src/plots/waterfallPlot.d.ts",
            "@com_palantir_plottable//:package/build/src/plots/xyPlot.d.ts",
        ],
        "Plottable.Scales": [
            "@com_palantir_plottable//:package/build/src/scales/index.d.ts",
            "@com_palantir_plottable//:package/build/src/scales/categoryScale.d.ts",
            "@com_palantir_plottable//:package/build/src/scales/colorScale.d.ts",
            "@com_palantir_plottable//:package/build/src/scales/interpolatedColorScale.d.ts",
            "@com_palantir_plottable//:package/build/src/scales/linearScale.d.ts",
            "@com_palantir_plottable//:package/build/src/scales/modifiedLogScale.d.ts",
            "@com_palantir_plottable//:package/build/src/scales/quantitativeScale.d.ts",
            "@com_palantir_plottable//:package/build/src/scales/scale.d.ts",
            "@com_palantir_plottable//:package/build/src/scales/timeScale.d.ts",
        ],
        "Plottable.Scales.TickGenerators": [
            "@com_palantir_plottable//:package/build/src/scales/tickGenerators.d.ts",
        ],
        "Plottable.Utils": [
            "@com_palantir_plottable//:package/build/src/utils/addD3SelectionMulti.d.ts",
            "@com_palantir_plottable//:package/build/src/utils/bucket.d.ts",
            "@com_palantir_plottable//:package/build/src/utils/callbackSet.d.ts",
            "@com_palantir_plottable//:package/build/src/utils/coerceD3.d.ts",
            "@com_palantir_plottable//:package/build/src/utils/entityStore.d.ts",
            "@com_palantir_plottable//:package/build/src/utils/makeEnum.d.ts",
            "@com_palantir_plottable//:package/build/src/utils/map.d.ts",
            "@com_palantir_plottable//:package/build/src/utils/set.d.ts",
            "@com_palantir_plottable//:package/build/src/utils/transformAwareTranslator.d.ts",
        ],
        "Plottable.Utils.Array": [
            "@com_palantir_plottable//:package/build/src/utils/arrayUtils.d.ts",
        ],
        "Plottable.Utils.Color": [
            "@com_palantir_plottable//:package/build/src/utils/colorUtils.d.ts",
        ],
        "Plottable.Utils.DOM": [
            "@com_palantir_plottable//:package/build/src/utils/domUtils.d.ts",
        ],
        "Plottable.Utils.Math": [
            "@com_palantir_plottable//:package/build/src/utils/mathUtils.d.ts",
        ],
        "Plottable.Utils.Stacking": [
            "@com_palantir_plottable//:package/build/src/utils/stackingUtils.d.ts",
        ],
        "Plottable.Utils.Window": [
            "@com_palantir_plottable//:package/build/src/utils/windowUtils.d.ts",
        ],
    },
    namespace_symbol_aliases = {
        "Plottable.Animators": {
            "AttributeToAppliedProjector": "Plottable.AttributeToAppliedProjector",
            "SimpleSelection": "Plottable.SimpleSelection",
        },
        "Plottable.Axes": {
            "Component": "Plottable.Components.Component",
            "Formatter": "Plottable.Formatters.Formatter",
            "Point": "Plottable.Point",
            "QuantitativeScale": "Plottable.Scales.QuantitativeScale",
            "Scale": "Plottable.Scales.Scale",
            "Scales": "Plottable.Scales",
            "SimpleSelection": "Plottable.SimpleSelection",
            "SpaceRequest": "Plottable.SpaceRequest",
        },
        "Plottable.Components": {
            "Bounds": "Plottable.Bounds",
            "Formatter": "Plottable.Formatters.Formatter",
            "IEntity": "Plottable.IEntity",
            "Interactions": "Plottable.Interactions",
            "Plots": "Plottable.Plots",
            "Point": "Plottable.Point",
            "QuantitativeScale": "Plottable.Scales.QuantitativeScale",
            "Scales": "Plottable.Scales",
            "SimpleSelection": "Plottable.SimpleSelection",
            "SpaceRequest": "Plottable.SpaceRequest",
            "SymbolFactory": "Plottable.SymbolFactories.SymbolFactory",
        },
        "Plottable.RenderController": {
            "Component": "Plottable.Components.Component",
            "RenderPolicies": "Plottable.RenderPolicies",
        },
        "Plottable.SymbolFactories": {
            "d3Shape": "d3",
        },
        "Plottable.Dispatchers": {
            "Component": "Plottable.Components.Component",
            "Dispatchers": "Plottable.Dispatchers",
            "Point": "Plottable.Point",
        },
        "Plottable.Drawers": {
            "AttributeToAppliedProjector": "Plottable.AttributeToAppliedProjector",
            "AttributeToProjector": "Plottable.AttributeToProjector",
            "Dataset": "Plottable.Dataset",
            "IAccessor": "Plottable.IAccessor",
            "IAnimator": "Plottable.Animators.IAnimator",
            "SimpleSelection": "Plottable.SimpleSelection",
            "SymbolFactory": "Plottable.SymbolFactories.SymbolFactory",
        },
        "Plottable.Interactions": {
            "Component": "Plottable.Components.Component",
            "Point": "Plottable.Point",
            "TransformableScale": "Plottable.Scales.TransformableScale",
        },
        "Plottable.Plots": {
            "AppliedDrawStep": "Plottable.Drawers.AppliedDrawStep",
            "AttributeToProjector": "Plottable.AttributeToProjector",
            "Bounds": "Plottable.Bounds",
            "Component": "Plottable.Components.Component",
            "Dataset": "Plottable.Dataset",
            "DrawStep": "Plottable.Drawers.DrawStep",
            "Drawers": "Plottable.Drawers",
            "Formatter": "Plottable.Formatters.Formatter",
            "IAccessor": "Plottable.IAccessor",
            "IAnimator": "Plottable.Animators.IAnimator",
            "IDrawer": "Plottable.Drawers.IDrawer",
            "IEntity": "Plottable.IEntity",
            "IScaleCallback": "Plottable.Scales.IScaleCallback",
            "Plots": "Plottable.Plots",
            "Point": "Plottable.Point",
            "Projector": "Plottable.Projector",
            "ProxyDrawer": "Plottable.Drawers.ProxyDrawer",
            "QuantitativeScale": "Plottable.Scales.QuantitativeScale",
            "Range": "Plottable.Range",
            "Scale": "Plottable.Scales.Scale",
            "SimpleSelection": "Plottable.SimpleSelection",
            "SymbolFactory": "Plottable.SymbolFactories.SymbolFactory",
            "TransformableScale": "Plottable.Scales.TransformableScale",
            "Utils": "Plottable.Utils",
            "d3Shape": "d3",
        },
        "Plottable.Scales": {
            "Dataset": "Plottable.Dataset",
            "Scales": "Plottable.Scales",
        },
        "Plottable.Scales.TickGenerators": {
            "QuantitativeScale": "Plottable.Scales.QuantitativeScale",
        },
        "Plottable.Utils": {
            "Bounds": "Plottable.Bounds",
            "Component": "Plottable.Components.Component",
            "Dataset": "Plottable.Dataset",
            "IAccessor": "Plottable.IAccessor",
            "Point": "Plottable.Point",
            "Range": "Plottable.Range",
            "SimpleSelection": "Plottable.SimpleSelection",
            "Utils": "Plottable.Utils",
        },
    },
    namespace_symbol_aliases_public = {
        "Plottable": {
            "Axis": "Plottable.Axes.Axis",
            "AxisOrientation": "Plottable.Axes.AxisOrientation",
            "ClickCallback": "Plottable.Interactions.ClickCallback",
            "Component": "Plottable.Components.Component",
            "ComponentCallback": "Plottable.Components.ComponentCallback",
            "ComponentContainer": "Plottable.Components.ComponentContainer",
            "Dispatcher": "Plottable.Dispatchers.Dispatcher",
            "DragBoxCallback": "Plottable.Components.DragBoxCallback",
            "DragCallback": "Plottable.Interactions.DragCallback",
            "EaseFn": "Plottable.Animators.EaseFn",
            "EaseName": "Plottable.Animators.EaseName",
            "Easing": "Plottable.Animators.Easing",
            "Formatter": "Plottable.Formatters.Formatter",
            "IAnimator": "Plottable.Animators.IAnimator",
            "IDragLineCallback": "Plottable.Components.IDragLineCallback",
            "IDrawer": "Plottable.Drawers.IDrawer",
            "IResizeHandler": "Plottable.Components.IResizeHandler",
            "IScaleCallback": "Plottable.Scales.IScaleCallback",
            "Interaction": "Plottable.Interactions.Interaction",
            "Key": "Plottable.Interactions.Key",
            "KeyCallback": "Plottable.Interactions.KeyCallback",
            "Null": "Plottable.Animators.Null",
            "Plot": "Plottable.Plots.Plot",
            "PointerCallback": "Plottable.Interactions.PointerCallback",
            "ProxyDrawer": "Plottable.Drawers.ProxyDrawer",
            "QuantitativeScale": "Plottable.Scales.QuantitativeScale",
            "Renderer": "Plottable.Plots.Renderer",
            "Scale": "Plottable.Scales.Scale",
            "SymbolFactory": "Plottable.SymbolFactories.SymbolFactory",
            "TimeInterval": "Plottable.Axes.TimeInterval",
            "TransformableScale": "Plottable.Scales.TransformableScale",
            "XAlignment": "Plottable.Components.XAlignment",
            "XYPlot": "Plottable.Plots.XYPlot",
            "YAlignment": "Plottable.Components.YAlignment",
        },
    },
)

# Removes the 'declare module' block inside this file, but keeps its content.
genrule(
    name = "kludge_d3_transition",
    srcs = ["@org_definitelytyped_types_d3_transition//:index.d.ts"],
    outs = ["d3-transition.d.ts"],
    cmd = "sed '/^declare module/d' $< | awk '/^}$$/ && !p {p++;next}1' >$@",
)

filegroup(
    name = "all_files",
    srcs = glob(["**"]),
    tags = ["notsan"],
)
