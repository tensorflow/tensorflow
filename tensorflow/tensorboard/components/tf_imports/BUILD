package(default_visibility = ["//tensorflow:internal"])

load("@io_bazel_rules_closure//closure:defs.bzl", "webfiles")

licenses(["notice"])  # Apache 2.0

webfiles(
    name = "d3",
    srcs = [
        "d3.html",
        "@org_d3js",
    ],
    path = "/tf-imports",
)

webfiles(
    name = "lodash",
    srcs = ["lodash.html"],
    path = "/tf-imports",
    deps = ["@com_lodash"],
)

webfiles(
    name = "graphlib",
    srcs = [
        "graphlib.html",
        "@io_github_cpettitt_graphlib",
    ],
    path = "/tf-imports",
    deps = [":lodash"],
)

webfiles(
    name = "dagre",
    srcs = [
        "dagre.html",
        "@io_github_cpettitt_dagre",
    ],
    path = "/tf-imports",
    deps = [
        ":graphlib",
        ":lodash",
    ],
)

webfiles(
    name = "plottable",
    srcs = [
        "plottable.html",
        "@com_palantir_plottable//:plottable.css",
        "@com_palantir_plottable//:plottable.js",
    ],
    path = "/tf-imports",
    deps = [":d3"],
)

# Generate single TypeScript typings file for d3.js with no ES6 imports.
#
# The DefinitelyTyped definition of d3 v4 was written under the assumption that
# we want to use d3 in a modularized way. We don't want to do that because its
# import statements use NodeJS namespaces, and the Web Compiler only supports
# W3C, ECMA, and IETF standards.
genrule(
    name = "d3v4_typings",
    srcs = [
        # please maintain a reverse topological order
        "@org_definitelytyped_types_d3_path//:index.d.ts",
        "@org_definitelytyped_types_d3_time//:index.d.ts",
        "@org_definitelytyped_types_d3_dsv//:index.d.ts",
        "@org_definitelytyped_types_d3_color//:index.d.ts",
        "@org_definitelytyped_types_d3_selection//:index.d.ts",
        # d3-transition defines stuff in d3-selection wat?
        # "@org_definitelytyped_types_d3_transition//:index.d.ts",
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
    ],
    outs = ["d3v4.d.ts"],
    cmd = "\n".join([
        "(",
        "  echo 'declare namespace d3 {'",
        "  for f in $(SRCS); do",
        "    echo",
        "    echo /////////////////////////////////////////////",
        "    echo // $$f",
        "    echo /////////////////////////////////////////////",
        "    echo",
        "    sed '/^import /d' $$f",
        "  done",
        "  echo '}'",
        ") >$@",
    ]),
)

filegroup(
    name = "all_files",
    srcs = glob(["**"]),
    tags = ["notsan"],
)
