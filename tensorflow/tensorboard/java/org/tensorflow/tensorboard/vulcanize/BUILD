package(default_visibility = ["//tensorflow/tensorboard:internal"])

licenses(["notice"])  # Apache 2.0

java_binary(
    name = "Vulcanize",
    srcs = ["Vulcanize.java"],
    jvm_flags = [
        "-Xss20m",  # JSCompiler needs big stacks for recursive parsing
        "-XX:+UseParallelGC",  # Best GC when app isn't latency sensitive
        "-Djava.util.logging.SimpleFormatter.format='%1$$tY-%1$$tm-%1$$td %1$$tH:%1$$tM:%1$$tS.%1$$tL %4$$-6s %5$$s%6$$s%n'",  # Less log spam
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_guava",
        "@com_google_protobuf_java",
        "@io_bazel_rules_closure//closure/compiler",
        "@io_bazel_rules_closure//java/io/bazel/rules/closure:webpath",
        "@io_bazel_rules_closure//java/io/bazel/rules/closure/webfiles:build_info_java_proto",
        "@io_bazel_rules_closure//java/org/jsoup/nodes",
        "@org_jsoup",
    ],
)

java_binary(
    name = "Zipper",
    srcs = ["Zipper.java"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_guava",
        "@com_google_protobuf_java",
        "@io_bazel_rules_closure//java/io/bazel/rules/closure/webfiles",
        "@io_bazel_rules_closure//java/io/bazel/rules/closure/webfiles:build_info_java_proto",
    ],
)

# These JS files are always taken into consideration by the Closure Compiler
# when vulcanizing, per vulcanize.bzl.
filegroup(
    name = "jslibs",
    srcs = [
        # Ordering probably matters
        "@com_google_javascript_closure_compiler_externs",
        "@com_google_javascript_closure_compiler_externs_polymer",
        "externs.js",
        "@com_google_javascript_closure_library//:closure/goog/base.js",
        "@com_google_javascript_closure_library//:closure/goog/deps.js",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "all_files",
    srcs = glob(["**"]),
    tags = ["notsan"],
)
