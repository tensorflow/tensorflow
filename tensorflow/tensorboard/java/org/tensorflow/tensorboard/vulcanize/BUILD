package(default_visibility = ["//tensorflow:internal"])

licenses(["notice"])  # Apache 2.0

java_binary(
    name = "Vulcanize",
    srcs = ["Vulcanize.java"],
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

filegroup(
    name = "all_files",
    srcs = glob(["**"]),
    tags = ["notsan"],
)
