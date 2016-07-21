"""Build extension for generating C++ header file from an Avro schema.

Example usage:

load("//third_party/avro:build_defs.bzl", "avro_gen_cpp")

avro_gen_cpp(
    name = "myrule",
    srcs = ["myschema.json"],
    outs = ["myschema.h"],
    namespace = "mynamespace",
)
"""

def avro_gen_cpp(name, srcs, outs, namespace, visibility=None):
    native.genrule(
        name = name,
        srcs = srcs,
        outs = outs,
        cmd = ("$(location @avro_archive//:avrogencpp)" +
               " --include-prefix external/avro_archive/avro-cpp-1.8.0/api" +
               " --namespace " + namespace +
               " --no-union-typedef" +
               " --input $(SRCS)" +
               " --output $@"),
        tools = ["@avro_archive//:avrogencpp"],
        visibility = visibility,
    )
