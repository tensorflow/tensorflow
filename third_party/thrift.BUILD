# Description:
#   Apache Thrift library

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("@org_tensorflow//third_party:common.bzl", "template_rule")

cc_library(
    name = "thrift",
    srcs = [
        "lib/cpp/src/thrift/Thrift.h",
        "lib/cpp/src/thrift/transport/TBufferTransports.h",
        "lib/cpp/src/thrift/transport/TBufferTransports.cpp",
        "lib/cpp/src/thrift/transport/TTransportException.h",
        "lib/cpp/src/thrift/transport/TTransportException.cpp",
        "lib/cpp/src/thrift/protocol/TProtocol.h",
        "lib/cpp/src/thrift/protocol/TProtocol.cpp",
        "lib/cpp/src/thrift/config.h",
        "compiler/cpp/src/thrift/version.h",
    ],
    hdrs = [
    ],
    includes = [
        "lib/cpp/src",
    ],
    copts = [
        "-Iexternal/boost",
        "-Iexternal/thrift/lib/cpp/src",
    ],
    deps = [
        "@boost",
    ],
)

template_rule(
    name = "version_h",
    src = "compiler/cpp/src/thrift/version.h.in",
    out = "compiler/cpp/src/thrift/version.h",
    substitutions = {
        "@PACKAGE_VERSION@": "0.11.0",
    },
)

template_rule(
    name = "config_h",
    src = "build/cmake/config.h.in",
    out = "lib/cpp/src/thrift/config.h",
    substitutions = {
        "#cmakedefine": "#define",
        "${PACKAGE}": "thrift",
        "${PACKAGE_BUGREPORT}": "",
        "${PACKAGE_NAME}": "thrift",
        "${PACKAGE_TARNAME}": "thrift",
        "${PACKAGE_URL}": "",
        "${PACKAGE_VERSION}": "0.11.0",
        "${PACKAGE_STRING}": "thrift 0.11.0",
    },
)
