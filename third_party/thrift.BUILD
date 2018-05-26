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
        "lib/cpp/src/thrift/TLogging.h",
        "lib/cpp/src/thrift/TOutput.h",
        "lib/cpp/src/thrift/TBase.h",
        "lib/cpp/src/thrift/TToString.h",
        "lib/cpp/src/thrift/TApplicationException.h",
        "lib/cpp/src/thrift/transport/PlatformSocket.h",
        "lib/cpp/src/thrift/transport/TTransport.h",
        "lib/cpp/src/thrift/transport/TBufferTransports.h",
        "lib/cpp/src/thrift/transport/TBufferTransports.cpp",
        "lib/cpp/src/thrift/transport/TTransportException.h",
        "lib/cpp/src/thrift/transport/TTransportException.cpp",
        "lib/cpp/src/thrift/transport/TVirtualTransport.h",
        "lib/cpp/src/thrift/protocol/TProtocol.h",
        "lib/cpp/src/thrift/protocol/TProtocol.cpp",
        "lib/cpp/src/thrift/protocol/TProtocolException.h",
        "lib/cpp/src/thrift/protocol/TVirtualProtocol.h",
        "lib/cpp/src/thrift/thrift-config.h",
        "lib/cpp/src/thrift/config.h",
        "lib/cpp/src/thrift/stdcxx.h",
        "compiler/cpp/src/thrift/version.h",
    ],
    hdrs = [
        "lib/cpp/src/thrift/protocol/TCompactProtocol.h",
        "lib/cpp/src/thrift/protocol/TCompactProtocol.tcc",
        "lib/cpp/src/thrift/protocol/TDebugProtocol.h",
        "lib/cpp/src/thrift/protocol/TBinaryProtocol.h",
        "lib/cpp/src/thrift/protocol/TBinaryProtocol.tcc",
    ],
    includes = [
        "lib/cpp/src",
    ],
    copts = [
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
