licenses(["notice"])  # BSD/MIT-like license

filegroup(
    name = "LICENSE",
    visibility = ["//visibility:public"],
)


HEADERS = [
  "dnnl_config.h",
  "dnnl_debug.h",
  "dnnl.h",
  "dnnl.hpp",
  "dnnl_types.h",
  "dnnl_version.h",
  "mkldnn_config.h",
  "mkldnn_debug.h",
  "mkldnn_dnnl_mangling.h",
  "mkldnn.h",
  "mkldnn.hpp",
  "mkldnn_types.h",
  "mkldnn_version.h",
]

genrule(
    name = "link_headers",
    outs = HEADERS,
    cmd = """
      for i in $(OUTS); do
        i=$${i##*/}
        ln -sf $(INCLUDEDIR)/mkl-dnn/$$i $(@D)/$$i
      done
    """,
)

cc_library(
    name = "mkl_headers",
    hdrs = HEADERS,
    includes = ["."],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "mkl_dnn",
    hdrs = HEADERS,
    linkopts = ["-lmkldnn"],
    includes = ["."],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "mkl_libs_linux",
    hdrs = HEADERS,
    linkopts = ["-lmkldnn"],
    includes = ["."],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "mkldnn_single_threaded",
    hdrs = HEADERS,
    linkopts = ["-lmkldnn"],
    includes = ["."],
    visibility = ["//visibility:public"],
)
