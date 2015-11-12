genrule(
    name = "copy_six",
    srcs = ["six-1.10.0/six.py"],
    outs = ["six.py"],
    cmd = "cp $< $(@)",
)

py_library(
    name = "six",
    srcs = ["six.py"],
    visibility = ["//visibility:public"],
    srcs_version = "PY2AND3",
)
