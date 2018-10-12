licenses(["restricted"])  # GPLv3

filegroup(
    name = "LICENSE",
    visibility = ["//visibility:public"],
)

filegroup(
    name = "templates",
    visibility = ["//visibility:public"],
)

genrule(
    name = "lnswiglink",
    outs = ["swiglink"],
    cmd = "ln -s $$(which swig) $@",
)

sh_binary(
    name = "swig",
    srcs = ["swiglink"],
    visibility = ["//visibility:public"],
)
