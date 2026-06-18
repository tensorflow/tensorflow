licenses(["notice"])  # Apache-2.0

genrule(
    name = "lncython",
    outs = ["cython"],
    cmd = "ln -s $$(which cython) $@",
)

sh_binary(
    name = "cython_binary",
    srcs = ["cython"],
    visibility = ["//visibility:public"],
)
