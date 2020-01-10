licenses(["unencumbered"])  # Public Domain

# Production build of SQLite library that's baked into TensorFlow.
cc_library(
    name = "org_sqlite",
    linkopts = ["-lsqlite3"],
    visibility = ["//visibility:public"],
)

# This is a Copybara sync helper for Google.
py_library(
    name = "python",
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
)
