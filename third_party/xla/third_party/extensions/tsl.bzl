"""Module extension for tsl"""

load("//third_party:repo.bzl", "tf_vendored")

tsl_extension = module_extension(
    implementation = lambda mctx: tf_vendored(name = "tsl", path = "third_party/tsl"),
)
