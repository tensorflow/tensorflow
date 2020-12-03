
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def initialize_rbe():

	http_archive(
	    name = "tf_toolchains",
	    sha256 = "d60f9637c64829e92dac3f4477a2c45cdddb9946c5da0dd46db97765eb9de08e",
	    strip_prefix = "toolchains-1.1.5",
	    urls = [
	        "http://mirror.tensorflow.org/github.com/tensorflow/toolchains/archive/v1.1.5.tar.gz",
	        "https://github.com/tensorflow/toolchains/archive/v1.1.5.tar.gz",
	    ],
	)

	load("@tf_toolchains//toolchains/remote_config:configs.bzl", "initialize_rbe_configs")

	# Loads all external repos to configure RBE builds.
	# Moved to root workspace file to avoid downstream breakages of bazel imports of workspace.bzl.
	initialize_rbe_configs()