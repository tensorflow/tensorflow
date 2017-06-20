workspace(name = "org_tensorflow_tensorboard")

http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "bc41b80486413aaa551860fc37471dbc0666e1dbb5236fb6177cb83b0c105846",
    strip_prefix = "rules_closure-dec425a4ff3faf09a56c85d082e4eed05d8ce38f",
    urls = [
        "http://mirror.bazel.build/github.com/bazelbuild/rules_closure/archive/dec425a4ff3faf09a56c85d082e4eed05d8ce38f.tar.gz",  # 2017-06-02
        "https://github.com/bazelbuild/rules_closure/archive/dec425a4ff3faf09a56c85d082e4eed05d8ce38f.tar.gz",
    ],
)


load("@io_bazel_rules_closure//closure:defs.bzl", "closure_repositories")

closure_repositories()

load("//third_party:workspace.bzl", "tensorboard_workspace")

# Please add all new dependencies in workspace.bzl.
tensorboard_workspace()
