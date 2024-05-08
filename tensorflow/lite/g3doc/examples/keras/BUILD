load("//third_party/py/tensorflow_docs/google:tf_org.bzl", "tf_org_check_links", "tf_org_notebook_strict_test")

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

licenses(["notice"])

tf_org_check_links(name = "check_links")

tf_org_notebook_strict_test(
    name = "jax_backend_resnet50_example",
    size = "medium",
    ipynb = "keras_jax_backend_to_tfl.ipynb",
    tags = [
        "requires-net:external",
    ],
    deps = [
        "//third_party/py/PIL:pil",
        "//third_party/py/keras",
        "//third_party/py/numpy",
        "//third_party/py/requests",
    ],
)
