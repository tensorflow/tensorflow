load("//third_party/py/tensorflow_docs/google:tf_org.bzl", "tf_org_check_links", "tf_org_notebook_test")

licenses(["notice"])

tf_org_check_links(name = "check_links")

# We couldn't get internal notebook tests working for these:
# Usually the reason is the use of external data.
#
# * bert_glue.ipynb
# * classify_text_with_bert.ipynb
# * nmt_with_attention.ipynb
# * fine_tune_bert.ipynb
# * uncertainty_quantification_with_sngp_bert.ipynb

tf_org_notebook_test(
    name = "text_generation",
    ipynb = "text_generation.ipynb",
    tags = ["requires-net:external"],
    deps = [
        "//third_party/py/matplotlib",
        "@pypi//numpy:pkg",
        #  tensorflow datasets dep,
    ],
)

tf_org_notebook_test(
    name = "text_similarity",
    ipynb = "text_similarity.ipynb",
    deps = [
        "//third_party/py/tensorflow_text",
    ],
)

tf_org_notebook_test(
    name = "transformer",
    size = "enormous",
    ipynb = "transformer.ipynb",
    tags = ["requires-net:external"],
    deps = [
        "//third_party/py/matplotlib",
        "@pypi//numpy:pkg",
        #  tensorflow datasets dep,
        "//third_party/py/tensorflow_text",
    ],
)

tf_org_notebook_test(
    name = "text_classification_rnn",
    size = "large",
    ipynb = "text_classification_rnn.ipynb",
    tags = ["requires-net:external"],
    deps = [
        "//third_party/py/matplotlib",
        "@pypi//numpy:pkg",
        #  tensorflow datasets dep,
    ],
)
