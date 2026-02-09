load("//third_party/py/tensorflow_docs/google:tf_org.bzl", "tf_org_check_links", "tf_org_notebook_test")

licenses(["notice"])

tf_org_check_links(name = "check_links")

# We couldn't get internal notebook tests working for these:
# Usually the reason is the use of external data.
#
# * tokenizers.ipynb
# * word_embeddings.ipynb
# * subwords_tokenizer.ipynb
# * decoding_api.ipynb

tf_org_notebook_test(
    name = "unicode",
    ipynb = "unicode.ipynb",
    deps = [],
)

tf_org_notebook_test(
    name = "bert_preprocessing_guide",
    ipynb = "bert_preprocessing_guide.ipynb",
    deps = [
        "//third_party/py/tensorflow_text",
    ],
)
