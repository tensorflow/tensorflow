This directory contains models for unsupervised training of word embeddings
using the model described in:

(Mikolov, et. al.) [Efficient Estimation of Word Representations in Vector Space](http://arxiv.org/abs/1301.3781),
ICLR 2013.

Detailed instructions on how to get started and use them are available in the
tutorials. Brief instructions are below.

* [Word2Vec Tutorial](http://tensorflow.org/tutorials/word2vec/index.md)

To download the example text and evaluation data:

```shell
wget http://mattmahoney.net/dc/text8.zip -O text8.zip
unzip text8.zip
wget https://storage.googleapis.com/google-code-archive-source/v2/code.google.com/word2vec/source-archive.zip
unzip -p source-archive.zip  word2vec/trunk/questions-words.txt > questions-words.txt
rm source-archive.zip
```

Assuming you are using the pip package install and have cloned the git
repository, navigate into this directory and run using:

```shell
cd tensorflow/models/embedding
python word2vec_optimized.py \
  --train_data=text8 \
  --eval_data=questions-words.txt \
  --save_path=/tmp/
```

To run the code from sources using bazel:

```shell
bazel run -c opt tensorflow/models/embedding/word2vec_optimized -- \
  --train_data=text8 \
  --eval_data=questions-words.txt \
  --save_path=/tmp/
```

Here is a short overview of what is in this directory.

File | What's in it?
--- | ---
`word2vec.py` | A version of word2vec implemented using TensorFlow ops and minibatching.
`word2vec_test.py` | Integration test for word2vec.
`word2vec_optimized.py` | A version of word2vec implemented using C ops that does no minibatching.
`word2vec_optimized_test.py` | Integration test for word2vec_optimized.
`word2vec_kernels.cc` | Kernels for the custom input and training ops.
`word2vec_ops.cc` | The declarations of the custom ops.
