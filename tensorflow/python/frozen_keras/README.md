# DO NOT USE

Everything under this package is for internal usage, and only serves a
dependency from legacy TF v1 APIs that relies on Keras. Any active development
should happen in third_party/tensorflow/python/keras instead.

## Background

In order to build a more modular Tensorflow and Keras, we decided to split the
Keras code into its own repository. Having TensorFlow depend on
Keras is a red flag as it is a reverse dependency. As some legacy TF V1 APIs
are using Keras classes as base classes, like `Layer`, we decided to keep a copy
of the trimmed Keras code to resolve the reverse dependency. This will also
ensure the stability of the TF V1 API will be not affected by the active
development of the Keras project.
