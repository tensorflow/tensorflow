# KDNN build dependency

KDNN v3.1.0 is provided as a Bazel external repository. The default build
downloads and verifies the archive, extracts the RPM, applies the TensorFlow
header adapter, and exposes the threadpool `libkdnn.a` as `@kdnn//:kdnn`.

For an installed package use `KDNN_ROOT=/usr/local/kdnn`. For an offline build
use `KDNN_ARCHIVE=/path/to/BoostKit-boostcore-kdnn_3.1.0.zip`.

Enable the TensorFlow integration with `--define=enable_kdnn=true`.
