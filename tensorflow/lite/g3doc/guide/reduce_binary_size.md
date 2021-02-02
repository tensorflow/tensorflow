# Reduce TensorFlow Lite binary size

## Overview

When deploying models for on-device machine learning (ODML) applications, it is
important to be aware of the limited memory that is available on mobile devices.
Model binary sizes are closely correlated to the number of ops used in the
model. TensorFlow Lite enables you to reduce model binary sizes by using
selective builds. Selective builds skip unused operations in your model set and
produce a compact library with just the runtime and the op kernels required for
the model to run on your mobile device.

Selective build applies on the following three operations libraries.

1.  [TensorFlow Lite built-in ops library](https://www.tensorflow.org/lite/guide/ops_compatibility)
1.  [TensorFlow Lite custom ops](https://www.tensorflow.org/lite/guide/ops_custom)
1.  [Select TensorFlow ops library](https://www.tensorflow.org/lite/guide/ops_select)

The table below demonstrates the impact of selective builds for some common use
cases:

<table>
  <thead>
    <tr>
      <th>Model Name</th>
      <th>Domain</th>
      <th>Target architecture</th>
      <th>AAR file size(s)</th>
    </tr>
  </thead>
  <tr>
    <td rowspan = 2>
      <a href="https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz">Mobilenet_1.0_224(float)</a>
    </td>
    <td rowspan = 2>Image classification</td>
    <td>armeabi-v7a</td>
    <td>tensorflow-lite.aar (296,635 bytes)</td>
  </tr>
   <tr>
    <td>arm64-v8a</td>
    <td>tensorflow-lite.aar (382,892 bytes)</td>
  </tr>
  <tr>
    <td rowspan = 2>
      <a href="https://tfhub.dev/google/lite-model/spice/">SPICE</a>
    </td>
    <td rowspan = 2>Sound pitch extraction</td>
    <td>armeabi-v7a</td>
    <td>tensorflow-lite.aar (375,813 bytes)<br />tensorflow-lite-select-tf-ops.aar (1,676,380 bytes)</td>
  </tr>
   <tr>
    <td>arm64-v8a</td>
    <td>tensorflow-lite.aar (421,826 bytes)<br />tensorflow-lite-select-tf-ops.aar (2,298,630 bytes)</td>
  </tr>
  <tr>
    <td rowspan = 2>
      <a href="https://tfhub.dev/deepmind/i3d-kinetics-400/1">i3d-kinetics-400</a>
    </td>
    <td rowspan = 2>Video classification</td>
    <td>armeabi-v7a</td>
    <td>tensorflow-lite.aar (240,085 bytes)<br />tensorflow-lite-select-tf-ops.aar (1,708,597 bytes)</td>
  </tr>
   <tr>
    <td>arm64-v8a</td>
    <td>tensorflow-lite.aar (273,713 bytes)<br />tensorflow-lite-select-tf-ops.aar (2,339,697 bytes)</td>
  </tr>
 </table>

Note: This feature is currently experimental and available since version 2.4 and
may change.

## Known issues/limitations

1.  Selective Build for C API and iOS version is not supported currently.

## Selectively build TensorFlow Lite with Bazel

This section assumes that you have downloaded TensorFlow source codes and
[set up the local development environment](https://www.tensorflow.org/lite/guide/build_android#set_up_build_environment_without_docker)
to Bazel.

### Build AAR files for Android project

You can build the custom TensorFlow Lite AARs by providing your model file paths
as follows.

```sh
sh tensorflow/lite/tools/build_aar.sh \
  --input_models=/a/b/model_one.tflite,/c/d/model_two.tflite \
  --target_archs=x86,x86_64,arm64-v8a,armeabi-v7a
```

The above command will generate the AAR file `bazel-bin/tmp/tensorflow-lite.aar`
for TensorFlow Lite built-in and custom ops; and optionally, generates the aar
file `bazel-bin/tmp/tensorflow-lite-select-tf-ops.aar` if your models contain
Select TensorFlow ops. Note that this builds a "fat" AAR with several different
architectures; if you don't need all of them, use the subset appropriate for
your deployment environment.

### Advanced Usage: Build with custom ops

If you have developed Tensorflow Lite models with custom ops, you can build them
by adding the following flags to the build command:

```sh
sh tensorflow/lite/tools/build_aar.sh \
  --input_models=/a/b/model_one.tflite,/c/d/model_two.tflite \
  --target_archs=x86,x86_64,arm64-v8a,armeabi-v7a \
  --tflite_custom_ops_srcs=/e/f/file1.cc,/g/h/file2.h \
  --tflite_custom_ops_deps=dep1,dep2
```

The `tflite_custom_ops_srcs` flag contains source files of your custom ops and
the `tflite_custom_ops_deps` flag contains dependencies to build those source
files. Note that these dependencies must exist in the TensorFlow repo.

## Selectively Build TensorFlow Lite with Docker

This section assumes that you have installed
[Docker](https://docs.docker.com/get-docker/) on your local machine and
downloaded the TensorFlow Lite Dockerfile
[here](https://www.tensorflow.org/lite/guide/build_android#set_up_build_environment_using_docker).

After downloading the above Dockerfile, you can build the docker image by
running:

```shell
docker build . -t tflite-builder -f tflite-android.Dockerfile
```

### Build AAR files for Android project

Download the script for building with Docker by running:

```sh
curl -o build_aar_with_docker.sh \
  https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/lite/tools/build_aar_with_docker.sh &&
chmod +x build_aar_with_docker.sh
```

Then, you can build the custom TensorFlow Lite AAR by providing your model file
paths as follows.

```sh
sh build_aar_with_docker.sh \
  --input_models=/a/b/model_one.tflite,/c/d/model_two.tflite \
  --target_archs=x86,x86_64,arm64-v8a,armeabi-v7a \
  --checkpoint=master \
  [--cache_dir=<path to cache directory>]
```

The `checkpoint` flag is a commit, a branch or a tag of the TensorFlow repo that
you want to checkout before building the libraries; by default it is the latest
release branch. The above command will generate the AAR file
`tensorflow-lite.aar` for TensorFlow Lite built-in and custom ops and optionally
the AAR file `tensorflow-lite-select-tf-ops.aar` for Select TensorFlow ops in
your current directory.

The --cache_dir specify the cache directory. If not provided, the script will
create a directory named `bazel-build-cache` under current working directory for
caching.

## Add AAR files to project

Add AAR files by directly
[importing the AAR into your project](https://www.tensorflow.org/lite/guide/build_android#add_aar_directly_to_project),
or by
[publishing the custom AAR to your local Maven repository](https://www.tensorflow.org/lite/guide/build_android#install_aar_to_local_maven_repository).
Note that you have to add the AAR files for `tensorflow-lite-select-tf-ops.aar`
as well if you generate it.
