# Build TensorFlow Lite for iOS

This document describes how to build TensorFlow Lite iOS library on your own.
Normally, you do not need to locally build TensorFlow Lite iOS library. If you
just want to use it, the easiest way is using the prebuilt stable or nightly
releases of the TensorFlow Lite CocoaPods. See [iOS quickstart](ios.md) for more
details on how to use them in your iOS projects.

## Building locally

In some cases, you might wish to use a local build of TensorFlow Lite, for
example when you want to make local changes to TensorFlow Lite and test those
changes in your iOS app or you prefer using static framework to our provided
dynamic one. To create a universal iOS framework for TensorFlow Lite locally,
you need to build it using Bazel on a macOS machine.

### Install Xcode

If you have not already, you will need to install Xcode 8 or later and the tools
using `xcode-select`:

```sh
xcode-select --install
```

If this is a new install, you will need to accept the license agreement for all
users with the following command:

```sh
sudo xcodebuild -license accept
```

### Install Bazel

Bazel is the primary build system for TensorFlow. Install Bazel as per the
[instructions on the Bazel website][bazel-install]. Make sure to choose a
version between `_TF_MIN_BAZEL_VERSION` and `_TF_MAX_BAZEL_VERSION` in
[`configure.py` file][configure-py] at the root of `tensorflow` repository.

### Configure WORKSPACE and .bazelrc

Run the `./configure` script in the root TensorFlow checkout directory, and
answer "Yes" when the script asks if you wish to build TensorFlow with iOS
support.

### Build TensorFlowLiteC dynamic framework (recommended)

Note: This step is not necessary if (1) you are using Bazel for your app, or (2)
you only want to test local changes to the Swift or Objective-C APIs. In these
cases, skip to the [Use in your own application](#use_in_your_own_application)
section below.

Once Bazel is properly configured with iOS support, you can build the
`TensorFlowLiteC` framework with the following command.

```sh
bazel build --config=ios_fat -c opt --cxxopt=--std=c++17 \
  //tensorflow/lite/ios:TensorFlowLiteC_framework
```

This command will generate the `TensorFlowLiteC_framework.zip` file under
`bazel-bin/tensorflow/lite/ios/` directory under your TensorFlow root directory.
By default, the generated framework contains a "fat" binary, containing armv7,
arm64, and x86_64 (but no i386). To see the full list of build flags used when
you specify `--config=ios_fat`, please refer to the iOS configs section in the
[`.bazelrc` file][bazelrc].

### Build TensorFlowLiteC static framework

By default, we only distribute the dynamic framework via Cocoapods. If you want
to use the static framework instead, you can build the `TensorFlowLiteC` static
framework with the following command:

```
bazel build --config=ios_fat -c opt --cxxopt=--std=c++17 \
  //tensorflow/lite/ios:TensorFlowLiteC_static_framework
```

The command will generate a file named `TensorFlowLiteC_static_framework.zip`
under `bazel-bin/tensorflow/lite/ios/` directory under your TensorFlow root
directory. This static framework can be used in the exact same way as the
dynamic one.

### Selectively build TFLite frameworks

You can build smaller frameworks targeting only a set of models using selective
build, which will skip unused operations in your model set and only include the
op kernels required to run the given set of models. The command is as following:

```sh
bash tensorflow/lite/ios/build_frameworks.sh \
  --input_models=model1.tflite,model2.tflite \
  --target_archs=x86_64,armv7,arm64
```

The above command will generate the static framework
`bazel-bin/tensorflow/lite/ios/tmp/TensorFlowLiteC_framework.zip` for TensorFlow
Lite built-in and custom ops; and optionally, generates the static framework
`bazel-bin/tensorflow/lite/ios/tmp/TensorFlowLiteSelectTfOps_framework.zip` if
your models contain Select TensorFlow ops. Note that the `--target_archs` flag
can be used to specify your deployment architectures.

## Use in your own application

### CocoaPods developers

There are three CocoaPods for TensorFlow Lite:

*   `TensorFlowLiteSwift`: Provides the Swift APIs for TensorFlow Lite.
*   `TensorFlowLiteObjC`: Provides the Objective-C APIs for TensorFlow Lite.
*   `TensorFlowLiteC`: Common base pod, which embeds the TensorFlow Lite core
    runtime and exposes the base C APIs used by the above two pods. Not meant to
    be directly used by users.

As a developer, you should choose either `TensorFlowLiteSwift` or
`TensorFlowLiteObjC` pod based on the language in which your app is written, but
not both. The exact steps for using local builds of TensorFlow Lite differ,
depending on which exact part you would like to build.

#### Using local Swift or Objective-C APIs

If you are using CocoaPods, and only wish to test some local changes to the
TensorFlow Lite's [Swift APIs][swift-api] or [Objective-C APIs][objc-api],
follow the steps here.

1.  Make changes to the Swift or Objective-C APIs in your `tensorflow` checkout.

1.  Open the `TensorFlowLite(Swift|ObjC).podspec` file, and update this line: \
    `s.dependency 'TensorFlowLiteC', "#{s.version}"` \
    to be: \
    `s.dependency 'TensorFlowLiteC', "~> 0.0.1-nightly"` \
    This is to ensure that you are building your Swift or Objective-C APIs
    against the latest available nightly version of `TensorFlowLiteC` APIs
    (built every night between 1-4AM Pacific Time) rather than the stable
    version, which may be outdated compared to your local `tensorflow` checkout.
    Alternatively, you could choose to publish your own version of
    `TensorFlowLiteC` and use that version (see
    [Using local TensorFlow Lite core](#using_local_tensorflow_lite_core)
    section below).

1.  In the `Podfile` of your iOS project, change the dependency as follows to
    point to the local path to your `tensorflow` root directory. \
    For Swift: \
    `pod 'TensorFlowLiteSwift', :path => '<your_tensorflow_root_dir>'` \
    For Objective-C: \
    `pod 'TensorFlowLiteObjC', :path => '<your_tensorflow_root_dir>'`

1.  Update your pod installation from your iOS project root directory. \
    `$ pod update`

1.  Reopen the generated workspace (`<project>.xcworkspace`) and rebuild your
    app within Xcode.

#### Using local TensorFlow Lite core

You can set up a private CocoaPods specs repository, and publish your custom
`TensorFlowLiteC` framework to your private repo. You can copy this
[podspec file][tflite-podspec] and modify a few values:

```ruby
  ...
  s.version      = <your_desired_version_tag>
  ...
  # Note the `///`, two from the `file://` and one from the `/path`.
  s.source       = { :http => "file:///path/to/TensorFlowLiteC_framework.zip" }
  ...
  s.vendored_frameworks = 'TensorFlowLiteC.framework'
  ...
```

After creating your own `TensorFlowLiteC.podspec` file, you can follow the
[instructions on using private CocoaPods][private-cocoapods] to use it in your
own project. You can also modify the `TensorFlowLite(Swift|ObjC).podspec` to
point to your custom `TensorFlowLiteC` pod and use either Swift or Objective-C
pod in your app project.

### Bazel developers

If you are using Bazel as the main build tool, you can simply add
`TensorFlowLite` dependency to your target in your `BUILD` file.

For Swift:

```python
swift_library(
  deps = [
      "//tensorflow/lite/swift:TensorFlowLite",
  ],
)
```

For Objective-C:

```python
objc_library(
  deps = [
      "//tensorflow/lite/objc:TensorFlowLite",
  ],
)
```

When you build your app project, any changes to the TensorFlow Lite library will
be picked up and built into your app.

### Modify Xcode project settings directly

It is highly recommended to use CocoaPods or Bazel for adding TensorFlow Lite
dependency into your project. If you still wish to add `TensorFlowLiteC`
framework manually, you'll need to add the `TensorFlowLiteC` framework as an
embedded framework to your application project. Unzip the
`TensorFlowLiteC_framework.zip` generated from the above build to get the
`TensorFlowLiteC.framework` directory. This directory is the actual framework
which Xcode can understand.

Once you've prepared the `TensorFlowLiteC.framework`, first you need to add it
as an embedded binary to your app target. The exact project settings section for
this may differ depending on your Xcode version.

*   Xcode 11: Go to the 'General' tab of the project editor for your app target,
    and add the `TensorFlowLiteC.framework` under 'Frameworks, Libraries, and
    Embedded Content' section.
*   Xcode 10 and below: Go to the 'General' tab of the project editor for your
    app target, and add the `TensorFlowLiteC.framework` under 'Embedded
    Binaries'. The framework should also be added automatically under 'Linked
    Frameworks and Libraries' section.

When you add the framework as an embedded binary, Xcode would also update the
'Framework Search Paths' entry under 'Build Settings' tab to include the parent
directory of your framework. In case this does not happen automatically, you
should manually add the parent directory of the `TensorFlowLiteC.framework`
directory.

Once these two settings are done, you should be able to import and call the
TensorFlow Lite's C API, defined by the header files under
`TensorFlowLiteC.framework/Headers` directory.

[bazel-install]: https://docs.bazel.build/versions/master/install-os-x.html
[bazelrc]: https://github.com/tensorflow/tensorflow/blob/master/.bazelrc
[configure-py]: https://github.com/tensorflow/tensorflow/blob/master/configure.py
[objc-api]: https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/objc
[private-cocoapods]: https://guides.cocoapods.org/making/private-cocoapods.html
[swift-api]: https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/swift
[tflite-podspec]: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/ios/TensorFlowLiteC.podspec
