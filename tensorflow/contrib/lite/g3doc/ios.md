
# Build TensorFlow Lite for iOS

This document describes how to build TensorFlow Lite iOS library. If you just
want to use it, the easiest way is using the TensorFlow Lite CocoaPod releases.
See [TensorFlow Lite iOS Demo](demo_ios.md) for examples.


## Building

To create a universal iOS library for TensorFlow Lite, you need to build it
using Xcode's command line tools on a MacOS machine. If you have not already,
you will need to install Xcode 8 or later and the tools using `xcode-select`:

```bash
xcode-select --install
```

If this is a new install, you will need to run XCode once to agree to the
license before continuing.

(You will also need to have [Homebrew](http://brew.sh/) installed.)

Then install
[automake](https://en.wikipedia.org/wiki/Automake)/[libtool](https://en.wikipedia.org/wiki/GNU_Libtool):

```bash
brew install automake
brew install libtool
```
If you get an error where either automake or libtool install but do not link correctly, you'll first need to:
```bash
sudo chown -R $(whoami) /usr/local/*
```
Then follow the instructions to perform the linking:
```bash
brew link automake
brew link libtool
```

Then you need to run a shell script to download the dependencies you need:

```bash
tensorflow/contrib/lite/tools/make/download_dependencies.sh
```

This will fetch copies of libraries and data from the web and install them in
`tensorflow/contrib/lite/downloads`.

With all of the dependencies set up, you can now build the library for all five
supported architectures on iOS:

```bash
tensorflow/contrib/lite/tools/make/build_ios_universal_lib.sh
```

Under the hood this uses a makefile in `tensorflow/contrib/lite` to build the
different versions of the library, followed by a call to `lipo` to bundle them
into a universal file containing armv7, armv7s, arm64, i386, and x86_64
architectures. The resulting library is in
`tensorflow/contrib/lite/tools/make/gen/lib/libtensorflow-lite.a`.

If you get an error such as `no such file or directory: 'x86_64'` when running 
`build_ios_universal_lib.sh`: open Xcode > Preferences > Locations, and ensure 
a value is selected in the "Command Line Tools" dropdown.

## Using in your own application

You'll need to update various settings in your app to link against TensorFlow
Lite. You can view them in the example project at
`tensorflow/contrib/lite/examples/ios/simple/simple.xcodeproj` but here's a full
rundown:

-   You'll need to add the library at
    `tensorflow/contrib/lite/gen/lib/libtensorflow-lite.a` to your linking build
    stage, and in Search Paths add `tensorflow/contrib/lite/gen/lib` to the
    Library Search Paths setting.

-   The _Header Search_ paths needs to contain:

    -   the root folder of tensorflow,
    -   `tensorflow/contrib/lite/downloads`
    -   `tensorflow/contrib/lite/downloads/flatbuffers/include`

-   C++11 support (or later) should be enabled by setting `C++ Language Dialect`
    to `GNU++11` (or `GNU++14`), and `C++ Standard Library` to `libc++`.
