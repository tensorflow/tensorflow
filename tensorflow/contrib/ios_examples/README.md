# TensorFlow iOS Examples

This folder contains examples of how to build applications for iOS devices using TensorFlow.

## Building the Examples

 - You'll need Xcode 7.3 or later, with the command-line tools installed.

 - Follow the instructions at
   [tensorflow/contrib/makefile](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/makefile)
   under "iOS" to compile a static library containing the core TensorFlow code.

 - Download
   [Inception v1](https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip),
   and extract the label and graph files into the data folders inside both the
   simple and camera examples:

```bash
mkdir -p ~/graphs
curl -o ~/graphs/inception5h.zip \
 https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip \
 && unzip ~/graphs/inception5h.zip -d ~/graphs/inception5h
cp ~/graphs/inception5h/* tensorflow/contrib/ios_examples/benchmark/data/
cp ~/graphs/inception5h/* tensorflow/contrib/ios_examples/camera/data/
cp ~/graphs/inception5h/* tensorflow/contrib/ios_examples/simple/data/
```

 - Load the Xcode project inside the `simple` subfolder, and press Command-R to
   build and run it on the simulator or your connected device.

 - You should see a single-screen app with a "Run Model" button. Tap that, and
   you should see some debug output appear below indicating that the example
   Grace Hopper image has been analyzed, with a military uniform recognized.
 
 - Once you have success there, make sure you have a real device connected and
   open up the Xcode project in the camera subfolder. Once you build and run
   that, you should get a live camera view that you can point at objects to get
   real-time recognition results.
 
## Troubleshooting

If you're hitting problems, here's a checklist of common things to investigate:

 - Make sure that you've run the `build_all_ios.sh` script 
   This will run `download_dependencies.sh`,`compile_ios_protobuf.sh` and `compile_ios_tensorflow.sh`.
   (check each one if they have run successful.)
 
 - Check that you have version 7.3 of Xcode.
 
 - If there's a complaint about no Sessions registered, that means that the C++
   global constructors that TensorFlow relies on for registration haven't been
   linked in properly. You'll have to make sure your project uses force_load, as
   described below.
 
## Creating your Own App

You'll need to update various settings in your app to link against
TensorFlow. You can view them in the example projects, but here's a full
rundown:

 - The `compile_ios_tensorflow.sh` script builds a universal static library in
   `tensorflow/contrib/makefile/gen/lib/libtensorflow-core.a`. You'll need to add
   this to your linking build stage, and in Search Paths add
   `tensorflow/contrib/makefile/gen/lib` to the Library Search Paths setting.
 
 - You'll also need to add `libprotobuf.a` and `libprotobuf-lite.a` from
   `tensorflow/contrib/makefile/gen/protobuf_ios/lib` to your _Build Stages_ and
   _Library Search Paths_.
 
 - The _Header Search_ paths needs to contain:
   - the root folder of tensorflow,
   - `tensorflow/contrib/makefile/downloads/protobuf/src`
   - `tensorflow/contrib/makefile/downloads`,
   - `tensorflow/contrib/makefile/downloads/eigen`, and
   - `tensorflow/contrib/makefile/gen/proto`.

 - In the Linking section, you need to add `-force_load` followed by the path to
   the TensorFlow static library in the _Other Linker_ Flags section. This ensures
   that the global C++ objects that are used to register important classes
   inside the library are not stripped out. To the linker, they can appear
   unused because no other code references the variables, but in fact their
   constructors have the important side effect of registering the class.

 - You'll need to include the Accelerate framework in the "Link Binary with
   Libraries" build phase of your project.
 
 - C++11 support (or later) should be enabled by setting `C++ Language Dialect` to
   `GNU++11` (or `GNU++14`), and `C++ Standard Library` to `libc++`.
 
 - The library doesn't currently support bitcode, so you'll need to disable that
   in your project settings.

 - Remove any use of the `-all_load` flag in your project. The protocol buffers
   libraries (full and lite versions) contain duplicate symbols, and the `-all_load`
   flag will cause these duplicates to become link errors. If you were using
   `-all_load` to avoid issues with Objective-C categories in static libraries,
   you may be able to replace it with the `-ObjC` flag.

## Reducing the binary size

TensorFlow is a comparatively large library for a mobile device, so it will
increase the size of your app. Currently on iOS we see around a 11 MB binary
footprint per CPU architecture, though we're actively working on reducing that.
It can be tricky to set up the right configuration in your own app to keep the
size minimized, so if you do run into this issue we recommend you start by
looking at the simple example to examine its size. Here's how you do that:

 - Open the Xcode project in tensorflow/contrib/ios_examples/simple.

 - Make sure you've followed the steps above to get the data files.

 - Choose "Generic iOS Device" as the build configuration.

 - Select Product->Build.

 - Once the build's complete, open the Report Navigator and select the logs.

 - Near the bottom, you'll see a line saying "Touch tf_ios_makefile_example.app".

 - Expand that line using the icon on the right, and copy the first argument to
   the Touch command.

 - Go to the terminal, type `ls -lah ` and then paste the path you copied.

 - For example it might look like `ls -lah /Users/petewarden/Library/Developer/Xcode/DerivedData/tf_ios_makefile_example-etdbksqytcnzeyfgdwiihzkqpxwr/Build/Products/Debug-iphoneos/tf_ios_makefile_example.app`

 - Running this command will show the size of the executable as the
   `tf_ios_makefile_example` line.

Right now you'll see a size of around 23 MB, since it's including two
architectures (armv7 and arm64). As a first step, you should make sure the size
increase you see in your own app is similar, and if it's larger, look at the
"Other Linker Flags" used in the Simple Xcode project settings to strip the
executable.

After that, you can manually look at modifying the list of kernels
included in tensorflow/contrib/makefile/tf_op_files.txt to reduce the number of
implementations to the ones you're actually using in your own model. We're
hoping to automate this step in the future, but for now manually removing them
is the best approach.
