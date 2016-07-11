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
   simple and camera examples.

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

 - Make sure that you've run the `download_dependencies.sh` and
   `compile_ios_protobuf.sh` scripts before you run `compile_ios_tensorflow`.
   (These should be called by `build_all_ios.sh` if you are using it, but check
   if they have run successful.)
 
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
   - `tensorflow/contrib/makefile/downloads/eigen-latest`, and
   - `tensorflow/contrib/makefile/gen/proto`.

 - In the Linking section, you need to add `-force_load` followed by the path to
   the TensorFlow static library in the _Other Linker_ Flags section. This ensures
   that the global C++ objects that are used to register important classes
   inside the library are not stripped out. To the linker, they can appear
   unused because no other code references the variables, but in fact their
   constructors have the important side effect of registering the class.
 
 - C++11 support (or later) should be enabled by setting `C++ Language Dialect` to
   `GNU++11` (or `GNU++14`), and `C++ Standard Library` to `libc++`.
 
 - The library doesn't currently support bitcode, so you'll need to disable that
   in your project settings.

 - Remove any use of the `-all_load` flag in your project. The protocol buffers
   libraries (full and lite versions) contain duplicate symbols, and the `-all_load`
   flag will cause these duplicates to become link errors. If you were using
   `-all_load` to avoid issues with Objective-C categories in static libraries,
   you may be able to replace it with the `-ObjC` flag.
