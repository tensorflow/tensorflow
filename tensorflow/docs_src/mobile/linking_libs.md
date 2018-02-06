# Integrating TensorFlow libraries

Once you have made some progress on a model that addresses the problem you’re
trying to solve, it’s important to test it out inside your application
immediately. There are often unexpected differences between your training data
and what users actually encounter in the real world, and getting a clear picture
of the gap as soon as possible improves the product experience.

This page talks about how to integrate the TensorFlow libraries into your own
mobile applications, once you have already successfully built and deployed the
TensorFlow mobile demo apps.

## Linking the library

After you've managed to build the examples, you'll probably want to call
TensorFlow from one of your existing applications. The very easiest way to do
this is to use the Pod installation steps described
@{$mobile/ios_build#using_cocoapods$here}, but if you want to build TensorFlow
from source (for example to customize which operators are included) you'll need
to break out TensorFlow as a framework, include the right header files, and link
against the built libraries and dependencies.

### Android

For Android, you just need to link in a Java library contained in a JAR file
called `libandroid_tensorflow_inference_java.jar`. There are three ways to
include this functionality in your program:

1. Include the jcenter AAR which contains it, as in this
 [example app](https://github.com/googlecodelabs/tensorflow-for-poets-2/blob/master/android/build.gradle#L59-L65)

2. Download the nightly precompiled version from
[ci.tensorflow.org](http://ci.tensorflow.org/view/Nightly/job/nightly-android/lastSuccessfulBuild/artifact/out/).

3. Build the JAR file yourself using the instructions [in our Android Github repo](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/android)

### iOS

Pulling in the TensorFlow libraries on iOS is a little more complicated. Here is
a checklist of what you’ll need to do to your iOS app:

- Link against tensorflow/contrib/makefile/gen/lib/libtensorflow-core.a, usually
  by adding `-L/your/path/tensorflow/contrib/makefile/gen/lib/` and
  `-ltensorflow-core` to your linker flags.

- Link against the generated protobuf libraries by adding
  `-L/your/path/tensorflow/contrib/makefile/gen/protobuf_ios/lib` and
  `-lprotobuf` and `-lprotobuf-lite` to your command line.

- For the include paths, you need the root of your TensorFlow source folder as
  the first entry, followed by
  `tensorflow/contrib/makefile/downloads/protobuf/src`,
  `tensorflow/contrib/makefile/downloads`,
  `tensorflow/contrib/makefile/downloads/eigen`, and
  `tensorflow/contrib/makefile/gen/proto`.

- Make sure your binary is built with `-force_load` (or the equivalent on your
  platform), aimed at the TensorFlow library to ensure that it’s linked
  correctly. More detail on why this is necessary can be found in the next
  section, [Global constructor magic](#global_constructor_magic). On Linux-like
  platforms, you’ll need different flags, more like
  `-Wl,--allow-multiple-definition -Wl,--whole-archive`.

You’ll also need to link in the Accelerator framework, since this is used to
speed up some of the operations.

## Global constructor magic

One of the subtlest problems you may run up against is the “No session factory
registered for the given session options” error when trying to call TensorFlow
from your own application. To understand why this is happening and how to fix
it, you need to know a bit about the architecture of TensorFlow.

The framework is designed to be very modular, with a thin core and a large
number of specific objects that are independent and can be mixed and matched as
needed. To enable this, the coding pattern in C++ had to let modules easily
notify the framework about the services they offer, without requiring a central
list that has to be updated separately from each implementation. It also had to
allow separate libraries to add their own implementations without needing a
recompile of the core.

To achieve this capability, TensorFlow uses a registration pattern in a lot of
places. In the code, it looks like this:

    class MulKernel : OpKernel {
      Status Compute(OpKernelContext* context) { … }
    };
    REGISTER_KERNEL(MulKernel, “Mul”);

This would be in a standalone `.cc` file linked into your application, either
as part of the main set of kernels or as a separate custom library. The magic
part is that the `REGISTER_KERNEL()` macro is able to inform the core of
TensorFlow that it has an implementation of the Mul operation, so that it can be
called in any graphs that require it.

From a programming point of view, this setup is very convenient. The
implementation and registration code live in the same file, and adding new
implementations is as simple as compiling and linking it in. The difficult part
comes from the way that the `REGISTER_KERNEL()` macro is implemented. C++
doesn’t offer a good mechanism for doing this sort of registration, so we have
to resort to some tricky code. Under the hood, the macro is implemented so that
it produces something like this:

    class RegisterMul {
     public:
      RegisterMul() {
        global_kernel_registry()->Register(“Mul”, [](){
          return new MulKernel()
        });
      }
    };
    RegisterMul g_register_mul;

This sets up a class `RegisterMul` with a constructor that tells the global
kernel registry what function to call when somebody asks it how to create a
“Mul” kernel. Then there’s a global object of that class, and so the constructor
should be called at the start of any program.

While this may sound sensible, the unfortunate part is that the global object
that’s defined is not used by any other code, so linkers not designed with this
in mind will decide that it can be deleted. As a result, the constructor is
never called, and the class is never registered. All sorts of modules use this
pattern in TensorFlow, and it happens that `Session` implementations are the
first to be looked for when the code is run, which is why it shows up as the
characteristic error when this problem occurs.

The solution is to force the linker to not strip any code from the library, even
if it believes it’s unused. On iOS, this step can be accomplished with the
`-force_load` flag, specifying a library path, and on Linux you need
`--whole-archive`. These persuade the linker to not be as aggressive about
stripping, and should retain the globals.

The actual implementation of the various `REGISTER_*` macros is a bit more
complicated in practice, but they all suffer the same underlying problem. If
you’re interested in how they work, [op_kernel.h](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/op_kernel.h#L1091)
is a good place to start investigating.

## Protobuf problems

TensorFlow relies on
the [Protocol Buffer](https://developers.google.com/protocol-buffers/) library,
commonly known as protobuf. This library takes definitions of data structures
and produces serialization and access code for them in a variety of
languages. The tricky part is that this generated code needs to be linked
against shared libraries for the exact same version of the framework that was
used for the generator. This can be an issue when `protoc`, the tool used to
generate the code, is from a different version of protobuf than the libraries in
the standard linking and include paths. For example, you might be using a copy
of `protoc` that was built locally in `~/projects/protobuf-3.0.1.a`, but you have
libraries installed at `/usr/local/lib` and `/usr/local/include` that are from
3.0.0.

The symptoms of this issue are errors during the compilation or linking phases
with protobufs. Usually, the build tools take care of this, but if you’re using
the makefile, make sure you’re building the protobuf library locally and using
it, as shown in [this Makefile](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/makefile/Makefile#L18).

Another situation that can cause problems is when protobuf headers and source
files need to be generated as part of the build process. This process makes
building more complex, since the first phase has to be a pass over the protobuf
definitions to create all the needed code files, and only after that can you go
ahead and do a build of the library code.

### Multiple versions of protobufs in the same app

Protobufs generate headers that are needed as part of the C++ interface to the
overall TensorFlow library. This complicates using the library as a standalone
framework.

If your application is already using version 1 of the protocol buffers library,
you may have trouble integrating TensorFlow because it requires version 2. If
you just try to link both versions into the same binary, you’ll see linking
errors because some of the symbols clash. To solve this particular problem, we
have an experimental script at [rename_protobuf.sh](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/makefile/rename_protobuf.sh).

You need to run this as part of the makefile build, after you’ve downloaded all
the dependencies:

    tensorflow/contrib/makefile/download_dependencies.sh
    tensorflow/contrib/makefile/rename_protobuf.sh

## Calling the TensorFlow API

Once you have the framework available, you then need to call into it. The usual
pattern is that you first load your model, which represents a preset set of
numeric computations, and then you run inputs through that model (for example,
images from a camera) and receive outputs (for example, predicted labels).

On Android, we provide the Java Inference Library that is focused on just this
use case, while on iOS and Raspberry Pi you call directly into the C++ API.

### Android

Here’s what a typical Inference Library sequence looks like on Android:

    // Load the model from disk.
    TensorFlowInferenceInterface inferenceInterface =
    new TensorFlowInferenceInterface(assetManager, modelFilename);

    // Copy the input data into TensorFlow.
    inferenceInterface.feed(inputName, floatValues, 1, inputSize, inputSize, 3);

    // Run the inference call.
    inferenceInterface.run(outputNames, logStats);

    // Copy the output Tensor back into the output array.
    inferenceInterface.fetch(outputName, outputs);

You can find the source of this code in the [Android examples](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/android/src/org/tensorflow/demo/TensorFlowImageClassifier.java#L107).

### iOS and Raspberry Pi

Here’s the equivalent code for iOS and Raspberry Pi:

    // Load the model.
    PortableReadFileToProto(file_path, &tensorflow_graph);

    // Create a session from the model.
    tensorflow::Status s = session->Create(tensorflow_graph);
    if (!s.ok()) {
      LOG(FATAL) << "Could not create TensorFlow Graph: " << s;
    }

    // Run the model.
    std::string input_layer = "input";
    std::string output_layer = "output";
    std::vector<tensorflow::Tensor> outputs;
    tensorflow::Status run_status = session->Run({{input_layer, image_tensor}},
                               {output_layer}, {}, &outputs);
    if (!run_status.ok()) {
      LOG(FATAL) << "Running model failed: " << run_status;
    }

    // Access the output data.
    tensorflow::Tensor* output = &outputs[0];

This is all based on the
[iOS sample code](https://www.tensorflow.org/code/tensorflow/examples/ios/simple/RunModelViewController.mm),
but there’s nothing iOS-specific; the same code should be usable on any platform
that supports C++.

You can also find specific examples for Raspberry Pi
[here](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/pi_examples/label_image/label_image.cc).
