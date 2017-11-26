# Building TensorFlow on iOS

## Using CocoaPods

The simplest way to get started with TensorFlow on iOS is using the CocoaPods
package management system. You can add the `TensorFlow-experimental` pod to your
Podfile, which installs a universal binary framework. This makes it easy to get
started but has the disadvantage of being hard to customize, which is important
in case you want to shrink your binary size. If you do need the ability to
customize your libraries, see later sections on how to do that.

## Creating your own app

If you'd like to add TensorFlow capabilities to your own app, do the following:

- Create your own app or load your already-created app in XCode.

- Add a file named Podfile at the project root directory with the following content:

        target 'YourProjectName'
        pod 'TensorFlow-experimental'

- Run `pod install` to download and install the `TensorFlow-experimental` pod.

- Open `YourProjectName.xcworkspace` and add your code.

- In your app's **Build Settings**, make sure to add `$(inherited)` to the 
  **Other Linker Flags**, and **Header Search Paths** sections.

## Running the Samples

You'll need Xcode 7.3 or later to run our iOS samples.

There are currently three examples: simple, benchmark, and camera. For now, you
can download the sample code by cloning the main tensorflow repository (we are
planning to make the samples available as a separate repository later).

From the root of the tensorflow folder, download [Inception
v1](https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip),
and extract the label and graph files into the data folders inside both the
simple and camera examples using these steps:

    mkdir -p ~/graphs
    curl -o ~/graphs/inception5h.zip \
     https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip \
     && unzip ~/graphs/inception5h.zip -d ~/graphs/inception5h
    cp ~/graphs/inception5h/* tensorflow/examples/ios/benchmark/data/
    cp ~/graphs/inception5h/* tensorflow/examples/ios/camera/data/
    cp ~/graphs/inception5h/* tensorflow/examples/ios/simple/data/

Change into one of the sample directories, download the
[Tensorflow-experimental](https://cocoapods.org/pods/TensorFlow-experimental)
pod, and open the Xcode workspace. Note that installing the pod can take a long
time since it is big (~450MB). If you want to run the simple example, then:

    cd tensorflow/examples/ios/simple
    pod install
    open tf_simple_example.xcworkspace   # note .xcworkspace, not .xcodeproj
                                         # this is created by pod install

Run the simple app in the XCode simulator. You should see a single-screen app
with a **Run Model** button. Tap that, and you should see some debug output
appear below indicating that the example Grace Hopper image in directory data
has been analyzed, with a military uniform recognized.

Run the other samples using the same process. The camera example requires a real
device connected. Once you build and run that, you should get a live camera view
that you can point at objects to get real-time recognition results.

### iOS Example details

There are three demo applications for iOS, all defined in Xcode projects inside
[tensorflow/examples/ios](https://www.tensorflow.org/code/tensorflow/examples/ios/).

- **Simple**: This is a minimal example showing how to load and run a TensorFlow
  model in as few lines as possible. It just consists of a single view with a
  button that executes the model loading and inference when its pressed.

- **Camera**: This is very similar to the Android TF Classify demo. It loads
  Inception v3 and outputs its best label estimate for what’s in the live camera
  view. As with the Android version, you can train your own custom model using
  TensorFlow for Poets and drop it into this example with minimal code changes.

- **Benchmark**: is quite close to Simple, but it runs the graph repeatedly and
  outputs similar statistics to the benchmark tool on Android.


### Troubleshooting

- Make sure you use the TensorFlow-experimental pod (and not TensorFlow).

- The TensorFlow-experimental pod is current about ~450MB. The reason it is so
  big is because we are bundling multiple platforms, and the pod includes all
  TensorFlow functionality (e.g. operations). The final app size after build is
  substantially smaller though (~25MB). Working with the complete pod is
  convenient during development, but see below section on how you can build your
  own custom TensorFlow library to reduce the size.

## Building the TensorFlow iOS libraries from source

While Cocoapods is the quickest and easiest way of getting started, you sometimes
need more flexibility to determine which parts of TensorFlow your app should be
shipped with. For such cases, you can build the iOS libraries from the
sources. [This
guide](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/ios#building-the-tensorflow-ios-libraries-from-source)
contains detailed instructions on how to do that.

