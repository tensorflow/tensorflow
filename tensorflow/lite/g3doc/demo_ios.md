
# iOS Demo App

The TensorFlow Lite demo is a camera app that continuously classifies whatever
it sees from your device's back camera, using a quantized MobileNet model. These
instructions walk you through building and running the demo on an iOS device.

## Prerequisites

* You must have [Xcode](https://developer.apple.com/xcode/) installed and have a
  valid Apple Developer ID, and have an iOS device set up and linked to your
  developer account with all of the appropriate certificates. For these
  instructions, we assume that you have already been able to build and deploy an
  app to an iOS device with your current developer environment.

* The demo app requires a camera and must be executed on a real iOS device. You
  can build it and run with the iPhone Simulator but it won't have any camera
  information to classify.

* You don't need to build the entire TensorFlow library to run the demo, but you
  will need to clone the TensorFlow repository if you haven't already:

        git clone https://github.com/tensorflow/tensorflow

* You'll also need the Xcode command-line tools:

        xcode-select --install

    If this is a new install, you will need to run the Xcode application once to
    agree to the license before continuing.

## Building the iOS Demo App

1. Install CocoaPods if you don't have it:

        sudo gem install cocoapods

2. Download the model files used by the demo app (this is done from inside the
   cloned directory):

        sh tensorflow/lite/examples/ios/download_models.sh

3. Install the pod to generate the workspace file:

        cd tensorflow/lite/examples/ios/camera
        pod install

    If you have installed this pod before and that command doesn't work, try

        pod update

    At the end of this step you should have a file called 
    `tflite_camera_example.xcworkspace`.

4. Open the project in Xcode by typing this on the command line:

        open tflite_camera_example.xcworkspace

    This launches Xcode if it isn't open already and opens the
    `tflite_camera_example` project.

5. Build and run the app in Xcode.

    Note that as mentioned earlier, you must already have a device set up and
    linked to your Apple Developer account in order to deploy the app on a
    device.

You'll have to grant permissions for the app to use the device's camera. Point
the camera at various objects and enjoy seeing how the model classifies things!
