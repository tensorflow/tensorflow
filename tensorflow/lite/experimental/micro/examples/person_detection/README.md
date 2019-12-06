# Micro Vision Example

This example shows how you can use Tensorflow Lite to run a 250 kilobyte neural
network to recognize people in images captured by a camera.  It is designed to
run on systems with small amounts of memory such as microcontrollers and DSPs.

## Table of Contents
-   [Getting Started](#getting-started)
-   [Running on Arduino](#running-on-arduino)
-   [Running on SparkFun Edge](#running-on-sparkfun-edge)
-   [Debugging Image Capture](#debugging-image-capture)

## Getting Started

To compile and test this example on a desktop Linux or MacOS machine, download
[the TensorFlow source code](https://github.com/tensorflow/tensorflow), `cd`
into the source directory from a terminal, and then run the following command:

```
make -f tensorflow/lite/experimental/micro/tools/make/Makefile
```

This will take a few minutes, and downloads frameworks the code uses like
[CMSIS](https://developer.arm.com/embedded/cmsis) and
[flatbuffers](https://google.github.io/flatbuffers/). Once that process has
finished, run:

```
make -f tensorflow/lite/experimental/micro/tools/make/Makefile test_person_detection_test
```

You should see a series of files get compiled, followed by some logging output
from a test, which should conclude with `~~~ALL TESTS PASSED~~~`. If you see
this, it means that a small program has been built and run that loads a trained
TensorFlow model, runs some example images through it, and got the expected
outputs. This particular test runs images with a and without a person in them,
and checks that the network correctly identifies them.

To understand how TensorFlow Lite does this, you can look at the `TestInvoke()`
function in
[person_detection_test.cc](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/micro/examples/person_detection/person_detection_test.cc).
It's a fairly small amount of code, creating an interpreter, getting a handle to
a model that's been compiled into the program, and then invoking the interpreter
with the model and sample inputs.

## Running on Arduino

The following instructions will help you build and deploy this sample
to [Arduino](https://www.arduino.cc/) devices.

The sample has been tested with the following device:

- [Arduino Nano 33 BLE Sense](https://store.arduino.cc/usa/nano-33-ble-sense-with-headers)

You will also need the following camera module:

- [Arducam Mini 2MP Plus](https://www.amazon.com/Arducam-Module-Megapixels-Arduino-Mega2560/dp/B012UXNDOY)

### Hardware

Connect the Arducam pins as follows:

|Arducam pin name|Arduino pin name|
|----------------|----------------|
|CS|D7 (unlabelled, immediately to the right of D6)|
|MOSI|D11|
|MISO|D12|
|SCK|D13|
|GND|GND (either pin marked GND is fine)|
|VCC|3.3 V|
|SDA|A4|
|SCL|A5|

### Obtain and import the library

To use this sample application with Arduino, we've created an Arduino library
that includes it as an example that you can open in the Arduino IDE.

Download the current nightly build of the library: [micro_speech.zip](https://storage.googleapis.com/tensorflow-nightly/github/tensorflow/tensorflow/lite/experimental/micro/tools/make/gen/arduino_x86_64/prj/micro_speech/micro_speech.zip)

Next, import this zip file into the Arduino IDE by going to
`Sketch -> Include Library -> Add .ZIP Library...`.

### Install libraries

In addition to the TensorFlow library, you'll also need to install two
libraries:

* The Arducam library, so our code can interface with the hardware
* The JPEGDecoder library, so we can decode JPEG-encoded images

The Arducam Arduino library is available from GitHub at
[https://github.com/ArduCAM/Arduino](https://github.com/ArduCAM/Arduino).
To install it, download or clone the repository. Next, copy its `ArduCAM`
subdirectory into your `Arduino/libraries` directory. To find this directory on
your machine, check the *Sketchbook location* in the Arduino IDE's
*Preferences* window.

After downloading the library, you'll need to edit one of its files to make sure
it is configured for the Arducam Mini 2MP Plus. To do so, open the following
file:

```
Arduino/libraries/ArduCAM/memorysaver.h
```

You'll see a bunch of `#define` statements listed. Make sure that they are all
commented out, except for `#define OV2640_MINI_2MP_PLUS`, as so:

```
//Step 1: select the hardware platform, only one at a time
//#define OV2640_MINI_2MP
//#define OV3640_MINI_3MP
//#define OV5642_MINI_5MP
//#define OV5642_MINI_5MP_BIT_ROTATION_FIXED
#define OV2640_MINI_2MP_PLUS
//#define OV5642_MINI_5MP_PLUS
//#define OV5640_MINI_5MP_PLUS
```

Once you save the file, we're done configuring the Arducam library.

Our next step is to install the JPEGDecoder library. We can do this from within
the Arduino IDE. First, go to the *Manage Libraries...* option in the *Tools*
menu and search for `JPEGDecoder`. You should install version _1.8.0_ of the
library.

Once the library has installed, we'll need to configure it to disable some
optional components that are not compatible with the Arduino Nano 33 BLE Sense.
Open the following file:

```
Arduino/libraries/JPEGDecoder/src/User_Config.h
```

Make sure that both `#define LOAD_SD_LIBRARY` and `#define LOAD_SDFAT_LIBRARY`
are commented out, as shown in this excerpt from the file:

```c++
// Comment out the next #defines if you are not using an SD Card to store the JPEGs
// Commenting out the line is NOT essential but will save some FLASH space if
// SD Card access is not needed. Note: use of SdFat is currently untested!

//#define LOAD_SD_LIBRARY // Default SD Card library
//#define LOAD_SDFAT_LIBRARY // Use SdFat library instead, so SD Card SPI can be bit bashed
```

Once you've saved the file, you are done installing libraries.

### Load and run the example

Go to `File -> Examples`. You should see an
example near the bottom of the list named `TensorFlowLite`. Select
it and click `person_detection` to load the example. Connect your device, then
build and upload the example.

To test the camera, start by pointing the device's camera at something that is
definitely not a person, or just covering it up. The next time the blue LED
flashes, the device will capture a frame from the camera and begin to run
inference. Since the vision model we are using for person detection is
relatively large, it takes a long time to run inference—around 19 seconds at the
time of writing, though it's possible TensorFlow Lite has gotten faster since
then.

After 19 seconds or so, the inference result will be translated into another LED
being lit. Since you pointed the camera at something that isn't a person, the
red LED should light up.

Now, try pointing the device's camera at yourself! The next time the blue LED
flashes, the device will capture another image and begin to run inference. After
19 seconds, the green LED should light up!

Remember, image data is captured as a snapshot before each inference, whenever
the blue LED flashes. Whatever the camera is pointed at during that moment is
what will be fed into the model. It doesn't matter where the camera is pointed
until the next time an image is captured, when the blue LED will flash again.

If you're getting seemingly incorrect results, make sure you are in an
environment with good lighting. You should also make sure that the camera is
oriented correctly, with the pins pointing downwards, so that the images it
captures are the right way up—the model was not trained to recognize upside-down
people! In addition, it's good to remember that this is a tiny model, which
trades accuracy for small size. It works very well, but it isn't accurate 100%
of the time.

We can also see the results of inference via the Arduino Serial Monitor. To do
this, open the *Serial Monitor* from the *Tools* menu. You'll see a detailed
log of what is happening while our application runs. It's also interesting to
check the *Show timestamp* box, so you can see how long each part of the process
takes:

```
14:17:50.714 -> Starting capture
14:17:50.714 -> Image captured
14:17:50.784 -> Reading 3080 bytes from ArduCAM
14:17:50.887 -> Finished reading
14:17:50.887 -> Decoding JPEG and converting to greyscale
14:17:51.074 -> Image decoded and processed
14:18:09.710 -> Person score: 246 No person score: 66
```

From the log, we can see that it took around 170 ms to capture and read the
image data from the camera module, 180 ms to decode the JPEG and convert it to
greyscale, and 18.6 seconds to run inference.

## Running on SparkFun Edge

The following instructions will help you build and deploy this sample on the
[SparkFun Edge development board](https://sparkfun.com/products/15170).  This
sample requires the Sparkfun Himax camera for the Sparkfun Edge board.  It is
not available for purchase yet.

If you're new to using this board, we recommend walking through the
[AI on a microcontroller with TensorFlow Lite and SparkFun Edge](https://codelabs.developers.google.com/codelabs/sparkfun-tensorflow)
codelab to get an understanding of the workflow.

### Compile the binary

The following command will download the required dependencies and then compile a
binary for the SparkFun Edge:

```
make -f tensorflow/lite/experimental/micro/tools/make/Makefile TARGET=sparkfun_edge person_detection_bin
```

The binary will be created in the following location:

```
tensorflow/lite/experimental/micro/tools/make/gen/sparkfun_edge_cortex-m4/bin/person_detection.bin
```

### Sign the binary

The binary must be signed with cryptographic keys to be deployed to the device.
We'll now run some commands that will sign our binary so it can be flashed to
the SparkFun Edge. The scripts we are using come from the Ambiq SDK, which is
downloaded when the `Makefile` is run.

Enter the following command to set up some dummy cryptographic keys we can use
for development:

```
cp tensorflow/lite/experimental/micro/tools/make/downloads/AmbiqSuite-Rel2.0.0/tools/apollo3_scripts/keys_info0.py \
tensorflow/lite/experimental/micro/tools/make/downloads/AmbiqSuite-Rel2.0.0/tools/apollo3_scripts/keys_info.py
```

Next, run the following command to create a signed binary:

```
python3 tensorflow/lite/experimental/micro/tools/make/downloads/AmbiqSuite-Rel2.0.0/tools/apollo3_scripts/create_cust_image_blob.py \
--bin tensorflow/lite/experimental/micro/tools/make/gen/sparkfun_edge_cortex-m4/bin/person_detection.bin \
--load-address 0xC000 \
--magic-num 0xCB \
-o main_nonsecure_ota \
--version 0x0
```

This will create the file `main_nonsecure_ota.bin`. We'll now run another
command to create a final version of the file that can be used to flash our
device with the bootloader script we will use in the next step:

```
python3 tensorflow/lite/experimental/micro/tools/make/downloads/AmbiqSuite-Rel2.0.0/tools/apollo3_scripts/create_cust_wireupdate_blob.py \
--load-address 0x20000 \
--bin main_nonsecure_ota.bin \
-i 6 \
-o main_nonsecure_wire \
--options 0x1
```

You should now have a file called `main_nonsecure_wire.bin` in the directory
where you ran the commands. This is the file we'll be flashing to the device.

### Flash the binary

Next, attach the board to your computer via a USB-to-serial adapter.

**Note:** If you're using the [SparkFun Serial Basic Breakout](https://www.sparkfun.com/products/15096),
you should [install the latest drivers](https://learn.sparkfun.com/tutorials/sparkfun-serial-basic-ch340c-hookup-guide#drivers-if-you-need-them)
before you continue.

Once connected, assign the USB device name to an environment variable:

```
export DEVICENAME=put your device name here
```

Set another variable with the baud rate:

```
export BAUD_RATE=921600
```

Now, hold the button marked `14` on the device. While still holding the button,
hit the button marked `RST`. Continue holding the button marked `14` while
running the following command:

```
python3 tensorflow/lite/experimental/micro/tools/make/downloads/AmbiqSuite-Rel2.0.0/tools/apollo3_scripts/uart_wired_update.py \
-b ${BAUD_RATE} ${DEVICENAME} \
-r 1 \
-f main_nonsecure_wire.bin \
-i 6
```

You should see a long stream of output as the binary is flashed to the device.
Once you see the following lines, flashing is complete:

```
Sending Reset Command.
Done.
```

If you don't see these lines, flashing may have failed. Try running through the
steps in [Flash the binary](#flash-the-binary) again (you can skip over setting
the environment variables). If you continue to run into problems, follow the
[AI on a microcontroller with TensorFlow Lite and SparkFun Edge](https://codelabs.developers.google.com/codelabs/sparkfun-tensorflow)
codelab, which includes more comprehensive instructions for the flashing
process.

The binary should now be deployed to the device. Hit the button marked `RST` to
reboot the board. You should see the device's four LEDs flashing in sequence.

Debug information is logged by the board while the program is running. To view
it, establish a serial connection to the board using a baud rate of `115200`.
On OSX and Linux, the following command should work:

```
screen ${DEVICENAME} 115200
```

To stop viewing the debug output with `screen`, hit `Ctrl+A`, immediately
followed by the `K` key, then hit the `Y` key.

## Debugging Image Capture
When the sample is running, check the LEDs to determine whether the inference is
running correctly.  If the red light is stuck on, it means there was an error
communicating with the camera.  This is likely due to an incorrectly connected
or broken camera.

During inference, the blue LED will toggle every time inference is complete. The
orange LED indicates that no person was found, and the green LED indicates a
person was found. The red LED should never turn on, since it indicates an error.

In order to view the captured image, set the DUMP_IMAGE define in main.cc.  This
causes the board to log raw image info to the console. After the board has been
flashed and reset, dump the log to a text file:


```
screen -L -Logfile <dump file> ${DEVICENAME} 115200
```

Next, run the raw to bitmap converter to view captured images:

```
python3 raw_to_bitmap.py -r GRAY -i <dump file>
```
