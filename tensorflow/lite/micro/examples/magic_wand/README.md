# Magic wand example

This example shows how you can use TensorFlow Lite to run a 20 kilobyte neural
network model to recognize gestures with an accelerometer. It's designed to run
on systems with very small amounts of memory, such as microcontrollers.

The example application reads data from the accelerometer on an Arduino Nano 33
BLE Sense or SparkFun Edge board and indicates when it has detected a gesture,
then outputs the gesture to the serial port.

## Table of contents

-   [Getting started](#getting-started)
-   [Deploy to Arduino](#deploy-to-arduino)
-   [Deploy to SparkFun Edge](#deploy-to-sparkfun-edge)
-   [Run the tests on a development machine](#run-the-tests-on-a-development-machine)
-   [Train your own model](#train-your-own-model)

## Deploy to Arduino

The following instructions will help you build and deploy this sample
to [Arduino](https://www.arduino.cc/) devices.

The sample has been tested with the following devices:

- [Arduino Nano 33 BLE Sense](https://store.arduino.cc/usa/nano-33-ble-sense-with-headers)

### Install the Arduino_TensorFlowLite library

This example application is included as part of the official TensorFlow Lite
Arduino library. To install it, open the Arduino library manager in
`Tools -> Manage Libraries...` and search for `Arduino_TensorFlowLite`.

### Install and patch the accelerometer driver

This example depends on the [Arduino_LSM9DS1](https://github.com/arduino-libraries/Arduino_LSM9DS1)
library to communicate with the device's accelerometer. However, the library
must be patched in order to enable the accelerometer's FIFO buffer.

Follow these steps to install and patch the driver:

#### Install the correct version

In the Arduino IDE, go to `Tools -> Manage Libraries...` and search for
`Arduino_LSM9DS1`. **Install version 1.0.0 of the driver** to ensure the
following instructions work.

#### Patch the driver

The driver will be installed to your `Arduino/libraries` directory, in the
subdirectory `Arduino_LSM9DS1`.

Open the following file:

```
Arduino_LSM9DS1/src/LSM9DS1.cpp
```

Go to the function named `LSM9DS1Class::begin()`. Insert the following lines at
the end of the function, immediately before the `return 1` statement:

```cpp
// Enable FIFO (see docs https://www.st.com/resource/en/datasheet/DM00103319.pdf)
writeRegister(LSM9DS1_ADDRESS, 0x23, 0x02);
// Set continuous mode
writeRegister(LSM9DS1_ADDRESS, 0x2E, 0xC0);
```

Next, go to the function named `LSM9DS1Class::accelerationAvailable()`. You will
see the following lines:

```cpp
if (readRegister(LSM9DS1_ADDRESS, LSM9DS1_STATUS_REG) & 0x01) {
  return 1;
}
```

Comment out those lines and replace them with the following:

```cpp
// Read FIFO_SRC. If any of the rightmost 8 bits have a value, there is data
if (readRegister(LSM9DS1_ADDRESS, 0x2F) & 63) {
  return 1;
}
```

Next, save the file. Patching is now complete.

### Load and run the example

Once the library has been added, go to `File -> Examples`. You should see an
example near the bottom of the list named `TensorFlowLite`. Select
it and click `magic_wand` to load the example.

Use the Arduino Desktop IDE to build and upload the example. Once it is running,
you should see the built-in LED on your device flashing.

Open the Arduino Serial Monitor (`Tools -> Serial Monitor`).

You will see the following message:

```
Magic starts！
```

Hold the Arduino with its components facing upwards and the USB cable to your
left. Perform the gestures "WING", "RING"(clockwise), and "SLOPE", and you
should see the corresponding output:

```
WING:
*         *         *
 *       * *       *
  *     *   *     *
   *   *     *   *
    * *       * *
     *         *
```

```
RING:
          *
       *     *
     *         *
    *           *
     *         *
       *     *
          *
```

```
SLOPE:
        *
       *
      *
     *
    *
   *
  *
 * * * * * * * *
```

## Deploy to SparkFun Edge

The following instructions will help you build and deploy this sample on the
[SparkFun Edge development board](https://sparkfun.com/products/15170).

If you're new to using this board, we recommend walking through the
[AI on a microcontroller with TensorFlow Lite and SparkFun Edge](https://codelabs.developers.google.com/codelabs/sparkfun-tensorflow)
codelab to get an understanding of the workflow.

### Compile the binary

Run the following command to build a binary for SparkFun Edge.

```
make -f tensorflow/lite/micro/tools/make/Makefile TARGET=sparkfun_edge magic_wand_bin
```

The binary will be created in the following location:

```
tensorflow/lite/micro/tools/make/gen/sparkfun_edge_cortex-m4/bin/magic_wand.bin
```

### Sign the binary

The binary must be signed with cryptographic keys to be deployed to the device.
We'll now run some commands that will sign our binary so it can be flashed to
the SparkFun Edge. The scripts we are using come from the Ambiq SDK, which is
downloaded when the `Makefile` is run.

Enter the following command to set up some dummy cryptographic keys we can use
for development:

```
cp tensorflow/lite/micro/tools/make/downloads/AmbiqSuite-Rel2.0.0/tools/apollo3_scripts/keys_info0.py \
tensorflow/lite/micro/tools/make/downloads/AmbiqSuite-Rel2.0.0/tools/apollo3_scripts/keys_info.py
```

Next, run the following command to create a signed binary:

```
python3 tensorflow/lite/micro/tools/make/downloads/AmbiqSuite-Rel2.0.0/tools/apollo3_scripts/create_cust_image_blob.py \
--bin tensorflow/lite/micro/tools/make/gen/sparkfun_edge_cortex-m4/bin/magic_wand.bin \
--load-address 0xC000 \
--magic-num 0xCB \
-o main_nonsecure_ota \
--version 0x0
```

This will create the file `main_nonsecure_ota.bin`. We'll now run another
command to create a final version of the file that can be used to flash our
device with the bootloader script we will use in the next step:

```
python3 tensorflow/lite/micro/tools/make/downloads/AmbiqSuite-Rel2.0.0/tools/apollo3_scripts/create_cust_wireupdate_blob.py \
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

**Note:** If you're using the
[SparkFun Serial Basic Breakout](https://www.sparkfun.com/products/15096), you
should
[install the latest drivers](https://learn.sparkfun.com/tutorials/sparkfun-serial-basic-ch340c-hookup-guide#drivers-if-you-need-them)
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
python3 tensorflow/lite/micro/tools/make/downloads/AmbiqSuite-Rel2.0.0/tools/apollo3_scripts/uart_wired_update.py \
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
reboot the board.

Do the three magic gestures and you will see the corresponding LED light on! Red
for "Wing", blue for "Ring" and green for "Slope".

Debug information is logged by the board while the program is running. To view
it, establish a serial connection to the board using a baud rate of `115200`. On
OSX and Linux, the following command should work:

```
screen ${DEVICENAME} 115200
```

You will see the following message:

```
Magic starts！
```

Keep the chip face up, do magic gestures "WING", "RING"(clockwise), and "SLOPE"
with your wand, and you will see the corresponding output like this!

```
WING:
*         *         *
 *       * *       *
  *     *   *     *
   *   *     *   *
    * *       * *
     *         *
```

```
RING:
          *
       *     *
     *         *
    *           *
     *         *
       *     *
          *
```

```
SLOPE:
        *
       *
      *
     *
    *
   *
  *
 * * * * * * * *
```

To stop viewing the debug output with `screen`, hit `Ctrl+A`, immediately
followed by the `K` key, then hit the `Y` key.

## Run the tests on a development machine

To compile and test this example on a desktop Linux or macOS machine, first
clone the TensorFlow repository from GitHub to a convenient place:

```bash
git clone --depth 1 https://github.com/tensorflow/tensorflow.git
```

Next, put this folder under the
tensorflow/tensorflow/lite/micro/examples/ folder, then `cd` into
the source directory from a terminal and run the following command:

```bash
make -f tensorflow/lite/micro/tools/make/Makefile test_magic_wand_test
```

This will take a few minutes, and downloads frameworks the code uses like
[CMSIS](https://developer.arm.com/embedded/cmsis) and
[flatbuffers](https://google.github.io/flatbuffers/). Once that process has
finished, you should see a series of files get compiled, followed by some
logging output from a test, which should conclude with `~~~ALL TESTS PASSED~~~`.

If you see this, it means that a small program has been built and run that loads
the trained TensorFlow model, runs some example inputs through it, and got the
expected outputs.

To understand how TensorFlow Lite does this, you can look at the source in
[hello_world_test.cc](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/micro/examples/hello_world/hello_world_test.cc).
It's a fairly small amount of code that creates an interpreter, gets a handle to
a model that's been compiled into the program, and then invokes the interpreter
with the model and sample inputs.

## Train your own model

To train your own model, or create a new model for a new set of gestures,
follow the instructions in [magic_wand/train/README.md](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/micro/examples/magic_wand/train/README.md).
