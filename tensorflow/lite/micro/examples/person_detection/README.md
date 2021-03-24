# Person detection example

This example shows how you can use Tensorflow Lite to run a 250 kilobyte neural
network to recognize people in images captured by a camera.  It is designed to
run on systems with small amounts of memory such as microcontrollers and DSPs.
This uses the experimental int8 quantized version of the person detection model.

## Table of contents

-   [Getting started](#getting-started)
-   [Running on ARC EM SDP](#running-on-arc-em-sdp)
-   [Running on Arduino](#running-on-arduino)
-   [Running on ESP32](#running-on-esp32)
-   [Running on HIMAX WE1 EVB](#running-on-himax-we1-evb)
-   [Running on SparkFun Edge](#running-on-sparkfun-edge)
-   [Run the tests on a development machine](#run-the-tests-on-a-development-machine)
-   [Debugging image capture](#debugging-image-capture)
-   [Training your own model](#training-your-own-model)

## Running on ARC EM SDP

The following instructions will help you to build and deploy this example to
[ARC EM SDP](https://www.synopsys.com/dw/ipdir.php?ds=arc-em-software-development-platform)
board. General information and instructions on using the board with TensorFlow
Lite Micro can be found in the common
[ARC targets description](/tensorflow/lite/micro/tools/make/targets/arc/README.md).

This example uses asymmetric int8 quantization and can therefore leverage
optimized int8 kernels from the embARC MLI library

The ARC EM SDP board contains a rich set of extension interfaces. You can choose
any compatible camera and modify
[image_provider.cc](/tensorflow/lite/micro/examples/person_detection/image_provider.cc)
file accordingly to use input from your specific camera. By default, results of
running this example are printed to the console. If you would like to instead
implement some target-specific actions, you need to modify
[detection_responder.cc](/tensorflow/lite/micro/examples/person_detection/detection_responder.cc)
accordingly.

The reference implementations of these files are used by default on the EM SDP.

### Initial setup

Follow the instructions on the
[ARC EM SDP Initial Setup](/tensorflow/lite/micro/tools/make/targets/arc/README.md#ARC-EM-Software-Development-Platform-ARC-EM-SDP)
to get and install all required tools for work with ARC EM SDP.

### Generate Example Project

The example project for ARC EM SDP platform can be generated with the following
command:

```
make -f tensorflow/lite/micro/tools/make/Makefile \
TARGET=arc_emsdp ARC_TAGS=reduce_codesize \
OPTIMIZED_KERNEL_DIR=arc_mli \
generate_person_detection_int8_make_project
```

Note that `ARC_TAGS=reduce_codesize` applies example specific changes of code to
reduce total size of application. It can be omitted.

### Build and Run Example

For more detailed information on building and running examples see the
appropriate sections of general descriptions of the
[ARC EM SDP usage with TFLM](/tensorflow/lite/micro/tools/make/targets/arc/README.md#ARC-EM-Software-Development-Platform-ARC-EM-SDP).
In the directory with generated project you can also find a
*README_ARC_EMSDP.md* file with instructions and options on building and
running. Here we only briefly mention main steps which are typically enough to
get it started.

1.  You need to
    [connect the board](/tensorflow/lite/micro/tools/make/targets/arc/README.md#connect-the-board)
    and open an serial connection.

2.  Go to the generated example project director

    ```
    cd tensorflow/lite/micro/tools/make/gen/arc_emsdp_arc/prj/person_detection_int8/make
    ```

3.  Build the example using

    ```
    make app
    ```

4.  To generate artefacts for self-boot of example from the board use

    ```
    make flash
    ```

5.  To run application from the board using microSD card:

    *   Copy the content of the created /bin folder into the root of microSD
        card. Note that the card must be formatted as FAT32 with default cluster
        size (but less than 32 Kbytes)
    *   Plug in the microSD card into the J11 connector.
    *   Push the RST button. If a red LED is lit beside RST button, push the CFG
        button.
    *   Type or copy next commands one-by-another into serial terminal: `setenv
        loadaddr 0x10800000 setenv bootfile app.elf setenv bootdelay 1 setenv
        bootcmd fatload mmc 0 \$\{loadaddr\} \$\{bootfile\} \&\& bootelf
        saveenv`
    *   Push the RST button.

6.  If you have the MetaWare Debugger installed in your environment:

    *   To run application from the console using it type `make run`.
    *   To stop the execution type `Ctrl+C` in the console several times.

In both cases (step 5 and 6) you will see the application output in the serial
terminal.

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

### Install the Arduino_TensorFlowLite library

Download the current nightly build of the library:
[person_detection.zip](https://storage.googleapis.com/download.tensorflow.org/data/tf_lite_micro_person_data_int8_grayscale_2020_01_13.zip)

This example application is included as part of the official TensorFlow Lite
Arduino library. To install it, open the Arduino library manager in
`Tools -> Manage Libraries...` and search for `Arduino_TensorFlowLite`.

### Install other libraries

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
inference. The vision model we are using for person detection is relatively
large, but with cmsis-nn optimizations it only takes around 800ms to run the
model.

After a moment, the inference result will be translated into another LED being
lit. Since you pointed the camera at something that isn't a person, the red LED
should light up.

Now, try pointing the device's camera at yourself! The next time the blue LED
flashes, the device will capture another image and begin to run inference. After
a brief puase, the green LED should light up!

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

## Running on ESP32

The following instructions will help you build and deploy this sample to
[ESP32](https://www.espressif.com/en/products/hardware/esp32/overview) devices
using the [ESP IDF](https://github.com/espressif/esp-idf).

The sample has been tested on ESP-IDF version 4.0 with the following devices: -
[ESP32-DevKitC](http://esp-idf.readthedocs.io/en/latest/get-started/get-started-devkitc.html) -
[ESP-EYE](https://github.com/espressif/esp-who/blob/master/docs/en/get-started/ESP-EYE_Getting_Started_Guide.md)

ESP-EYE is a board which has a built-in camera which can be used to run this
example , if you want to use other esp boards you will have to connect camera
externally and write your own
[image_provider.cc](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/micro/examples/person_detection/esp/image_provider.cc).
and
[app_camera_esp.c](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/micro/examples/person_detection/esp/app_camera_esp.c).
You can also write you own
[detection_responder.cc](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/micro/examples/person_detection/detection_responder.cc).

### Install the ESP IDF

Follow the instructions of the
[ESP-IDF get started guide](https://docs.espressif.com/projects/esp-idf/en/latest/get-started/index.html)
to setup the toolchain and the ESP-IDF itself.

The next steps assume that the
[IDF environment variables are set](https://docs.espressif.com/projects/esp-idf/en/latest/get-started/index.html#step-4-set-up-the-environment-variables) :

*   The `IDF_PATH` environment variable is set
*   `idf.py` and Xtensa-esp32 tools (e.g. `xtensa-esp32-elf-gcc`) are in `$PATH`
*   `esp32-camera` should be downloaded in `components/` dir of example as
    explained in `Building the example`(below)

### Generate the examples

The example project can be generated with the following command:
```
make -f tensorflow/lite/micro/tools/make/Makefile TARGET=esp generate_person_detection_esp_project
```

### Building the example

Go to the example project directory
```
cd tensorflow/lite/micro/tools/make/gen/esp_xtensa-esp32/prj/person_detection/esp-idf
```

As the `person_detection` example requires an external component `esp32-camera`
for functioning hence we will have to manually clone it in `components/`
directory of the example with following command.
```
git clone https://github.com/espressif/esp32-camera.git components/esp32-camera
```

Then build with `idf.py` `idf.py build`

### Load and run the example

To flash (replace `/dev/ttyUSB0` with the device serial port):
```
idf.py --port /dev/ttyUSB0 flash
```

Monitor the serial output:
```
idf.py --port /dev/ttyUSB0 monitor
```

Use `Ctrl+]` to exit.

The previous two commands can be combined:
```
idf.py --port /dev/ttyUSB0 flash monitor
```

## Running on HIMAX WE1 EVB

The following instructions will help you build and deploy this example to
[HIMAX WE1 EVB](https://github.com/HimaxWiseEyePlus/bsp_tflu/tree/master/HIMAX_WE1_EVB_board_brief)
board. To understand more about using this board, please check
[HIMAX WE1 EVB user guide](https://github.com/HimaxWiseEyePlus/bsp_tflu/tree/master/HIMAX_WE1_EVB_user_guide).

### Initial Setup

To use the HIMAX WE1 EVB, please make sure following software are installed:

#### MetaWare Development Toolkit

See
[Install the Synopsys DesignWare ARC MetaWare Development Toolkit](/tensorflow/lite/micro/tools/make/targets/arc/README.md#install-the-synopsys-designware-arc-metaware-development-toolkit)
section for instructions on toolchain installation.

#### Make Tool version

A `'make'` tool is required for deploying Tensorflow Lite Micro applications on
HIMAX WE1 EVB, See
[Check make tool version](/tensorflow/lite/micro/tools/make/targets/arc/README.md#make-tool)
section for proper environment.

#### Serial Terminal Emulation Application

There are 2 main purposes for HIMAX WE1 EVB Debug UART port

-   print application output
-   burn application to flash by using xmodem send application binary

You can use any terminal emulation program (like [PuTTY](https://www.putty.org/)
or [minicom](https://linux.die.net/man/1/minicom)).

### Generate Example Project

The example project for HIMAX WE1 EVB platform can be generated with the
following command:

Download related third party data

```
make -f tensorflow/lite/micro/tools/make/Makefile TARGET=himax_we1_evb third_party_downloads
```

Generate person detection project

```
make -f tensorflow/lite/micro/tools/make/Makefile generate_person_detection_int8_make_project TARGET=himax_we1_evb
```

### Build and Burn Example

Following the Steps to run person detection example at HIMAX WE1 EVB platform.

1.  Go to the generated example project directory.

    ```
    cd tensorflow/lite/micro/tools/make/gen/himax_we1_evb_arc/prj/person_detection_int8/make
    ```

2.  Build the example using

    ```
    make app
    ```

3.  After example build finish, copy ELF file and map file to image generate
    tool directory. \
    image generate tool directory located at
    `'tensorflow/lite/micro/tools/make/downloads/himax_we1_sdk/image_gen_linux_v3/'`

    ```
    cp person_detection_int8.elf himax_we1_evb.map ../../../../../downloads/himax_we1_sdk/image_gen_linux_v3/
    ```

4.  Go to flash image generate tool directory.

    ```
    cd ../../../../../downloads/himax_we1_sdk/image_gen_linux_v3/
    ```

    make sure this tool directory is in $PATH. You can permanently set it to
    PATH by

    ```
    export PATH=$PATH:$(pwd)
    ```

5.  run image generate tool, generate flash image file.

    *   Before running image generate tool, by typing `sudo chmod +x image_gen`
        and `sudo chmod +x sign_tool` to make sure it is executable.

    ```
    image_gen -e person_detection_int8.elf -m himax_we1_evb.map -o out.img
    ```

6.  Download flash image file to HIMAX WE1 EVB by UART:

    *   more detail about download image through UART can be found at
        [HIMAX WE1 EVB update Flash image](https://github.com/HimaxWiseEyePlus/bsp_tflu/tree/master/HIMAX_WE1_EVB_user_guide#flash-image-update)

After these steps, press reset button on the HIMAX WE1 EVB, you will see
application output in the serial terminal.

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
make -f tensorflow/lite/micro/tools/make/Makefile TARGET=sparkfun_edge person_detection_int8_bin
```

The binary will be created in the following location:

```
tensorflow/lite/micro/tools/make/gen/sparkfun_edge_cortex-m4/bin/person_detection_int8.bin
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
--bin tensorflow/lite/micro/tools/make/gen/sparkfun_edge_cortex-m4/bin/person_detection_int8.bin \
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
reboot the board. You should see the device's four LEDs flashing in sequence.

Debug information is logged by the board while the program is running. To view
it, establish a serial connection to the board using a baud rate of `115200`.
On OSX and Linux, the following command should work:

```
screen ${DEVICENAME} 115200
```

To stop viewing the debug output with `screen`, hit `Ctrl+A`, immediately
followed by the `K` key, then hit the `Y` key.

## Run the tests on a development machine

To compile and test this example on a desktop Linux or MacOS machine, download
[the TensorFlow source code](https://github.com/tensorflow/tensorflow), `cd`
into the source directory from a terminal, and then run the following command:

```
make -f tensorflow/lite/micro/tools/make/Makefile
```

This will take a few minutes, and downloads frameworks the code uses like
[CMSIS](https://developer.arm.com/embedded/cmsis) and
[flatbuffers](https://google.github.io/flatbuffers/). Once that process has
finished, run:

```
make -f tensorflow/lite/micro/tools/make/Makefile test_person_detection_test
```

You should see a series of files get compiled, followed by some logging output
from a test, which should conclude with `~~~ALL TESTS PASSED~~~`. If you see
this, it means that a small program has been built and run that loads a trained
TensorFlow model, runs some example images through it, and got the expected
outputs. This particular test runs images with a and without a person in them,
and checks that the network correctly identifies them.

To understand how TensorFlow Lite does this, you can look at the `TestInvoke()`
function in
[person_detection_test.cc](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/micro/examples/person_detection/person_detection_test.cc).
It's a fairly small amount of code, creating an interpreter, getting a handle to
a model that's been compiled into the program, and then invoking the interpreter
with the model and sample inputs.

## Debugging image capture
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

## Training your own model

You can train your own model with some easy-to-use scripts. See
[training_a_model.md](training_a_model.md) for instructions.
