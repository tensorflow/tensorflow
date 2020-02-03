# Image Recognition Example

## Table of Contents
 - [Introduction](#introduction)
 - [Hardware](#hardware)
 - [Building](#building)
    - [Building the testcase](#building-the-testcase)
    - [Building the image recognition application](#building-the-image-recognition-application)
        - [Prerequisites](#prerequisites)
        - [Compiling and flashing](#compiling-and-flashing)

## Introduction
This example shows how you can use Tensorflow Lite Micro to perform image
recognition on a [STM32F746 discovery kit](https://www.st.com/en/evaluation-tools/32f746gdiscovery.html)
with a STM32F4DIS-CAM camera module attached.
It classifies the captured image into 1 of 10 different classes, and those
classes are "Plane", "Car", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse",
"Ship", "Truck".

## Hardware
[STM32F746G-DISCO board (Cortex-M7)](https://www.st.com/en/evaluation-tools/32f746gdiscovery.html)\
[STM32F4DIS-CAM Camera module](https://www.element14.com/community/docs/DOC-67585?ICID=knode-STM32F4-cameramore)

## Building
These instructions have been tested on Ubuntu 16.04.

### Building the test case
```
$ make -f tensorflow/lite/micro/tools/make/Makefile image_recognition_test
```
This will build and run the test case. As input, the test case uses
the first 10 images of the test batch included in the [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)
dataset. Details surrounding the dataset can be found in [this paper](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf).

### Building the image recognition application
#### Prerequisites
Install mbed-cli:
```
$ sudo pip install mbed-cli
```

Install the arm-none-eabi-toolchain.

For Ubuntu, this can be done by installing
the package ```gcc-arm-none-eabi```. In Ubuntu 16.04, the version included in
the repository is 4.9.3 while the recommended version is 6 and
up. Later versions can be downloaded from [here](https://developer.arm.com/tools-and-software/open-source-software/developer-tools/gnu-toolchain/gnu-rm/downloads)
for Windows, Mac OS X and Linux.

#### Compiling and flashing
In order to generate the mbed project, run the following command:
```
$ make -f tensorflow/lite/micro/tools/make/Makefile TAGS=disco_f746ng generate_image_recognition_mbed_project
```
This will copy all of the necessary files needed to build and flash the
application.

Navigate to the output folder:
```
$ cd tensorflow/lite/micro/tools/make/gen/linux_x86_64/prj/image_recognition/mbed/
```

The following instructions for compiling and flashing can also be found in the
file README_MBED.md in the output folder.

To load the dependencies required, run:
```
$ mbed config root .
$ mbed deploy
```

In order to compile, run:
```
mbed compile -m auto  -t GCC_ARM --profile release
```

```-m auto```: Automatically detects the correct target if the Discovery board
               is connected to the computer. If the board is not connected,
               replace ```auto``` with ```DISCO_F746NG```.\
```-t GCC_ARM```: Specifies the toolchain used to compile. ```GCC_ARM```
                  indicates that the arm-none-eabi-toolchain will be used.\
```--profile release```: Build the ```release``` profile. The different profiles
                         can be found under mbed-os/tools/profiles/.

This will produce a file named ```mbed.bin``` in
```BUILD/DISCO_F746NG/GCC_ARM-RELEASE/```. To flash it to the board, simply copy
the file to the volume mounted as a USB drive. Alternatively, the ```-f```
option can be appended to flash automatically after compilation.

On Ubuntu 16.04 (and possibly other Linux distributions) there may be an error
message when running ```mbed compile``` saying that the Python module
```pywin32``` failed to install. This message can be ignored.
