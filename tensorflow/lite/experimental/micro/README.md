# TensorFlow Lite for Microcontrollers

This an experimental port of TensorFlow Lite aimed at micro controllers and
other devices with only kilobytes of memory. It doesn't require any operating
system support, any standard C or C++ libraries, or dynamic memory allocation,
so it's designed to be portable even to 'bare metal' systems. The core runtime
fits in 16KB on a Cortex M3, and with enough operators to run a speech keyword
detection model, takes up a total of 22KB.

## Table of Contents

-   [Getting Started](#getting-started)
    *   [Examples](#examples)
    *   [Getting Started with Portable Reference Code](#getting-started-with-portable-reference-code)
    *   [Building Portable Reference Code using Make](#building-portable-reference-code-using-make)
    *   [Building for the "Blue Pill" STM32F103 using Make](#building-for-the-blue-pill-stm32f103-using-make)
    *   [Building for "Hifive1" SiFive FE310 development board using Make](#building-for-hifive1-sifive-fe310-development-board)
    *   [Building for Ambiq Micro Apollo3Blue EVB using Make](#building-for-ambiq-micro-apollo3blue-evb-using-make)
        *   [Additional Apollo3 Instructions](#additional-apollo3-instructions)
    *   [Building for the Eta Compute ECM3531 EVB using Make](#Building-for-the-Eta-Compute-ECM3531-EVB-using-Make)

-   [Goals](#goals)

-   [Generating Project Files](#generating-project-files)

-   [Generating Arduino Libraries](#generating-arduino-libraries)

-   [How to Port TensorFlow Lite Micro to a New Platform](#how-to-port-tensorflow-lite-micro-to-a-new-platform)

    *   [Requirements](#requirements)
    *   [Getting Started](#getting-started-1)
    *   [Troubleshooting](#troubleshooting)
    *   [Optimizing for your Platform](#optimizing-for-your-platform)
    *   [Code Module Organization](#code-module-organization)
    *   [Working with Generated Projects](#working-with-generated-projects)
    *   [Supporting a Platform with Makefiles](#supporting-a-platform-with-makefiles)
    *   [Supporting a Platform with Emulation Testing](#supporting-a-platform-with-emulation-testing)
    *   [Implementing More Optimizations](#implementing-more-optimizations)

# Getting Started

## Examples

The fastest way to learn how TensorFlow Lite for Microcontrollers works is by
exploring and running our examples, which include application code and trained
TensorFlow models.

The following examples are available:

- [hello_world](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/micro/examples/hello_world)
  * Uses a very simple model, trained to reproduce a sine wave, to control an
    LED or animation
  * Application code for Arduino, SparkFun Edge, and STM32F746
  * Colab walkthrough of model training and conversion

- [micro_speech](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/micro/examples/micro_speech)
  * Uses a 20 KB model to recognize keywords in spoken audio
  * Application code for Arduino, SparkFun Edge, and STM32F746
  * Python scripts for model training and conversion

- [person_detection](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/micro/examples/person_detection)
  * Uses a 250 KB model to recognize presence or absence of a person in images
    captured by a camera
  * Application code for SparkFun Edge

- [magic_wand](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/micro/examples/magic_wand)
  * Uses a 20 KB model to recognize gestures using accelerometer data
  * Application code for Arduino and SparkFun Edge

## Pre-generated Project Files

One of the challenges of embedded software development is that there are a lot
of different architectures, devices, operating systems, and build systems. We
aim to support as many of the popular combinations as we can, and make it as
easy as possible to add support for others.

If you're a product developer, we have build instructions or pre-generated
project files that you can download for the following platforms:

Device                                                                                         | Mbed                                                                           | Keil                                                                           | Make/GCC
---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------ | --------
[STM32F746G Discovery Board](https://www.st.com/en/evaluation-tools/32f746gdiscovery.html)     | [Download](https://drive.google.com/open?id=1OtgVkytQBrEYIpJPsE8F6GUKHPBS3Xeb) | -                                                                              | [Instructions](#generating-project-files)
["Blue Pill" STM32F103-compatible development board](https://github.com/google/stm32_bare_lib) | -                                                                              | -                                                                              | [Instructions](#building-for-the-blue-pill-stm32f103-using-make)
[Ambiq Micro Apollo3Blue EVB using Make](https://ambiqmicro.com/apollo-ultra-low-power-mcus/)  | -                                                                              | -                                                                              | [Instructions](#building-for-ambiq-micro-apollo3blue-evb-using-make)
[Generic Keil uVision Projects](http://www2.keil.com/mdk5/uvision/)                            | -                                                                              | [Download](https://drive.google.com/open?id=1Lw9rsdquNKObozClLPoE5CTJLuhfh5mV) | -
[Eta Compute ECM3531 EVB](https://etacompute.com/)                                             | -                                                                              | -                                                                              | [Instructions](#Building-for-the-Eta-Compute-ECM3531-EVB-using-Make)

If your device is not yet supported, it may not be too hard to add support. You
can learn about that process
[here](#how-to-port-tensorflow-lite-micro-to-a-new-platform). We're looking
forward to getting your help expanding this table!

## Getting Started with Portable Reference Code

If you don't have a particular microcontroller platform in mind yet, or just
want to try out the code before beginning porting, the easiest way to begin is
by
[downloading the platform-agnostic reference code](https://drive.google.com/open?id=1cawEQAkqquK_SO4crReDYqf_v7yAwOY8).
You'll see a series of folders inside the archive, with each one containing just
the source files you need to build one binary. There is a simple Makefile for
each folder, but you should be able to load the files into almost any IDE and
build them. There's also a [Visual Studio Code](https://code.visualstudio.com/) project file already set up, so
you can easily explore the code in a cross-platform IDE.

## Building Portable Reference Code using Make

It's easy to build portable reference code directly from GitHub using make if
you're on a Linux or OS X machine with an internet connection.

-   Open a terminal
-   Download the TensorFlow source with `git clone
    https://github.com/tensorflow/tensorflow.git`
-   Enter the source root directory by running `cd tensorflow`
-   Build and test the library with `make -f
    tensorflow/lite/experimental/micro/tools/make/Makefile test`

You should see a series of compilation steps, followed by `~~~ALL TESTS
PASSED~~~` for the various tests of the code that it will run. If there's an
error, you should get an informative message from make about what went wrong.

These tests are all built as simple binaries with few dependencies, so you can
run them manually. For example, here's how to run the depthwise convolution
test, and its output:

```
tensorflow/lite/experimental/micro/tools/make/gen/linux_x86_64/bin/depthwise_conv_test

Testing SimpleTest
Testing SimpleTestQuantized
Testing SimpleTestRelu
Testing SimpleTestReluQuantized
4/4 tests passed
~ALL TESTS PASSED~~~
```

Looking at the
[depthwise_conv_test.cc](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/micro/kernels/depthwise_conv_test.cc)
code, you'll see a sequence that looks like this:

```
...
TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(SimpleTest) {
...
}
...
TF_LITE_MICRO_TESTS_END
```

These macros work a lot like
[the Google test framework](https://github.com/google/googletest), but they
don't require any dependencies and just write results to stderr, rather than
aborting the program. If all the tests pass, then `~~~ALL TESTS PASSED~~~` is
output, and the test harness that runs the binary during the make process knows
that everything ran correctly. If there's an error, the lack of the expected
string lets the harness know that the test failed.

So, why are we running tests in this complicated way? So far, we've been
building binaries that run locally on the Mac OS or Linux machine you're
building on, but this approach becomes important when we're targeting simple
micro controller devices.

## Building for the "Blue Pill" STM32F103 using Make

The goal of this library is to enable machine learning on resource-constrained
micro controllers and DSPs, and as part of that we've targeted the
["Blue Pill" STM32F103-compatible development board](https://github.com/google/stm32_bare_lib)
as a cheap and popular platform. It only has 20KB of RAM and 64KB of flash, so
it's a good device to ensure we can run efficiently on small chips.

It's fairly easy to
[buy and wire up a physical board](https://github.com/google/stm32_bare_lib#wiring-up-your-blue-pill),
but even if you don't have an actual device, the
[Renode project](https://renode.io/) makes it easy to run a faithful emulation
on your desktop machine. You'll need [Docker](https://www.docker.com/)
installed, but once you have that set up, try running the following command:

`make -f tensorflow/lite/experimental/micro/tools/make/Makefile TARGET=bluepill
test`

You should see a similar set of outputs as you did in the previous section, with
the addition of some extra Docker logging messages. These are because we're
using Docker to run the Renode micro controller emulation tool, and the tests
themselves are being run on a simulated STM32F103 device. The communication
channels between an embedded device and the host are quite limited, so the test
harness looks at the output of the debug log to see if tests have passed, just
as it did in the previous section. This makes it a very flexible way to run
cross-platform tests, even when a platform has no operating system facilities,
as long as it can output debugging text logs.

To understand what's happening here, try running the same depthwise convolution
test, but through the emulated device test harness, with the following command:

```
tensorflow/lite/experimental/micro/testing/test_bluepill_binary.sh \
tensorflow/lite/experimental/micro/tools/make/gen/bluepill_cortex-m3/bin/depthwise_conv_test \
'~~~ALL TESTS PASSED~~~'

```

You should see output that looks something like this:

```
Sending build context to Docker daemon   21.5kB
Step 1/2 : FROM antmicro/renode:latest
 ---> 1b670a243e8f
Step 2/2 : LABEL maintainer="Pete Warden <petewarden@google.com>"
 ---> Using cache
 ---> 3afcd410846d
Successfully built 3afcd410846d
Successfully tagged renode_bluepill:latest
LOGS:
...
03:27:32.4340 [INFO] machine-0: Machine started.
03:27:32.4790 [DEBUG] cpu.uartSemihosting: [+0.22s host +0s virt 0s virt from start] Testing SimpleTest
03:27:32.4812 [DEBUG] cpu.uartSemihosting: [+2.21ms host +0s virt 0s virt from start]   Testing SimpleTestQuantized
03:27:32.4833 [DEBUG] cpu.uartSemihosting: [+2.14ms host +0s virt 0s virt from start]   Testing SimpleTestRelu
03:27:32.4834 [DEBUG] cpu.uartSemihosting: [+0.18ms host +0s virt 0s virt from start]   Testing SimpleTestReluQuantized
03:27:32.4838 [DEBUG] cpu.uartSemihosting: [+0.4ms host +0s virt 0s virt from start]   4/4 tests passed
03:27:32.4839 [DEBUG] cpu.uartSemihosting: [+41µs host +0s virt 0s virt from start]   ~~~ALL TESTS PASSED~~~
03:27:32.4839 [DEBUG] cpu.uartSemihosting: [+5µs host +0s virt 0s virt from start]
...
tensorflow/lite/experimental/micro/tools/make/gen/bluepill_cortex-m3/bin/depthwise_conv_test: PASS
```

There's a lot of output here, but you should be able to see that the same tests
that were covered when we ran locally on the development machine show up in the
debug logs here, along with the magic string `~~~ALL TESTS PASSED~~~`. This is
the exact same code as before, just compiled and run on the STM32F103 rather
than your desktop. We hope that the simplicity of this testing approach will
help make adding support for new platforms as easy as possible.

## Building for "Hifive1" SiFive FE310 development board

We've targeted the
["HiFive1" Arduino-compatible development board](https://www.sifive.com/boards/hifive1)
as a test platform for RISC-V MCU.

Similar to Blue Pill setup, you will need Docker installed. The binary can be
executed on either HiFive1 board or emulated using
[Renode project](https://renode.io/) on your desktop machine.

The following instructions builds and transfers the source files to the Docker
`docker build -t riscv_build \ -f
{PATH_TO_TENSORFLOW_ROOT_DIR}/tensorflow/lite/experimental/micro/testing/Dockerfile.riscv
\ {PATH_TO_TENSORFLOW_ROOT_DIR}/tensorflow/lite/experimental/micro/testing/`

You should see output that looks something like this:

```
Sending build context to Docker daemon  28.16kB
Step 1/4 : FROM antmicro/renode:latest
 ---> 19c08590e817
Step 2/4 : LABEL maintainer="Pete Warden <petewarden@google.com>"
 ---> Using cache
 ---> 5a7770d3d3f5
Step 3/4 : RUN apt-get update
 ---> Using cache
 ---> b807ab77eeb1
Step 4/4 : RUN apt-get install -y curl git unzip make g++
 ---> Using cache
 ---> 8da1b2aa2438
Successfully built 8da1b2aa2438
Successfully tagged riscv_build:latest
```

Building micro_speech_test binary

-   Launch the Docker that we just created using: `docker run -it-v
    /tmp/copybara_out:/workspace riscv_build:latest bash`
-   Enter the source root directory by running `cd /workspace`
-   Set the path to RISC-V tools: `export
    PATH=${PATH}:/workspace/tensorflow/lite/experimental/micro/tools/make/downloads/riscv_toolchain/bin/`
-   Build the binary: `make -f
    tensorflow/lite/experimental/micro/tools/make/Makefile TARGET=riscv32_mcu`

Launching Renode to test the binary, currently this set up is not automated.

-   Execute the binary on Renode: `renode -P 5000 --disable-xwt -e 's
    @/workspace/tensorflow/lite/experimental/micro/testing/sifive_fe310.resc'`

You should see the following log with the magic string `~~~ALL TEST PASSED~~~`:

```
02:25:22.2059 [DEBUG] uart0: [+17.25s host +80ms virt 80ms virt from start] core freq at 0 Hz
02:25:22.2065 [DEBUG] uart0: [+0.61ms host +0s virt 80ms virt from start]   Testing TestInvoke
02:25:22.4243 [DEBUG] uart0: [+0.22s host +0.2s virt 0.28s virt from start]   Ran successfully
02:25:22.4244 [DEBUG] uart0: [+42µs host +0s virt 0.28s virt from start]
02:25:22.4245 [DEBUG] uart0: [+0.15ms host +0s virt 0.28s virt from start]   1/1 tests passed
02:25:22.4247 [DEBUG] uart0: [+62µs host +0s virt 0.28s virt from start]   ~~~ALL TESTS PASSED~~~
02:25:22.4251 [DEBUG] uart0: [+8µs host +0s virt 0.28s virt from start]
02:25:22.4252 [DEBUG] uart0: [+0.39ms host +0s virt 0.28s virt from start]
02:25:22.4253 [DEBUG] uart0: [+0.16ms host +0s virt 0.28s virt from start]   Progam has exited with code:0x00000000
```

## Building for Ambiq Micro Apollo3Blue EVB using Make

Follow these steps to get the micro_speech yes example working on Apollo 3 EVB:

1.  Make sure to run the "Building Portable Reference Code using Make" section
    before performing the following steps
2.  Compile the project with the following command: make -f
    tensorflow/lite/experimental/micro/tools/make/Makefile TARGET=apollo3evb
    micro_speech_bin
3.  Install [Segger JLink tools](https://www.segger.com/downloads/jlink/)
4.  Connect the Apollo3 EVB (with mic shield in slot 3 of Microbus Shield board)
    to the computer and power it on.
5.  Start the GDB server in a new terminal with the following command:
    JLinkGDBServer -select USB -device AMA3B1KK-KBR -endian little -if SWD
    -speed 1000 -noir -noLocalhostOnly
    1.  The command has run successfully if you see the message "Waiting for GDB
        connection"
6.  Back in the original terminal, run the program via the debugger
    1.  Navigate to
        tensorflow/lite/experimental/micro/examples/micro_speech/apollo3evb
    2.  Start gdb by entering the following command: arm-none-eabi-gdb
    3.  Run the command script by entering the following command: source
        micro_speech.cmd. This script does the following:
        1.  Load the binary created in step 2
        2.  Reset
        3.  Begin program execution
        4.  Press Ctrl+c to exit
    4.  The EVB LEDs will indicate detection.
        1.  LED0 (rightmost LED) - ON when Digital MIC interface is initialized
        2.  LED1 - Toggles after each inference
        3.  LED2 thru 4 - "Ramp ON" when "Yes" is detected
    5.  Say "Yes"

### Additional Apollo3 Instructions

To flash a part with JFlash Lite, do the following:

1.  At the command line: JFlashLiteExe
2.  Device = AMA3B1KK-KBR
3.  Interface = SWD at 1000 kHz
4.  Data file =
    `tensorflow/lite/experimental/micro/tools/make/gen/apollo3evb_cortex-m4/bin/micro_speech.bin`
5.  Prog Addr = 0x0000C000

## Building for the Eta Compute ECM3531 EVB using Make

1.  Follow the instructions at
    [Tensorflow Micro Speech](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/micro/examples/micro_speech#getting-started)
    to down load the Tensorflow source code and the support libraries \(but do
    not run the make command shown there.\)
2.  Download the Eta Compute SDK, version 0.0.17. Contact info@etacompute.com
3.  You will need the Arm compiler arm-none-eabi-gcc, version 7.3.1
    20180622, release ARM/embedded-7-branch revision 261907, 7-2018-q2-update.
    This compiler is downloaded through make.
4.  Edit the file
    tensorflow/lite/experimental/micro/tools/make/targets/ecm3531_makefile.inc
    so that the variables ETA_SDK and GCC_ARM point to the correct directories.
5.  Compile the code with the command \
    &nbsp;&nbsp;&nbsp;&nbsp;make -f
    tensorflow/lite/experimental/micro/tools/make/Makefile TARGET=ecm3531
    TAGS="CMSIS" test \
    This will produce a set of executables in the
    tensorflow/lite/experimental/micro/tools/make/gen/ecm3531_cortex-m3/bin
    directory.
6.  To load an executable into SRAM \
    &nbsp;&nbsp;&nbsp;&nbsp;Start ocd \
    &nbsp;&nbsp;&nbsp;&nbsp;cd
    tensorflow/lite/experimental/micro/tools/make/targets/ecm3531 \
    &nbsp;&nbsp;&nbsp;&nbsp;./load_program name_of_executable, for e.g.,
    ./load_program audio_provider_test \
    &nbsp;&nbsp;&nbsp;&nbsp;Start PuTTY \(Connection type = Serial, Speed =
    11520, Data bits = 8, Stop bits = 1, Parity = None\) \
    The following output should appear: \
    Testing TestAudioProvider \
    Testing TestTimer \
    2/2 tests passed \
    \~\~\~ALL TESTS PASSED\~\~\~ \
    Execution time \(msec\) = 7
7.  To load into flash \
    &nbsp;&nbsp;&nbsp;&nbsp;Edit the variable ETA_LDS_FILE in
    tensorflow/lite/experimental/micro/tools/&nbsp;&nbsp;make/targets/ecm3531_makefile.inc
    to point to the ecm3531_flash.lds file \
    &nbsp;&nbsp;&nbsp;&nbsp;Recompile \( make -f
    tensorflow/lite/experimental/micro/tools/make/Makefile TARGET=ecm3531
    TAGS="CMSIS" test\) \
    &nbsp;&nbsp;&nbsp;&nbsp;cd
    tensorflow/lite/experimental/micro/tools/make/targets/ecm3531 \
    &nbsp;&nbsp;&nbsp;&nbsp;./flash_program executable_name to load into flash.

## Implement target optimized kernels

The reference kernels in tensorflow/lite/experimental/micro/kernels are
implemented in pure C/C++. It might not utilize all HW architecture specific
optimizations, such as DSP instructions etc. The instructions below provides an
example on how to compile an external lib with HW architecture specific
optimizations and link it with the microlite lib.

### CMSIS-NN optimized kernels (---under development---)

To utilize the CMSIS-NN optimized kernels, choose your target, e.g. Bluepill,
and build with:

```
make -f tensorflow/lite/experimental/micro/tools/make/Makefile TAGS=cmsis-nn TARGET=bluepill test
```

That will build the microlite lib including CMSIS-NN optimized kernels based on
the version downloaded by 'download_dependencies.sh', so make sure you have run
this script. If you want to utilize another version of CMSIS, clone it to a
custom location run the following command:

```
make -f tensorflow/lite/experimental/micro/tools/make/Makefile CMSIS_PATH=<CUSTOM_LOCATION> TAGS=cmsis-nn TARGET=bluepill test
```

To test the optimized kernel(s) on your target platform using mbed (depthwise
conv in this example), follow these steps:

1.  Clone CMSIS to a custom location (<CUSTOM_LOCATION>) url:
    https://github.com/ARM-software/CMSIS_5.git Make sure you're on the
    development branch.
2.  Generate the project for depthwise conv mbed test: `make -f
    tensorflow/lite/experimental/micro/tools/make/Makefile TAGS=cmsis-nn
    CMSIS_PATH=<CUSTOM_LOCATION> generate_depthwise_conv_test_mbed_project`
3.  Go to the generated mbed folder: `cd
    tensorflow/lite/experimental/micro/tools/make/gen/linux_x86_64/prj/depthwise_conv_test/mbed`
4.  Follow the steps in README_MBED.md to setup the environment. Or simply do:
    `mbed config root . mbed deploy python -c 'import fileinput, glob; for
    filename in glob.glob("mbed-os/tools/profiles/*.json"): for line in
    fileinput.input(filename, inplace=True):
    print(line.replace("\"-std=gnu++98\"","\"-std=gnu++11\",
    \"-fpermissive\""))'`
5.  Compile and flash. The 'auto' flag requires your target to be plugged in.
    `mbed compile -m auto -t GCC_ARM -f --source . --source
    <CUSTOM_LOCATION>/CMSIS/NN/Include --source
    <CUSTOM_LOCATION>/CMSIS/NN/Source/ConvolutionFunctions --source
    <CUSTOM_LOCATION>/CMSIS/DSP/Include --source
    <CUSTOM_LOCATION>/CMSIS/Core/Include -j8`

## Goals

The design goals are for the framework to be:

-   **Readable**: We want embedded software engineers to be able to understand
    what's required to run ML inference without having to study research papers.
    We've tried to keep the code base small, modular, and have reference
    implementations of all operations to help with this.

-   **Easy to modify**: We know that there are a lot of different platforms and
    requirements in the embedded world, and we don't expect to cover all of them
    in one framework. Instead, we're hoping that it can be a good starting point
    for developers to build on top of to meet their own needs. For example, we
    tried to make it easy to replace the implementations of key computational
    operators that are often crucial for performance, without having to touch
    the data flow and other runtime code. We want it to make more sense to use
    our workflow to handle things like model import and less-important
    operations, and customize the parts that matter, rather than having to
    reimplement everything in your own engine.

-   **Well-tested**: If you're modifying code, you need to know if your changes
    are correct. Having an easy way to test lets you develop much faster. To
    help there, we've written tests for all the components, and we've made sure
    that the tests can be run on almost any platform, with no dependencies apart
    from the ability to log text to a debug console somewhere. We also provide
    an easy way to run all the tests on-device as part of an automated test
    framework, and we use qemu/Renode emulation so that tests can be run even
    without physical devices present.

-   **Easy to integrate**: We want to be as open a system as possible, and use
    the best code available for each platform. To do that, we're going to rely
    on projects like
    [CMSIS-NN](https://www.keil.com/pack/doc/CMSIS/NN/html/index.html),
    [uTensor](https://github.com/uTensor/uTensor), and other vendor libraries to
    handle as much performance-critical code as possible. We know that there are
    an increasing number of options to accelerate neural networks on
    microcontrollers, so we're aiming to be a good host for deploying those
    hardware technologies too.

-   **Compatible**: We're using the same file schema, interpreter API, and
    kernel interface as regular TensorFlow Lite, so we leverage the large
    existing set of tools, documentation, and examples for the project. The
    biggest barrier to deploying ML models is getting them from a training
    environment into a form that's easy to run inference on, so we see reusing
    this rich ecosystem as being crucial to being easily usable. We also hope to
    integrate this experimental work back into the main codebase in the future.

To meet those goals, we've made some tradeoffs:

-   **Simple C++**: To help with readability, our code is written in a modern
    version of C++, but we generally treat it as a "better C", rather relying on
    more complex features such as template meta-programming. As mentioned
    earlier, we avoid any use of dynamic memory allocation (new/delete) or the
    standard C/C++ libraries, so we believe this should still be fairly
    portable. It does mean that some older devices with C-only toolchains won't
    be supported, but we're hoping that the reference operator implementations
    (which are simple C-like functions) can still be useful in those cases. The
    interfaces are also designed to be C-only, so it should be possible to
    integrate the resulting library with pure C projects.

-   **Interpreted**: Code generation is a popular pattern for embedded code,
    because it gives standalone code that's easy to modify and step through, but
    we've chosen to go with an interpreted approach. In our internal
    microcontroller work we've found that using an extremely stripped-down
    interpreter with almost no dependencies gives us a lot of the same
    advantages, but is easier to maintain. For example, when new updates come
    out for the underlying library, you can just merge your local modifications
    in a single step, rather than having to regenerate new code and then patch
    in any changes you subsequently made. The coarse granularity of the
    interpreted primitives means that each operation call typically takes
    hundreds of thousands of instruction cycles at least, so we don't see
    noticeable performance gains from avoiding what's essentially a single
    switch statement at the interpreter level to call each operation. We're
    still working on improving the packaging though, for example we're
    considering having the ability to snapshot all the source files and headers
    used for a particular model, being able to compile the code and data
    together as a library, and then access it through a minimal set of C
    interface calls which hide the underlying complexity.

-   **Flatbuffers**: We represent our models using
    [the standard flatbuffer schema used by the rest of TensorFlow Lite](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/schema/schema.fbs),
    with the difference that we always keep it in read-only program memory
    (typically flash) rather than relying on having a file system to read it
    from. This is a good fit because flatbuffer's serialized format is designed
    to be mapped into memory without requiring any extra memory allocations or
    modifications to access it. All of the functions to read model values work
    directly on the serialized bytes, and large sections of data like weights
    are directly accessible as sequential C-style arrays of their data type,
    with no strides or unpacking needed. We do get a lot of value from using
    flatbuffers, but there is a cost in complexity. The flat buffer library code
    is all inline
    [inside the main headers](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/schema/schema_generated.h),
    but it isn't straightforward to inspect their implementations, and the model
    data structures aren't easy to comprehend from the debugger. The header for
    the schema itself also has to be periodically updated when new information
    is added to the file format, though we try to handle that transparently for
    most developers by checking in a pre-generated version.

-   **Code Duplication**: Some of the code in this prototype largely duplicates
    the logic in other parts of the TensorFlow Lite code base, for example the
    operator wrappers. We've tried to keep share as much as we can between the
    two interpreters, but there are some assumptions built into the original
    runtime that make this difficult. We'll be working on modularizing the main
    interpreter so that we can move to an entirely shared system.

This initial preview release is designed to get early feedback, and is not
intended to be a final product. It only includes enough operations to run a
simple keyword recognition model, and the implementations are not optimized.
We're hoping this will be a good way to get feedback and collaborate to improve
the framework.

## Generating Project Files

It's not always easy or convenient to use a makefile-based build process,
especially if you're working on a product that uses a different IDE for the rest
of its code. To address that, it's possible to generate standalone project
folders for various popular build systems. These projects are self-contained,
with only the headers and source files needed by a particular binary, and
include project files to make loading them into an IDE easy. These can be
auto-generated for any target you can compile using the main Make system, using
a command like this:

```
make -f tensorflow/lite/experimental/micro/tools/make/Makefile TARGET=mbed TAGS="disco_f746ng" generate_micro_speech_mbed_project
```

This will create a folder in
`tensorflow/lite/experimental/micro/tools/make/gen/mbed_cortex-m4/prj/micro_speech_main_test/mbed`
that contains the source and header files, some Mbed configuration files, and a
README. You should then be able to copy this directory to another machine, and
use it just like any other Mbed project. There's more information about project
files [below](#working-with-generated-projects).

## Generating Arduino Libraries

It's possible to use the Arduino Desktop IDE to build TFL Micro targets for
Arduino devices. The source code is packaged as a .zip archive that you can add
in the IDE by going to Sketch->Include Library->Add .ZIP Library... Once you've
added the library, you can then go to File->Examples->TensorFlowLite to find a
simple sketches that you can use to build the examples.

You can generate the zip file from the source code here in git by running the
following command:

```
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/micro/tools/ci_build/test_arduino.sh
```

The resulting library can be found in `tensorflow/lite/experimental/micro/tools/make/gen/arduino_x86_64/prj/tensorflow_lite.zip`.
This generates a library that includes all of the examples as sketches, along
with the framework code you need to run your own examples.

## How to Port TensorFlow Lite Micro to a New Platform

Are you a hardware or operating system provider looking to run machine learning
on your platform? We're keen to help, and we've had experience helping other
teams do the same thing, so here are our recommendations.

### Requirements

Since the core neural network operations are pure arithmetic, and don't require
any I/O or other system-specific functionality, the code doesn't have to have
many dependencies. We've tried to enforce this, so that it's as easy as possible
to get TensorFlow Lite Micro running even on 'bare metal' systems without an OS.
Here are the core requirements that a platform needs to run the framework:

-   C/C++ compiler capable of C++11 compatibility. This is probably the most
    restrictive of the requirements, since C++11 is not as widely adopted in the
    embedded world as it is elsewhere. We made the decision to require it since
    one of the main goals of TFL Micro is to share as much code as possible with
    the wider TensorFlow codebase, and since that relies on C++11 features, we
    need compatibility to achieve it. We only use a small, sane, subset of C++
    though, so don't worry about having to deal with template metaprogramming or
    similar challenges!

-   Debug logging. The core network operations don't need any I/O functions, but
    to be able to run tests and tell if they've worked as expected, the
    framework needs some way to write out a string to some kind of debug
    console. This will vary from system to system, for example on Linux it could
    just be `fprintf(stderr, debug_string)` whereas an embedded device might
    write the string out to a specified UART. As long as there's some mechanism
    for outputting debug strings, you should be able to use TFL Micro on that
    platform.

-   Math library. The C standard `libm.a` library is needed to handle some of
    the mathematical operations used to calculate neural network results.

-   Global variable initialization. We do use a pattern of relying on global
    variables being set before `main()` is run in some places, so you'll need to
    make sure your compiler toolchain

And that's it! You may be wondering about some other common requirements that
are needed by a lot of non-embedded software, so here's a brief list of things
that aren't necessary to get started with TFL Micro on a new platform:

-   Operating system. Since the only platform-specific function we need is
    `DebugLog()`, there's no requirement for any kind of Posix or similar
    functionality around files, processes, or threads.

-   C or C++ standard libraries. The framework tries to avoid relying on any
    standard library functions that require linker-time support. This includes
    things like string functions, but still allows us to use headers like
    `stdtypes.h` which typically just define constants and typedefs.
    Unfortunately this distinction isn't officially defined by any standard, so
    it's possible that different toolchains may decide to require linked code
    even for the subset we use, but in practice we've found it's usually a
    pretty obvious decision and stable over platforms and toolchains.

-   Dynamic memory allocation. All the TFL Micro code avoids dynamic memory
    allocation, instead relying on local variables on the stack in most cases,
    or global variables for a few situations. These are all fixed-size, which
    can mean some compile-time configuration to ensure there's enough space for
    particular networks, but does avoid any need for a heap and the
    implementation of `malloc\new` on a platform.

-   Floating point. Eight-bit integer arithmetic is enough for inference on many
    networks, so if a model sticks to these kind of quantized operations, no
    floating point instructions should be required or executed by the framework.

### Getting Started

We recommend that you start trying to compile and run one of the simplest tests
in the framework as your first step. The full TensorFlow codebase can seem
overwhelming to work with at first, so instead you can begin with a collection
of self-contained project folders that only include the source files needed for
a particular test or executable. You can find a set of pre-generated projects
[here](https://drive.google.com/open?id=1cawEQAkqquK_SO4crReDYqf_v7yAwOY8).

As mentioned above, the one function you will need to implement for a completely
new platform is debug logging. If your device is just a variation on an existing
platform you may be able to reuse code that's already been written. To
understand what's available, begin with the default reference implementation at
[tensorflow/lite/experimental/micro/debug_log.cc](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/micro/debug_log.cc),
which uses fprintf and stderr. If your platform has this level of support for
the C standard library in its toolchain, then you can just reuse this.
Otherwise, you'll need to do some research into how your platform and device can
communicate logging statements to the outside world. As another example, take a
look at
[the Mbed version of `DebugLog()`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/micro/mbed/debug_log.cc),
which creates a UART object and uses it to output strings to the host's console
if it's connected.

Begin by navigating to the micro_error_reporter_test folder in the pregenerated
projects you downloaded. Inside here, you'll see a set of folders containing all
the source code you need. If you look through them, you should find a total of
around 60 C or C++ files that compiled together will create the test executable.
There's an example makefile in the directory that lists all of the source files
and include paths for the headers. If you're building on a Linux or MacOS host
system, you may just be able to reuse that same makefile to cross-compile for
your system, as long as you swap out the `CC` and `CXX` variables from their
defaults, to point to your cross compiler instead (for example
`arm-none-eabi-gcc` or `riscv64-unknown-elf-gcc`). Otherwise, set up a project
in the build system you are using. It should hopefully be fairly
straightforward, since all of the source files in the folder need to be
compiled, so on many IDEs you can just drag the whole lot in. Then you need to
make sure that C++11 compatibility is turned on, and that the right include
paths (as mentioned in the makefile) have been added.

You'll see the default `DebugLog()` implementation in
'tensorflow/lite/experimental/micro/debug_log.cc' inside the
micro_error_reporter_test folder. Modify that file to add the right
implementation for your platform, and then you should be able to build the set
of files into an executable. Transfer that executable to your target device (for
example by flashing it), and then try running it. You should see output that
looks something like this:

```
Number: 42
Badly-formed format string
Another  badly-formed  format string
~~ALL TESTS PASSED~~~
```

If not, you'll need to debug what went wrong, but hopefully with this small
starting project it should be manageable.

### Troubleshooting

When we've been porting to new platforms, it's often been hard to figure out
some of the fundamentals like linker settings and other toolchain setup flags.
If you are having trouble, see if you can find a simple example program for your
platform, like one that just blinks an LED. If you're able to build and run that
successfully, then start to swap in parts of the TF Lite Micro codebase to that
working project, taking it a step at a time and ensuring it's still working
after every change. For example, a first step might be to paste in your
`DebugLog()` implementation and call `DebugLog("Hello World!")` from the main
function.

Another common problem on embedded platforms is the stack size being too small.
Mbed defaults to 4KB for the main thread's stack, which is too small for most
models since TensorFlow Lite allocates buffers and other data structures that
require more memory. The exact size will depend on which model you're running,
but try increasing it if you are running into strange corruption issues that
might be related to stack overwriting.

### Optimizing for your Platform

The default reference implementations in TensorFlow Lite Micro are written to be
portable and easy to understand, not fast, so you'll want to replace performance
critical parts of the code with versions specifically tailored to your
architecture. The framework has been designed with this in mind, and we hope the
combination of small modules and many tests makes it as straightforward as
possible to swap in your own code a piece at a time, ensuring you have a working
version at every step. To write specialized implementations for a platform, it's
useful to understand how optional components are handled inside the build
system.

### Code Module Organization

We have adopted a system of small modules with platform-specific implementations
to help with portability. Every module is just a standard `.h` header file
containing the interface (either functions or a class), with an accompanying
reference implementation in a `.cc` with the same name. The source file
implements all of the code that's declared in the header. If you have a
specialized implementation, you can create a folder in the same directory as the
header and reference source, name it after your platform, and put your
implementation in a `.cc` file inside that folder. We've already seen one
example of this, where the Mbed and Bluepill versions of `DebugLog()` are inside
[mbed](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/micro/mbed)
and
[bluepill](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/micro/bluepill)
folders, children of the
[same directory](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/micro)
where the stdio-based
[`debug_log.cc`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/micro/debug_log.cc)
reference implementation is found.

The advantage of this approach is that we can automatically pick specialized
implementations based on the current build target, without having to manually
edit build files for every new platform. It allows incremental optimizations
from a always-working foundation, without cluttering the reference
implementations with a lot of variants.

To see why we're doing this, it's worth looking at the alternatives. TensorFlow
Lite has traditionally used preprocessor macros to separate out some
platform-specific code within particular files, for example:

```
#ifndef USE_NEON
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#define USE_NEON
#include <arm_neon.h>
#endif
```

There’s also a tradition in gemmlowp of using file suffixes to indicate
platform-specific versions of particular headers, with kernel_neon.h being
included by kernel.h if `USE_NEON` is defined. As a third variation, kernels are
separated out using a directory structure, with
tensorflow/lite/kernels/internal/reference containing portable implementations,
and tensorflow/lite/kernels/internal/optimized holding versions optimized for
NEON on Arm platforms.

These approaches are hard to extend to multiple platforms. Using macros means
that platform-specific code is scattered throughout files in a hard-to-find way,
and can make following the control flow difficult since you need to understand
the macro state to trace it. For example, I temporarily introduced a bug that
disabled NEON optimizations for some kernels when I removed
tensorflow/lite/kernels/internal/common.h from their includes, without realizing
it was where USE_NEON was defined!

It’s also tough to port to different build systems, since figuring out the right
combination of macros to use can be hard, especially since some of them are
automatically defined by the compiler, and others are only set by build scripts,
often across multiple rules.

The approach we are using extends the file system approach that we use for
kernel implementations, but with some specific conventions:

-   For each module in TensorFlow Lite, there will be a parent directory that
    contains tests, interface headers used by other modules, and portable
    implementations of each part.
-   Portable means that the code doesn’t include code from any libraries except
    flatbuffers, or other TF Lite modules. You can include a limited subset of
    standard C or C++ headers, but you can’t use any functions that require
    linking against those libraries, including fprintf, etc. You can link
    against functions in the standard math library, in <math.h>.
-   Specialized implementations are held inside subfolders of the parent
    directory, named after the platform or library that they depend on. So, for
    example if you had my_module/foo.cc, a version that used RISC-V extensions
    would live in my_module/riscv/foo.cc. If you had a version that used the
    CMSIS library, it should be in my_module/cmsis/foo.cc.
-   These specialized implementations should completely replace the top-level
    implementations. If this involves too much code duplication, the top-level
    implementation should be split into smaller files, so only the
    platform-specific code needs to be replaced.
-   There is a convention about how build systems pick the right implementation
    file. There will be an ordered list of 'tags' defining the preferred
    implementations, and to generate the right list of source files, each module
    will be examined in turn. If a subfolder with a tag’s name contains a .cc
    file with the same base name as one in the parent folder, then it will
    replace the parent folder’s version in the list of build files. If there are
    multiple subfolders with matching tags and file names, then the tag that’s
    latest in the ordered list will be chosen. This allows us to express “I’d
    like generically-optimized fixed point if it’s available, but I’d prefer
    something using the CMSIS library” using the list 'fixed_point cmsis'. These
    tags are passed in as `TAGS="<foo>"` on the command line when you use the
    main Makefile to build.
-   There is an implicit “reference” tag at the start of every list, so that
    it’s possible to support directory structures like the current
    tensorflow/kernels/internal where portable implementations are held in a
    “reference” folder that’s a sibling to the NEON-optimized folder.
-   The headers for each unit in a module should remain platform-agnostic, and
    be the same for all implementations. Private headers inside a sub-folder can
    be used as needed, but shouldn’t be referred to by any portable code at the
    top level.
-   Tests should be at the parent level, with no platform-specific code.
-   No platform-specific macros or #ifdef’s should be used in any portable code.

The implementation of these rules is handled inside the Makefile, with a
[`specialize` function](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/micro/tools/make/helper_functions.inc#L42)
that takes a list of reference source file paths as an input, and returns the
equivalent list with specialized versions of those files swapped in if they
exist.

### Working with Generated Projects

So far, I've recommended that you use the standalone generated projects for your
system. You might be wondering why you're not just checking out the full
[TensorFlow codebase from GitHub](https://github.com/tensorflow/tensorflow/)?
The main reason is that there is a lot more diversity of architectures, IDEs,
support libraries, and operating systems in the embedded world. Many of the
toolchains require their own copy of source files, or a list of sources to be
written to a project file. When a developer working on TensorFlow adds a new
source file or changes its location, we can't expect her to update multiple
different project files, many of which she may not have the right software to
verify the change was correct. That means we have to rely on a central listing
of source files (which in our case is held in the makefile), and then call a
tool to generate other project files from those. We could ask embedded
developers to do this process themselves after downloading the main source, but
running the makefile requires a Linux system which may not be available, takes
time, and involves downloading a lot of dependencies. That is why we've opted to
make regular snapshots of the results of generating these projects for popular
IDEs and platforms, so that embedded developers have a fast and friendly way to
start using TensorFlow Lite for Microcontrollers.

This does have the disadvantage that you're no longer working directly on the
main repository, instead you have a copy that's outside of source control. We've
tried to make the copy as similar to the main repo as possible, for example by
keeping the paths of all source files the same, and ensuring that there are no
changes between the copied files and the originals, but it still makes it
tougher to sync as the main repository is updated. There are also multiple
copies of the source tree, one for each target, so any change you make to one
copy has to be manually propagated across all the other projects you care about.
This doesn't matter so much if you're just using the projects as they are to
build products, but if you want to support a new platform and have the changes
reflected in the main code base, you'll have to do some extra work.

As an example, think about the `DebugLog()` implementation we discussed adding
for a new platform earlier. At this point, you have a new version of
`debug_log.cc` that does what's required, but how can you share that with the
wider community? The first step is to pick a tag name for your platform. This
can either be the operating system (for example 'mbed'), the name of a device
('bluepill'), or some other text that describes it. This should be a short
string with no spaces or special characters. Log in or create an account on
GitHub, fork the full
[TensorFlow codebase](https://github.com/tensorflow/tensorflow/) using the
'Fork' button on the top left, and then grab your fork by using a command like
`git clone https://github.com/<your user name>/tensorflow`.

You'll either need Linux, MacOS, or Windows with something like CygWin installed
to run the next steps, since they involve building a makefile. Run the following
commands from a terminal, inside the root of the source folder:

```
tensorflow/lite/experimental/micro/tools/make/download_dependencies.sh
make -f tensorflow/lite/experimental/micro/tools/make/Makefile generate_projects
```

This will take a few minutes, since it has to download some large toolchains for
the dependencies. Once it has finished, you should see some folders created
inside a path like
`tensorflow/lite/experimental/micro/tools/make/gen/linux_x86_64/prj/`. The exact
path depends on your host operating system, but you should be able to figure it
out from all the copy commands. These folders contain the generated project and
source files, with
`tensorflow/lite/experimental/micro/tools/make/gen/linux_x86_64/prj/keil`
containing the Keil uVision targets,
`tensorflow/lite/experimental/micro/tools/make/gen/linux_x86_64/prj/mbed` with
the Mbed versions, and so on.

If you've got this far, you've successfully set up the project generation flow.
Now you need to add your specialized implementation of `DebugLog()`. Start by
creating a folder inside `tensorflow/lite/experimental/micro/` named after the
tag you picked earlier. Put your `debug_log.cc` file inside this folder, and
then run this command, with '<your tag>' replaced by the actual folder name:

```
make -f tensorflow/lite/experimental/micro/tools/make/Makefile TAGS="<your tag>" generate_projects
```

If your tag name actually refers to a whole target architecture, then you'll use
TARGET or TARGET_ARCH instead. For example, here's how a simple RISC-V set of
projects is generated:

```
make -f tensorflow/lite/experimental/micro/tools/make/Makefile TARGET="riscv32_mcu" generate_projects
```

The way it works is the same as TAGS though, it just looks for specialized
implementations with the same containing folder name.

If you look inside the projects that have been created, you should see that the
default `DebugLog()` implementation is no longer present at
`tensorflow/lite/experimental/micro/debug_log.cc`, and instead
`tensorflow/lite/experimental/micro/<your tag>/debug_log.cc` is being used. Copy
over the generated project files and try building them in your own IDE. If
everything works, then you're ready to submit your change.

To do this, run something like:

```
git add tensorflow/lite/experimental/micro/<your tag>/debug_log.cc
git commit -a -m "Added DebugLog() support for <your platform>"
git push origin master
```

Then go back to `https://github.com/<your account>/tensorflow`, and choose "New
Pull Request" near the top. You should then be able to go through the standard
TensorFlow PR process to get your change added to the main repository, and
available to the rest of the community!

### Supporting a Platform with Makefiles

The changes you've made so far will enable other developers using the generated
projects to use your platform, but TensorFlow's continuous integration process
uses makefiles to build frequently and ensure changes haven't broken the build
process for different systems. If you are able to convert your build procedure
into something that can be expressed by a makefile, then we can integrate your
platform into our CI builds and make sure it continues to work.

Fully describing how to do this is beyond the scope of this documentation, but
the biggest needs are:

-   A command-line compiler that can be called for every source file.
-   A list of the arguments to pass into the compiler to build and link all
    files.
-   The correct linker map files and startup assembler to ensure `main()` gets
    called.

### Supporting a Platform with Emulation Testing

Integrating your platform into the makefile process should help us make sure
that it continues to build, but it doesn't guarantee that the results of the
build process will run correctly. Running tests is something we require to be
able to say that TensorFlow officially supports a platform, since otherwise we
can't guarantee that users will have a good experience when they try using it.
Since physically maintaining a full set of all supported hardware devices isn't
feasible, we rely on software emulation to run these tests. A good example is
our
[STM32F4 'Bluepill' support](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/micro/testing/test_bluepill_binary.sh),
which uses [Docker](https://www.docker.com/) and [Renode](https://renode.io/) to
run built binaries in an emulator. You can use whatever technologies you want,
the only requirements are that they capture the debug log output of the tests
being run in the emulator, and parse them for the string that indicates the test
was successful. These scripts need to run on Ubuntu 18.04, in a bash
environment, though Docker is available if you need to install extra software or
have other dependencies.

### Implementing More Optimizations

Clearly, getting debug logging support is only the beginning of the work you'll
need to do on a particular platform. It's very likely that you'll want to
optimize the core deep learning operations that take up the most time when
running models you care about. The good news is that the process for providing
optimized implementations is the same as the one you just went through to
provide your own logging. You'll need to identify parts of the code that are
bottlenecks, and then add specialized implementations in their own folders.
These don't need to be platform specific, they can also be broken out by which
library they rely on for example. [Here's where we do that for the CMSIS
implementation of integer fast-fourier
transforms](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/micro/examples/micro_speech/simple_features/simple_features_generator.cc).
This more complex case shows that you can also add helper source files alongside
the main implementation, as long as you
[mention them in the platform-specific makefile](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/micro/examples/micro_speech/CMSIS/Makefile.inc).
You can also do things like update the list of libraries that need to be linked
in, or add include paths to required headers.
