# TensorFlow Lite for Microcontrollers

This an experimental port of TensorFlow Lite aimed at micro controllers and other devices with only kilobytes of memory. It doesn't require any operating system support, any standard C or C++ libraries, or dynamic memory allocation, so it's designed to be portable even to 'bare metal' systems. The core runtime fits in 16KB on a Cortex M3, and with enough operators to run a speech keyword detection model, takes up a total of 22KB.

The design goals are for the framework to be:

- **Readable**: We want embedded software engineers to be able to understand what's required to run ML inference without having to study research papers. We've tried to keep the code base small, modular, and have reference implementations of all operations to help with this.

- **Easy to modify**: We know that there are a lot of different platforms and requirements in the embedded world, and we don't expect to cover all of them in one framework. Instead, we're hoping that it can be a good starting point for developers to build on top of to meet their own needs. For example, we tried to make it easy to replace the implementations of key computational operators that are often crucial for performance, without having to touch the data flow and other runtime code. We want it to make more sense to use our workflow to handle things like model import and less-important operations, and customize the parts that matter, rather than having to reimplement everything in your own engine.

- **Well-tested**: If you're modifying code, you need to know if your changes are correct. Having an easy way to test lets you develop much faster. To help there, we've written tests for all the components, and we've made sure that the tests can be run on almost any platform, with no dependencies apart from the ability to log text to a debug console somewhere. We also provide an easy way to run all the tests on-device as part of an automated test framework, and we use qemu/Renode emulation so that tests can be run even without physical devices present.

- **Easy to integrate**: We want to be as open a system as possible, and use the best code available for each platform. To do that, we're going to rely on projects like [CMSIS-NN](https://www.keil.com/pack/doc/CMSIS/NN/html/index.html), [uTensor](https://github.com/uTensor/uTensor), and other vendor libraries to handle as much performance-critical code as possible. We know that there are an increasing number of options to accelerate neural networks on microcontrollers, so we're aiming to be a good host for deploying those hardware technologies too.

- **Compatible**: We're using the same file schema, interpreter API, and kernel interface as regular TensorFlow Lite, so we leverage the large existing set of tools, documentation, and examples for the project. The biggest barrier to deploying ML models is getting them from a training environment into a form that's easy to run inference on, so we see reusing this rich ecosystem as being crucial to being easily usable. We also hope to integrate this experimental work back into the main codebase in the future.

To meet those goals, we've made some tradeoffs:

- **Simple C++**: To help with readability, our code is written in a modern version of C++, but we generally treat it as a "better C", rather relying on more complex features such as template meta-programming. As mentioned earlier, we avoid any use of dynamic memory allocation (new/delete) or the standard C/C++ libraries, so we believe this should still be fairly portable. It does mean that some older devices with C-only toolchains won't be supported, but we're hoping that the reference operator implementations (which are simple C-like functions) can still be useful in those cases. The interfaces are also designed to be C-only, so it should be possible to integrate the resulting library with pure C projects.

- **Interpreted**: Code generation is a popular pattern for embedded code, because it gives standalone code that's easy to modify and step through, but we've chosen to go with an interpreted approach. In our internal microcontroller work we've found that using an extremely stripped-down interpreter with almost no dependencies gives us a lot of the same advantages, but is easier to maintain. For example, when new updates come out for the underlying library, you can just merge your local modifications in a single step, rather than having to regenerate new code and then patch in any changes you subsequently made. The coarse granularity of the interpreted primitives means that each operation call typically takes hundreds of thousands of instruction cycles at least, so we don't see noticeable performance gains from avoiding what's essentially a single switch statement at the interpreter level to call each operation. We're still working on improving the packaging though, for example we're considering having the ability to snapshot all the source files and headers used for a particular model, being able to compile the code and data together as a library, and then access it through a minimal set of C interface calls which hide the underlying complexity.

- **Flatbuffers**: We represent our models using [the standard flatbuffer schema used by the rest of TensorFlow Lite](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/schema/schema.fbs), with the difference that we always keep it in read-only program memory (typically flash) rather than relying on having a file system to read it from. This is a good fit because flatbuffer's serialized format is designed to be mapped into memory without requiring any extra memory allocations or modifications to access it. All of the functions to read model values work directly on the serialized bytes, and large sections of data like weights are directly accessible as sequential C-style arrays of their data type, with no strides or unpacking needed. We do get a lot of value from using flatbuffers, but there is a cost in complexity. The flat buffer library code is all inline [inside the main headers](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/schema/schema_generated.h), but it isn't straightforward to inspect their implementations, and the model data structures aren't easy to comprehend from the debugger. The header for the schema itself also has to be periodically updated when new information is added to the file format, though we try to handle that transparently for most developers by checking in a pre-generated version.

- **Code Duplication**: Some of the code in this prototype largely duplicates the logic in other parts of the TensorFlow Lite code base, for example the operator wrappers. We've tried to keep share as much as we can between the two interpreters, but there are some assumptions built into the original runtime that make this difficult. We'll be working on modularizing the main interpreter so that we can move to an entirely shared system.

This initial preview release is designed to get early feedback, and is not intended to be a final product. It only includes enough operations to run a simple keyword recognition model, and the implementations are not optimized. We're hoping this will be a good way to get feedback and collaborate to improve the framework.

## Getting Started

Building requires a Linux or OS X machine.

 - Open a terminal
 - Download the TensorFlow source with `git clone https://github.com/tensorflow/tensorflow.git`
 - Enter the source root directory by running `cd tensorflow`
 - Download the dependencies by running `tensorflow/lite/experimental/micro/tools/make/download_dependencies.sh`. This may take a few minutes
 - Build and test the library with `make -f tensorflow/lite/experimental/micro/tools/make/Makefile test`

You should see a series of compilation steps, followed by `~~~ALL TESTS
PASSED~~~` for the various tests of the code that it will run. If there's an
error, you should get an informative message from make about what went wrong.

These tests are all built as simple binaries with few dependencies, so you can run them manually. For example, here's how to run the depthwise convolution test, and its output:

```
tensorflow/lite/experimental/micro/tools/make/gen/linux_x86_64/bin/tensorflow/lite/experimental/micro/kernels/depthwise_conv_test

Testing SimpleTest
Testing SimpleTestQuantized
Testing SimpleTestRelu
Testing SimpleTestReluQuantized
4/4 tests passed
~ALL TESTS PASSED~~~
```

Looking at the [depthwise_conv_test.cc](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/micro/kernels/depthwise_conv_test.cc) code, you'll see a sequence that looks like this:

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

So, why are we running tests in this complicated way? So far, we've been building binaries that run locally on the Mac OS or Linux machine you're building on, but this approach becomes important when we're targeting simple micro controller devices.

## Building for the "Blue Pill" STM32F103

The goal of this library is to enable machine learning on resource-constrained micro controllers and DSPs, and as part of that we've targeted the ["Blue Pill" STM32F103-compatible development board](https://github.com/google/stm32_bare_lib) as a cheap and popular platform. It only has 20KB of RAM and 64KB of flash, so it's a good device to ensure we can run efficiently on small chips.

It's fairly easy to [buy and wire up a physical board](https://github.com/google/stm32_bare_lib#wiring-up-your-blue-pill), but even if you don't have an actual device, the [Renode project](https://renode.io/) makes it easy to run a faithful emulation on your desktop machine. You'll need [Docker](https://www.docker.com/) installed, but once you have that set up, try running the following command:

`make -f tensorflow/lite/experimental/micro/tools/make/Makefile TARGET=bluepill test`

You should see a similar set of outputs as you did in the previous section, with the addition of some extra Docker logging messages. These are because we're using Docker to run the Renode micro controller emulation tool, and the tests themselves are being run on a simulated STM32F103 device. The communication channels between an embedded device and the host are quite limited, so the test harness looks at the output of the debug log to see if tests have passed, just as it did in the previous section. This makes it a very flexible way to run cross-platform tests, even when a platform has no operating system facilities, as long as it can output debugging text logs.

To understand what's happening here, try running the same depthwise convolution test, but through the emulated device test harness, with the following command:

```
tensorflow/lite/experimental/micro/testing/test_bluepill_binary.sh \
tensorflow/lite/experimental/micro/tools/make/gen/bluepill_cortex-m3/bin/tensorflow/lite/experimental/micro/kernels/depthwise_conv_test \
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
tensorflow/lite/experimental/micro/tools/make/gen/bluepill_cortex-m3/bin/tensorflow/lite/experimental/micro/kernels/depthwise_conv_test: PASS
```

There's a lot of output here, but you should be able to see that the same tests
that were covered when we ran locally on the development machine show up in the
debug logs here, along with the magic string `~~~ALL TESTS PASSED~~~`. This is
the exact same code as before, just compiled and run on the STM32F103 rather
than your desktop. We hope that the simplicity of this testing approach will
help make adding support for new platforms as easy as possible.

## Building for "Hifive1" SiFive FE310 development board
We've targeted the ["HiFive1" Arduino-compatible development board](https://www.sifive.com/boards/hifive1) as a test platform for RISC-V MCU.

Similar to Blue Pill setup, you will need Docker installed. The binary can be executed on either HiFive1 board or emulated using [Renode project](https://renode.io/) on your desktop machine.

The following instructions builds and transfers the source files to the Docker
```
docker build -t riscv_build \
-f {PATH_TO_TENSORFLOW_ROOT_DIR}/tensorflow/lite/experimental/micro/testing/Dockerfile.riscv \
{PATH_TO_TENSORFLOW_ROOT_DIR}/tensorflow/lite/experimental/micro/testing/
```

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

 - Lauch the Docker that we just created using: `docker run -it-v /tmp/copybara_out:/workspace riscv_build:latest bash`
 - Enter the source root directory by running `cd /workspace`
 - Download the dependencies by running `./tensorflow/lite/experimental/micro/tools/make/download_dependencies.sh`. This may take a few minutes.
 - Set the path to RISC-V tools: `export PATH=${PATH}:/workspace/tensorflow/lite/experimental/micro/tools/make/downloads/riscv_toolchain/bin/`
 - Build the binary: `make -f tensorflow/lite/experimental/micro/tools/make/Makefile TARGET=riscv32_mcu`

Lauching Renode to test the binary, currently this set up is not automated.

 - Until https://github.com/renode/renode/pull/30 is in the Docker image, patch the change manully: `sed -E -i 's/"rv32g"/"rv32imac"/g' /opt/renode/platforms/cpus/sifive-fe310.repl`


 - Execute the binary on Renode: `renode -P 5000 --disable-xwt -e 's @/workspace/tensorflow/lite/experimental/micro/testing/sifive_fe310.resc'`

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

## Building for Ambiq Micro Apollo3Blue EVB

Follow these steps to get the pushbutton yes/no example working on Apollo 3:

1.  Make sure to run the "Getting Started" section before performing the
    following steps
2.  Download Apollo3-SDK-2018.08.13 and place in
    `tensorflow/lite/experimental/micro/tools/make/downloads`. This is not yet
    publicly released, but you can contact ashah@ambiqmicro.com to request a
    copy.
3.  Compile the project with the following command: make -f
    tensorflow/lite/experimental/micro/tools/make/Makefile TARGET=apollo3evb
    pushbutton_cmsis_speech_test_bin
4.  Install [Segger JLink tools](https://www.segger.com/downloads/jlink/)
5.  Connect the Apollo3 EVB (with mic shield in slot 3 of Microbus Shield board)
    to the computer and power it on.
6.  Start the GDB server in a new terminal with the following command:
    JLinkGDBServer -select USB -device AMA3B1KK-KBR -endian little -if SWD
    -speed 1000 -noir -noLocalhostOnly
    1.  The command has run successfully if you see the message "Waiting for GDB
        connection"
7.  Back in the original terminal, run the program via the debugger
    1.  Navigate to
        tensorflow/lite/experimental/micro/examples/micro_speech/apollo3
    2.  Start gdb by entering the following command: arm-none-eabi-gdb
    3.  Run the command script by entering the following command: source
        pushbutton_cmsis_scores.cmd. This script does the following:
        1.  Load the binary created in step 6
        2.  Set a breakpoint after inference scores have been computed
        3.  Tell the debugger what variables should be printed out at this
            breakpoint
        4.  Begin program execution
        5.  Press Ctrl+c to exit
    4.  Press BTN2. An LED will flash for 1 second. Speak your utterance during
        this one second
    5.  The debugger will print out four numbers. They are the probabilites for
        1) no speech, 2) unknown speech, 3) yes, 4) no
    6.  The EVB LEDs will indicate detection.
        1.  LED0 (rightmost LED) - ON when capturing 1sec of audio
        2.  LED1 - ON when detecting silence
        3.  LED2 - ON when detecting UNKNOWN utterance
        4.  LED3 - ON when detecting YES utterance
        5.  LED4 (leftmost LED) - ON when detecting NO utterance

### Additional Apollo3 Instructions

To flash a part with JFlash Lite, do the following: 1. At the command line:
JFlashLiteExe 2. Device = AMA3B1KK-KBR 3. Interface = SWD at 1000 kHz 4. Data
file =
tensorflow/lite/experimental/micro/tools/make/gen/apollo3evb_cortex-m4/bin/pushbutton_cmsis_speech_test.bin
5. Prog Addr = 0x0000C000
