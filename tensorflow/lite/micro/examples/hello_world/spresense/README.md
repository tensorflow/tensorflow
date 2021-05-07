# Hello World Example for Spresense

Here explaines how to build and execute this Hello World Example for Spresense.
To try this on the Spresense, below hardware is required.

Spresense Main board, which is a microcontroller board.

## Table of contents

-   [How to build](#how-to-build)
-   [How to run](#how-to-run)

## How to build

The tensorflow.git will be downloaded in build system of Spresense.

### Initial setup

The Spresense SDK build system is required to build this example. The following
instructions will help you to make it on your PC.
[Spresense SDK Getting Started Guide:EN](https://developer.sony.com/develop/spresense/docs/sdk_set_up_en.html)
[Spresense SDK Getting Started Guide:JA](https://developer.sony.com/develop/spresense/docs/sdk_set_up_ja.html)
[Spresense SDK Getting Started Guide:CN](https://developer.sony.com/develop/spresense/docs/sdk_set_up_zh.html)

And after setup the build system, download
[Spresense repository](https://github.com/sonydevworld/spresense).

```
git clone --recursive https://github.com/sonydevworld/spresense.git
```

### Configure Spresense for this example

The Spresense SDK uses Kconfig mechanism for configuration of software
components. So at first, you need to configure it for this example. Spresense
SDK provides some default configurations, and there is a default config to build
this Hello World example.

1.  Go to sdk/ directory in the repository.

    ```
    cd spresense/sdk
    ```

2.  Execute config.py to configure for this example.

    ```
    ./tools/config.py examples/tf_example_hello_world
    ```

This command creates .config file in spesense/nuttx directory.

### Build and Flash the binary into Spresense Main board

After configured, execute make and then flash built image.

1.  Execute "make" command in the same directory you configured.

    ```
    make
    ```

2.  Flash built image into Spresense main board. If the build is successful, a
    file named nuttx.spk will be created in the current directory, and flash it
    into Spresense Main board. Make sure USB cable is connected between the
    board and your PC. The USB will be recognized as USB/serial device like
    /dev/ttyUSB0 in your PC. In this explanation, we will assume that the device
    is recognized as /dev/ttyUSB0.

    ```
    ./tools/flash.sh -c /dev/ttyUSB0 nuttx.spk
    ```

## How to run

To run the example, connect to the device with a terminal soft like "minicom".
Then you can see a "nsh>" prompt on it. (If you can't see the prompt, try to
press enter.)

1.  Execute tf_example command on the prompt.

    ```
    nsh> tf_example
    ```
