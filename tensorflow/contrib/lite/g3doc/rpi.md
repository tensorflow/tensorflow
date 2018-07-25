book_path: /mobile/_book.yaml
project_path: /mobile/_project.yaml

# TensorFlow Lite for Raspberry Pi

## Cross compiling
### Installing toolchian
This has been tested on Ubuntu 16.04.3 64bit and Tensorflow devel docker image [tensorflow/tensorflow:nightly-devel](https://hub.docker.com/r/tensorflow/tensorflow/tags/).

To cross compiling TensorFlow Lite. First you should install the toolchain and libs.
```bash
sudo apt-get update
sudo apt-get install crossbuild-essential-armhf
```
> If you are using docker, you may not use `sudo`

### Building
Clone this Tensorflow repository, Run this script at the root of the repository to download all the dependencies:
> The Tensorflow repository is in `/tensorflow` if you are using `tensorflow/tensorflow:nightly-devel` docker image, just try it.
```bash
./tensorflow/contrib/lite/download_dependencies.sh
```
Note than you only need to to this once.

You should then be able to compile:
```bash
./tensorflow/contrib/lite/build_rpi_lib.sh
```

This should compile a static library in:
`tensorflow/contrib/lite/gen/lib/rpi_armv7/libtensorflow-lite.a`.

## Native compiling
This has been tested on Raspberry Pi 3b, Raspbian GNU/Linux 9.1 (stretch), gcc version 6.3.0 20170516 (Raspbian 6.3.0-18+rpi1).

Log in to you RPI, install the toolchain.
```bash
sudo apt-get install build-essential
```

First, clone this TensorFlow repository. Run this at the root of the repository:
```bash
./tensorflow/contrib/lite/download_dependencies.sh
```
Note than you only need to to this once.

You should then be able to compile:
```bash
./tensorflow/contrib/lite/build_rpi_lib.sh
```

This should compile a static library in:
`tensorflow/contrib/lite/gen/lib/rpi_armv7/libtensorflow-lite.a`.
