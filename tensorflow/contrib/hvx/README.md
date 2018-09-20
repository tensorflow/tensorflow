# TensorFlow Runtime with HVX Acceleration

This README explain how to build and use the TensorFlow runtime with HVX Acceleration. HVX is an extension of Hexagon, a DSP provided by Qualcomm, which can compute vector calculations faster using less energy than ARM processors.

## Dependencies

* [Android SDK](https://developer.android.com/studio/index.html).
* [Android NDK](https://developer.android.com/ndk/index.html). Save the path in `${NDK_ROOT}`.
* A rooted Qualcomm-based Android device connected to the computer (preferably, a [Snapdragon Development Board](https://developer.qualcomm.com/hardware/additional-snapdragon), but it could be a rooted phone with a Qualcomm SoC, albeit this guide may not work with it). The device needs to be rooted for development and testing purposes, and shouldn't be needed in production. See [Behold, The Snapdragon MDP](https://developer.qualcomm.com/blog/behold-snapdragon-mdp) for more information.
* [Hexagon SDK v3.0](https://developer.qualcomm.com/software/hexagon-dsp-sdk/tools). Save the path in `${QUALCOMM_SDK}`.
* The current directory should be TensorFlow source code (`git clone https://github.com/tensorflow/tensorflow.git && cd tensorflow`), and saved into `${TF_ROOT_DIR}`.

You may also need to add a test signature in the device to run HVX-based binaries. Follow the instructions in `${QUALCOMM_SDK}/docs/Tools_Signing.html`, using Python 2.

Note that if the device is not rooted, you may not be able to get the serial number, push the test signature and/or run binary files that call HVX libraries.

## Quick Start Guide

We provide several tools to build and run inference with this runtime quickly.

### Run inception model with a prebuilt Hexagon library

If you don’t need to build your own implementation of Hexagon HVX, we provide a shortcut to execute graphs by using pre-compiled binaries.

```shell
./tensorflow/contrib/makefile/samples/build_and_run_inception_hexagon.sh -p
```

The `-p` option makes the script download dependencies (i.e., Hexagon HVX binaries and graphs models), copy them to the Android device and execute a test.

### Run inception model by building all from the source code

If you want to build your own implementation of Hexagon HVX, we provide a sample all-in-one script to execute graphs which downloads the source and builds everything that's necessary.

```shell
./tensorflow/contrib/makefile/samples/build_and_run_inception_hexagon.sh
```

## Building libraries

If you've finished walking through the quick start guide, you may want to try building each binary manually.

### Build libhexagon\_nn\_skel.so

Download Hexagon NN library from codeaurora.org and build it. For Hexagon SDK 3.0, we need use the compatible version([721b2d58f](https://source.codeaurora.org/quic/hexagon_nn/nnlib/commit/?id=721b2d58f0f4e2d5b182f41e6b7c4db5356bf0fb)) of nnlib.

```shell
git clone https://source.codeaurora.org/quic/hexagon_nn/nnlib
cd nnlib
git reset 721b2d58f --hard
```

Just follow the instructions in `README.HOW_TO_BUILD`. You can find the file `libhexagon_nn_skel.so` in `hexagon_Release_dynamic_toolv72_v60/ship`.
Then copy the generated binary to `${GEN_LIBS_DIR}`.

```shell
GEN_LIBS_DIR="/path/to/a/dir/to/store/hexagon/libraries"
cp -v "hexagon_Release_dynamic_toolv72_v60/ship/libhexagon_nn_skel.so" "${GEN_LIBS_DIR}"
```

### Build libhexagon\_controller.so

Download tensorflow and build hexagon controller.

```shell
GENERATED_NNLIB_DIRECTORY="/path/to/nnlib"
GENERATED_HEXAGON_CONTROLLER_DIRECTORY="${QUALCOMM_SDK}/examples/common/generated_hexagon_controller"
rm -rf "${GENERATED_HEXAGON_CONTROLLER_DIRECTORY}"
cp -af "${TF_ROOT_DIR}/tensorflow/contrib/hvx/hexagon_controller" \
   "${GENERATED_HEXAGON_CONTROLLER_DIRECTORY}"
cp -afv "${GENERATED_NNLIB_DIRECTORY}/interface" \
"${GENERATED_HEXAGON_CONTROLLER_DIRECTORY}/"
cp -afv "${GENERATED_NNLIB_DIRECTORY}/glue" \
"${GENERATED_HEXAGON_CONTROLLER_DIRECTORY}/"
make clean V=android_Release
rm -rf android_Release
make tree VERBOSE=1 V=android_Release
cp -v "${GENERATED_HEXAGON_CONTROLLER_DIRECTORY}/android_Release/ship/libhexagon_controller.so" "${GEN_LIBS_DIR}"
```

### Build TensorFlow linking Hexagon library

Build TensorFlow with `build_all_android.sh` specifying the `-x` option.

```shell
BUILD_ALL_ANDROID_PATH="${TF_ROOT_DIR}/tensorflow/contrib/makefile/build_all_android.sh"

CC_PREFIX=${CC_PREFIX} NDK_ROOT=${NDK_ROOT} "${BUILD_ALL_ANDROID_PATH}" \
-x "${GEN_LIBS_DIR}" \
-s "${TF_ROOT_DIR}/tensorflow/contrib/makefile/sub_makefiles/hexagon_graph_execution/Makefile.in" \
-t hexagon_graph_execution
```

### Push binaries to your Android device

Before running tests on your Android device, you need to push several binaries to it.

```shell
adb push "${GEN_LIBS_DIR}/libhexagon_controller.so" "/data/local/tmp"
adb push "${GEN_LIBS_DIR}/libhexagon_nn_skel.so" "/vendor/lib/rfsa/adsp"
adb push -p \
"${TF_ROOT_DIR}/tensorflow/contrib/makefile/gen/bin/hexagon_graph_execution" \
"/data/local/tmp/"
adb wait-for-device
ANDROID_EXEC_FILE_MODE=755
adb shell chmod "${ANDROID_EXEC_FILE_MODE}" \
"/data/local/tmp/hexagon_graph_execution"
adb wait-for-device
```

### Run tests on the device

Finally, you can run the inference tests on your device.

```shell
adb shell 'LD_LIBRARY_PATH=/data/local/tmp:$LD_LIBRARY_PATH' \
"/data/local/tmp/hexagon_graph_execution"
```

### Troubleshooting

#### Testsig issue

If you're using the Open-Q 820 Snapdragon Development Kit, you may run into an issue with running the executable due to a missing `testsig` library. From the Hexagon SDK documentation: *Dynamic shared objects are required to be digitally signed and then authenticated at runtime before they are allowed to be loaded and executed.* Generating a testsig library is necessary to run the unsigned sample library built from this project.

If the lack of a `testsig` library is your problem, you will see errors of the type:
`vendor/qcom/proprietary/adsprpc/src/fastrpc_apps_user.c:169::error: -1: 0 == (nErr = remotectl_open(name, (int*)ph, dlerrstr, sizeof(dlerrstr), &dlerr))`
appearing in `adb logcat` or ["Expected: (version) >= (1), actual: 0 vs 1" while running a binary from adb](https://github.com/tensorflow/tensorflow/issues/11210).

You need to add a test signature, as described at the beginning of this README. After rebooting your device, you should be able to run the sample application.

#### Qualcomm SDK Linux installation fails with "Malformed \uxxxx encoding"

The installation file is based on LaunchAnywhere, which fails in Linux if the `PS1` env variable contains non-common Unicode chars:

```
Preparing to install...
Extracting the JRE from the installer archive...
Unpacking the JRE...
Extracting the installation resources from the installer archive...
Configuring the installer for this system's environment...

Launching installer...

An internal LaunchAnywhere application error has occurred and this application cannot proceed. (LAX)

Stack Trace:
java.lang.IllegalArgumentException: Malformed \uxxxx encoding.
  at java.util.Properties.loadConvert(Properties.java:574)
  at java.util.Properties.load0(Properties.java:391)
  at java.util.Properties.load(Properties.java:317)
  at com.zerog.common.java.util.PropertiesUtil.loadProperties(Unknown Source)
  at com.zerog.lax.LAX.<init>(Unknown Source)
  at com.zerog.lax.LAX.main(Unknown Source)
```

It can be solved by temporarily assigning the `PS1` environment variable to something simple, such as '$'.

## Maintainers

* Satoshi Kataoka (satok@google.com, github.com/satok16)
