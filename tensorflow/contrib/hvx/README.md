# TensorFlow Runtime with HVX Acceleration

## Description

This README explain how to build and use the TensorFlow Runtime with HVX Acceleration. HVX is an extension of Hexagon which is a DSP provided by qualcomm which can compute vector calculations faster using lower energy than ARM processors.

## Quick Start Guide

We provides several tools to build and run inference with this runtime quickly.

#### All-in-one script to run inception model with prebuild hexagon library
If you don’t need to build your own implementation of hexagon HVX, we provide a shortcut to execute graphs by using pre-compiled binaries.

```
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
NDK_ROOT="/path/to/ndk" ./tensorflow/contrib/makefile/build_all_android.sh -X
```
(-X downloads dependencies to hexagon HVX and graphs, and copy all dependencies to android and execute a test)

#### All-in-one script to run inception model by building entire libraries from source code
 If you want to build your own implementation of hexagon HVX, we provide a sample all-in-one script to execute graphs which downloads source and build everything for hexagon.

```
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
QUALCOMM_SDK="/path/to/qualcomm/sdk" NDK_ROOT="/path/to/ndk" ./tensorflow/contrib/makefile/samples/build_and_run_inception_hexagon.sh
```

## Building libraries

If you've finished walking through the quick start guide, you may want to try building each binary manually.

#### Build libhexagon_nn_skel.so
Download hexagon nn library from codeaurora.org and build it.

```
git clone https://source.codeaurora.org/quic/hexagon_nn/nnlib
cd nnlib
```

(Just follow instructions in README.HOW_TO_BUILD. You can find libhexagon_nn_skel.so in hexagon_Release_dynamic_toolv72_v60/ship)
Then copy the generated binary to GEN_LIBS_DIR

```
GEN_LIBS_DIR="/path/to/a/dir/to/store/hexagon/libraries"
cp -v "hexagon_Release_dynamic_toolv72_v60/ship/libhexagon_nn_skel.so" "${GEN_LIBS_DIR}"
```

#### Build libhexagon_controller.so
Download tensorflow and build hexagon controller.

```
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
TF_ROOT_DIR="$(pwd)"
QUALCOMM_SDK="/path/to/qualcomm/sdk"
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

#### Build tensorflow linking hexagon library
Build tensorflow with the build_all_android.sh with specifying -x option.

```
BUILD_ALL_ANDROID_PATH="${TF_ROOT_DIR}/tensorflow/contrib/makefile/build_all_android.sh"
NDK_ROOT="/path/to/ndk/root"

CC_PREFIX=${CC_PREFIX} NDK_ROOT=${NDK_ROOT} "${BUILD_ALL_ANDROID_PATH}" \
-x "${GEN_LIBS_DIR}" \
-s "${TF_ROOT_DIR}/tensorflow/contrib/makefile/sub_makefiles/hexagon_graph_execution/Makefile.in" \
-t hexagon_graph_execution
```

#### Push binaries to your Android device

Before running tests on your Android device, you need to push several binaries to it.

```
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

#### Run tests on the device

Finally, you can run the inference tests on your device.

```
adb shell 'LD_LIBRARY_PATH=/data/local/tmp:$LD_LIBRARY_PATH' \
"/data/local/tmp/hexagon_graph_execution"
```
