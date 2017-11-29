#!/bin/bash -x
set -e
make -f tensorflow/contrib/lite/Makefile TARGET=IOS IOS_ARCH=x86_64 -j 8
make -f tensorflow/contrib/lite/Makefile TARGET=IOS IOS_ARCH=i386 -j 8
make -f tensorflow/contrib/lite/Makefile TARGET=IOS IOS_ARCH=armv7 -j 8
make -f tensorflow/contrib/lite/Makefile TARGET=IOS IOS_ARCH=armv7s -j 8
make -f tensorflow/contrib/lite/Makefile TARGET=IOS IOS_ARCH=arm64 -j 8

lipo \
tensorflow/contrib/lite/gen/lib/ios_x86_64/libtensorflow-lite.a \
tensorflow/contrib/lite/gen/lib/ios_i386/libtensorflow-lite.a \
tensorflow/contrib/lite/gen/lib/ios_armv7/libtensorflow-lite.a \
tensorflow/contrib/lite/gen/lib/ios_armv7s/libtensorflow-lite.a \
tensorflow/contrib/lite/gen/lib/ios_arm64/libtensorflow-lite.a \
-create \
-output tensorflow/contrib/lite/gen/lib/libtensorflow-lite.a
