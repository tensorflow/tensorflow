#!/usr/bin/env bash
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ==============================================================================

# Compile the nsync library for the platforms given as arguments.

set -e

prog=compile_nsync.sh
android_api_version=21
default_android_arch=armeabi-v7a
default_ios_arch="i386 x86_64 armv7 armv7s arm64"

usage="usage: $prog [-t linux|ios|android|macos|native]
        [-a architecture] [-v android_api_version]

A script to build nsync for tensorflow.
This script can be run on Linux or MacOS host platforms, and can target 
Linux, MacOS, iOS, or Android.

Options:
-t target_platform
The default target platform is the native host platform.

-a architecture
For Android and iOS target platforms, specify which architecture
to target.
For iOS, the default is: $default_ios_arch.
For Android, the default is: $default_android_arch.

-v android_api_version
Specify the Android API version; the default is $android_api_version."

# Deduce host platform.
host_platform=
nsync_path=
case `uname -s` in
Linux)  host_platform=linux  android_host=linux;;
Darwin) host_platform=macos  android_host=darwin;;
*)      echo "$prog: can't deduce host platform" >&2; exit 2;;
esac
host_arch=`uname -m`
case "$host_arch" in i[345678]86) host_arch=x86_32;; esac

# Parse command line.
target_platform=native   # Default is to build for the host.
target_arch=default
while
        arg="${1-}"
        case "$arg" in
        -*)     case "$arg" in -*t*) target_platform="${2?"$usage"}"; shift; esac
                case "$arg" in -*a*) target_arch="${2?"$usage"}"; shift; esac
                case "$arg" in -*v*) android_api_version="${2?"$usage"}"; shift; esac
                case "$arg" in -*[!atv]*) echo "$usage" >&2; exit 2;; esac;;
        "")     break;;
        *)      echo "$usage" >&2; exit 2;;
        esac
do
        shift
done

# Sanity check the target platform.
case "$target_platform" in
native) target_platform="$host_platform";;
esac

# Change directory to the root of the source tree.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../../.."

nsync_builds_dir=tensorflow/contrib/makefile/downloads/nsync/builds

case "$target_platform" in
ios)            case "$target_arch" in
                default) archs="$default_ios_arch";;
                *)       archs="$target_arch";;
                esac
                ;;
android)        case "$target_arch" in
                default) archs="$default_android_arch";;
                *)       archs="$target_arch";;
                esac
                ;;
*)              archs="$target_arch";;
esac

# For ios, the library names for the CPU types accumulate in $platform_libs
platform_libs=

# Compile nsync.
for arch in $archs; do
        nsync_platform_dir="$nsync_builds_dir/$arch.$target_platform.c++11"

        # Get Makefile for target.
        case "$target_platform" in
        linux)  makefile='
                        CC=${CC_PREFIX} g++
                        PLATFORM_CPPFLAGS=-DNSYNC_USE_CPP11_TIMEPOINT -DNSYNC_ATOMIC_CPP11 \
                                          -I../../platform/c++11 -I../../platform/gcc \
                                          -I../../platform/posix -pthread
                        PLATFORM_CFLAGS=-std=c++11 -Werror -Wall -Wextra -pedantic
                        PLATFORM_LDFLAGS=-pthread
                        MKDEP=${CC} -M -std=c++11
                        PLATFORM_C=../../platform/c++11/src/nsync_semaphore_mutex.cc \
                                   ../../platform/c++11/src/per_thread_waiter.cc \
                                   ../../platform/c++11/src/yield.cc \
                                   ../../platform/c++11/src/time_rep_timespec.cc \
                                   ../../platform/c++11/src/nsync_panic.cc
                        PLATFORM_OBJS=nsync_semaphore_mutex.o per_thread_waiter.o yield.o \
                                      time_rep_timespec.o nsync_panic.o
                        TEST_PLATFORM_C=../../platform/c++11/src/start_thread.cc
                        TEST_PLATFORM_OBJS=start_thread.o
                        include ../../platform/posix/make.common
                        include dependfile
                ';;

        ios)
                arch_flags=
                case "$arch" in
                i386|x86_64)
                        arch_flags="$arch_flags -mios-simulator-version-min=8.0"
                        arch_flags="$arch_flags -isysroot `xcrun --sdk iphonesimulator --show-sdk-path`"
                        ;;
                *)
                        arch_flags="$arch_flags -miphoneos-version-min=8.0"
                        arch_flags="$arch_flags -isysroot `xcrun --sdk iphoneos --show-sdk-path`"
                        ;;
                esac
                makefile='
                        CC=${CC_PREFIX} clang++
                        PLATFORM_CPPFLAGS=-DNSYNC_USE_CPP11_TIMEPOINT -DNSYNC_ATOMIC_CPP11 \
                                          -I../../platform/c++11 -I../../platform/gcc_no_tls \
                                          -I../../platform/macos -I../../platform/posix -pthread
                        PLATFORM_CFLAGS=-arch '"$arch"' -fno-exceptions -stdlib=libc++ \
                                        -fembed-bitcode '"$arch_flags"' -fPIC -x c++ \
                                        -std=c++11 -Werror -Wall -Wextra -pedantic
                        PLATFORM_LDFLAGS=-pthread
                        MKDEP=${CC} -x c++ -M -std=c++11
                        PLATFORM_C=../../platform/posix/src/clock_gettime.c \
                                   ../../platform/c++11/src/nsync_semaphore_mutex.cc \
                                   ../../platform/posix/src/per_thread_waiter.c \
                                   ../../platform/c++11/src/yield.cc \
                                   ../../platform/c++11/src/time_rep_timespec.cc \
                                   ../../platform/c++11/src/nsync_panic.cc
                        PLATFORM_OBJS=clock_gettime.o nsync_semaphore_mutex.o per_thread_waiter.o \
                                      yield.o time_rep_timespec.o nsync_panic.o
                        TEST_PLATFORM_C=../../platform/c++11/src/start_thread.cc
                        TEST_PLATFORM_OBJS=start_thread.o
                        include ../../platform/posix/make.common
                        include dependfile
                ';;

        macos)  makefile='
                        CC=${CC_PREFIX} clang++
                        PLATFORM_CPPFLAGS=-DNSYNC_USE_CPP11_TIMEPOINT -DNSYNC_ATOMIC_CPP11 \
                                          -I../../platform/c++11 -I../../platform/gcc \
                                          -I../../platform/macos -I../../platform/posix -pthread
                        PLATFORM_CFLAGS=-x c++ -std=c++11 -Werror -Wall -Wextra -pedantic
                        PLATFORM_LDFLAGS=-pthread
                        MKDEP=${CC} -x c++ -M -std=c++11
                        PLATFORM_C=../../platform/posix/src/clock_gettime.c \
                                   ../../platform/c++11/src/nsync_semaphore_mutex.cc \
                                   ../../platform/posix/src/per_thread_waiter.c \
                                   ../../platform/c++11/src/yield.cc \
                                   ../../platform/c++11/src/time_rep_timespec.cc \
                                   ../../platform/c++11/src/nsync_panic.cc
                        PLATFORM_OBJS=clock_gettime.o nsync_semaphore_mutex.o per_thread_waiter.o \
                                      yield.o time_rep_timespec.o nsync_panic.o
                        TEST_PLATFORM_C=../../platform/c++11/src/start_thread.cc
                        TEST_PLATFORM_OBJS=start_thread.o
                        include ../../platform/posix/make.common
                        include dependfile
                ';;

        android)
                # The Android build uses many different names for the same
                # platform in different parts of the tree, so things get messy here.

                # Make $android_os_arch be the OS-arch name for the host
                # binaries used in the NDK tree.
                case "$host_platform" in
                linux)  android_os_arch=linux;;
                macos)  android_os_arch=darwin;;
                *)      android_os_arch="$host_platform";;
                esac
                case "$host_arch" in
                x86_32) android_os_arch="$android_os_arch"-x86;;
                *)      android_os_arch="$android_os_arch-$host_arch";;
                esac

                case "$arch" in
                arm64-v8a)              toolchain="aarch64-linux-android-4.9"
                                        sysroot_arch="arm64"
                                        bin_prefix="aarch64-linux-android"
                                        march_option=
                                        ;;
                armeabi)                toolchain="arm-linux-androideabi-4.9"
                                        sysroot_arch="arm"
                                        bin_prefix="arm-linux-androideabi"
                                        march_option=
                                        ;;
                armeabi-v7a)            toolchain="arm-linux-androideabi-4.9"
                                        sysroot_arch="arm"
                                        bin_prefix="arm-linux-androideabi"
                                        march_option="-march=armv7-a -mfloat-abi=softfp -mfpu=neon"
                                        ;;
                armeabi-v7a-hard)       toolchain="arm-linux-androideabi-4.9"
                                        sysroot_arch="arm"
                                        bin_prefix="arm-linux-androideabi"
                                        march_option="-march=armv7-a -mfpu=neon"
                                        ;;
                mips)                   toolchain="mipsel-linux-android-4.9"
                                        sysroot_arch="mips"
                                        bin_prefix="mipsel-linux-android"
                                        march_option=
                                        ;;
                mips64)                 toolchain="mips64el-linux-android-4.9"
                                        sysroot_arch="mips64"
                                        bin_prefix="mips64el-linux-android"
                                        march_option=
                                        ;;
                x86)                    toolchain="x86-4.9"
                                        sysroot_arch="x86"
                                        bin_prefix="i686-linux-android"
                                        march_option=
                                        ;;
                x86_64)                 toolchain="x86_64-4.9"
                                        sysroot_arch="x86_64"
                                        bin_prefix="x86_64-linux-android"
                                        march_option=
                                        ;;
                *)                      echo "android is not supported for $arch" >&2
                                        echo "$usage" >&2
                                        exit 2
                                        ;;
                esac


                android_target_platform=armeabi
                case "$NDK_ROOT" in
                "")     echo "$prog: requires \$NDK_ROOT for android build" >&2
                        exit 2;;
                esac

                makefile='
                        CC=${CC_PREFIX} \
                           ${NDK_ROOT}/toolchains/'"$toolchain"'/prebuilt/'"$android_os_arch"'/bin/'"$bin_prefix"'-g++
                        PLATFORM_CPPFLAGS=--sysroot \
                                          $(NDK_ROOT)/platforms/android-'"$android_api_version"'/arch-'"$sysroot_arch"' \
                                          -DNSYNC_USE_CPP11_TIMEPOINT -DNSYNC_ATOMIC_CPP11 \
                                          -I$(NDK_ROOT)/sources/android/support/include \
                                          -I$(NDK_ROOT)/sources/cxx-stl/gnu-libstdc++/4.9/include \
                                          -I$(NDK_ROOT)/sources/cxx-stl/gnu-libstdc++/4.9/libs/'"$arch"'/include \
                                          -I../../platform/c++11 -I../../platform/gcc \
                                          -I../../platform/posix -pthread
                        PLATFORM_CFLAGS=-std=c++11 -Wno-narrowing '"$march_option"' -fPIE
                        PLATFORM_LDFLAGS=-pthread
                        MKDEP=${CC} -M -std=c++11
                        PLATFORM_C=../../platform/c++11/src/nsync_semaphore_mutex.cc \
                                   ../../platform/c++11/src/per_thread_waiter.cc \
                                   ../../platform/c++11/src/yield.cc \
                                   ../../platform/c++11/src/time_rep_timespec.cc \
                                   ../../platform/c++11/src/nsync_panic.cc
                        PLATFORM_OBJS=nsync_semaphore_mutex.o per_thread_waiter.o yield.o \
                                      time_rep_timespec.o nsync_panic.o
                        TEST_PLATFORM_C=../../platform/c++11/src/start_thread.cc
                        TEST_PLATFORM_OBJS=start_thread.o
                        include ../../platform/posix/make.common
                        include dependfile
                ';;

                *)      echo "$usage" >&2; exit 2;;
        esac

        if [ ! -d "$nsync_platform_dir" ]; then
                mkdir "$nsync_platform_dir"
                echo "$makefile" | sed 's,^[ \t]*,,' > "$nsync_platform_dir/Makefile"
                touch "$nsync_platform_dir/dependfile"
        fi
        if (cd "$nsync_platform_dir" && make depend nsync.a >&2); then
                case "$target_platform" in
                ios)    platform_libs="$platform_libs '$nsync_platform_dir/nsync.a'";;
                *)      echo "$nsync_platform_dir/nsync.a";;
                esac
        else
                exit 2  # The if-statement suppresses the "set -e" on the "make".
        fi
done

case "$target_platform" in
ios)    nsync_platform_dir="$nsync_builds_dir/lipo.$target_platform.c++11"
        mkdir "$nsync_platform_dir"
        eval lipo $platform_libs -create -output '$nsync_platform_dir/nsync.a'
        echo "$nsync_platform_dir/nsync.a"
        ;;
esac
