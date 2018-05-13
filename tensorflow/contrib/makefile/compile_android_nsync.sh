#!/bin/bash -e

toolchain_path=""

prog=compile_nsync_android.sh

usage() {
    echo "Usage: $(basename "$0") [t]"
    echo "-t Absolute path to a toolchain"
    exit 1
}

while getopts "t:" opt_name; do
    case "$opt_name" in
        t) toolchain_path="${OPTARG}";;
        *) usage;;
    esac
done
shift $((OPTIND - 1))

if [[ -z "${toolchain_path}" ]]
then
    echo "You need to specify toolchain path. Use -t"
    exit 1
fi

# Change directory to the root of the source tree.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../../.."

nsync_builds_dir=tensorflow/contrib/makefile/downloads/nsync/builds

nsync_platform_dir="$nsync_builds_dir/android_distribution"

makefile='
CC='"$toolchain_path"'/bin/clang++
PLATFORM_CPPFLAGS=--sysroot \
'"$toolchain_path"'/sysroot \
-DNSYNC_USE_CPP11_TIMEPOINT -DNSYNC_ATOMIC_CPP11 \
-I../../platform/c++11 -I../../platform/gcc \
-I../../platform/posix -pthread
PLATFORM_CFLAGS=-std=c++11 -Wno-narrowing -fPIE -fPIC
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
'

if [ ! -d "$nsync_platform_dir" ]; then
    mkdir "$nsync_platform_dir"
    echo "$makefile" | sed $'s,^[ \t]*,,' > "$nsync_platform_dir/Makefile"
    touch "$nsync_platform_dir/dependfile"
fi

if (cd "$nsync_platform_dir" && make depend nsync.a >&2); then
    echo "$nsync_platform_dir/nsync.a"
else
    exit 2
fi

