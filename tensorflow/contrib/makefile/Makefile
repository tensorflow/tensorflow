# This Makefile compiles a library containing the C++ runtime for the TensorFlow
# library. It's designed for use on platforms with limited resources where
# running a full Bazel build would be prohibitive, or for cross-compilation onto
# embedded systems. It includes only a bare-bones set of functionality.
#
# The default setup below is aimed at Unix-like devices, and should work on
# modern Linux and OS X distributions without changes.
#
# If you have another platform, you'll need to take a careful look at the
# compiler flags and folders defined below. They're separated into two sections,
# the first for the host (the machine you're compiling on) and the second for
# the target (the machine you want the program to run on).

SHELL := /bin/bash

# Host compilation settings

# Find where we're running from, so we can store generated files here.
ifeq ($(origin MAKEFILE_DIR), undefined)
	MAKEFILE_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
endif

HAS_GEN_HOST_PROTOC := \
$(shell test -f $(MAKEFILE_DIR)/gen/protobuf-host/bin/protoc && echo "true" ||\
echo "false")

# Hexagon integration
ifdef HEXAGON_LIBS
	LIBGEMM_WRAPPER := $(HEXAGON_LIBS)/libhexagon_controller.so
	ifeq ($(shell test -f $(LIBGEMM_WRAPPER) 2> /dev/null; echo $$?), 0)
    $(info "Use hexagon libs at " $(LIBGEMM_WRAPPER))
	else
    $(error "hexagon libs not found at " $(LIBGEMM_WRAPPER))
	endif
	ifdef HEXAGON_INCLUDE
		ifeq ($(shell test -d $(HEXAGON_INCLUDE) 2> /dev/null; echo $$?), 0)
      $(info "Use hexagon libs at " $(HEXAGON_INCLUDE))
		else
      $(error "hexagon libs not found at " $(HEXAGON_INCLUDE))
		endif
	else
    $(error "HEXAGON_INCLUDE is not set.")
	endif
	ifneq ($(TARGET),ANDROID)
    $(error "hexagon is only supported on Android")
	endif
endif # HEXAGON_LIBS

# If ANDROID_TYPES is not set assume __ANDROID_TYPES_SLIM__
ifeq ($(ANDROID_TYPES),)
	ANDROID_TYPES := -D__ANDROID_TYPES_SLIM__
endif

# Try to figure out the host system
HOST_OS :=
ifeq ($(OS),Windows_NT)
	HOST_OS = WINDOWS
else
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Linux)
	        HOST_OS := LINUX
	endif
	ifeq ($(UNAME_S),Darwin)
		HOST_OS := OSX
	endif
endif

HOST_ARCH := $(shell if [[ $(shell uname -m) =~ i[345678]86 ]]; then echo x86_32; else echo $(shell uname -m); fi)

# Where compiled objects are stored.
HOST_OBJDIR := $(MAKEFILE_DIR)/gen/host_obj/
HOST_BINDIR := $(MAKEFILE_DIR)/gen/host_bin/
HOST_GENDIR := $(MAKEFILE_DIR)/gen/host_obj/

# Settings for the host compiler.
HOST_CXX := $(CC_PREFIX) gcc
HOST_CXXFLAGS := --std=c++11
HOST_LDOPTS :=
ifeq ($(HAS_GEN_HOST_PROTOC),true)
	HOST_LDOPTS += -L$(MAKEFILE_DIR)/gen/protobuf-host/lib
endif
HOST_LDOPTS += -L/usr/local/lib

HOST_INCLUDES := \
-I. \
-I$(MAKEFILE_DIR)/../../../ \
-I$(MAKEFILE_DIR)/downloads/ \
-I$(MAKEFILE_DIR)/downloads/eigen \
-I$(MAKEFILE_DIR)/downloads/gemmlowp \
-I$(MAKEFILE_DIR)/downloads/nsync/public \
-I$(MAKEFILE_DIR)/downloads/fft2d \
-I$(MAKEFILE_DIR)/downloads/double_conversion \
-I$(HOST_GENDIR)
ifeq ($(HAS_GEN_HOST_PROTOC),true)
	HOST_INCLUDES += -I$(MAKEFILE_DIR)/gen/protobuf-host/include
endif
# This is at the end so any globally-installed frameworks like protobuf don't
# override local versions in the source tree.
HOST_INCLUDES += -I/usr/local/include

HOST_LIBS := \
$(HOST_NSYNC_LIB) \
-lstdc++ \
-lprotobuf \
-lpthread \
-lm \
-lz

# If we're on Linux, also link in the dl library.
ifeq ($(HOST_OS),LINUX)
	HOST_LIBS += -ldl -lpthread
endif

# If we're on a Pi, link in pthreads and dl
ifeq ($(HOST_OS),PI)
	HOST_LIBS += -ldl -lpthread
endif


# proto_text is a tool that converts protobufs into a form we can use more
# compactly within TensorFlow. It's a bit like protoc, but is designed to
# produce a much more minimal result so we can save binary space.
# We have to build it on the host system first so that we can create files
# that are needed for the runtime building.
PROTO_TEXT := $(HOST_BINDIR)proto_text
# The list of dependencies is derived from the Bazel build file by running
# the gen_file_lists.sh script on a system with a working Bazel setup.
PROTO_TEXT_CC_FILES := $(shell cat $(MAKEFILE_DIR)/proto_text_cc_files.txt)
PROTO_TEXT_PB_CC_LIST := \
	$(shell cat $(MAKEFILE_DIR)/proto_text_pb_cc_files.txt) \
	$(wildcard tensorflow/contrib/makefile/downloads/double_conversion/double-conversion/*.cc)
PROTO_TEXT_PB_H_LIST := $(shell cat $(MAKEFILE_DIR)/proto_text_pb_h_files.txt)

# Locations of the intermediate files proto_text generates.
PROTO_TEXT_PB_H_FILES := $(addprefix $(HOST_GENDIR), $(PROTO_TEXT_PB_H_LIST))
PROTO_TEXT_CC_OBJS := $(addprefix $(HOST_OBJDIR), $(PROTO_TEXT_CC_FILES:.cc=.o))
PROTO_TEXT_PB_OBJS := $(addprefix $(HOST_OBJDIR), $(PROTO_TEXT_PB_CC_LIST:.cc=.o))
PROTO_TEXT_OBJS := $(PROTO_TEXT_CC_OBJS) $(PROTO_TEXT_PB_OBJS)

# Target device settings.

# Default to running on the same system we're compiling on.
# You should override TARGET on the command line if you're cross-compiling, e.g.
# make -f tensorflow/contrib/makefile/Makefile TARGET=ANDROID
TARGET := $(HOST_OS)

# Where compiled objects are stored.
GENDIR := $(MAKEFILE_DIR)/gen/
OBJDIR := $(GENDIR)obj/
LIBDIR := $(GENDIR)lib/
BINDIR := $(GENDIR)bin/
PBTGENDIR := $(GENDIR)proto_text/
PROTOGENDIR := $(GENDIR)proto/
DEPDIR := $(GENDIR)dep/
$(shell mkdir -p $(DEPDIR) >/dev/null)

# Settings for the target compiler.
CXX := $(CC_PREFIX) gcc
OPTFLAGS := -O2

ifneq ($(TARGET),ANDROID)
  OPTFLAGS += -march=native
endif

CXXFLAGS := --std=c++11 -DIS_SLIM_BUILD -fno-exceptions -DNDEBUG $(OPTFLAGS)
LDFLAGS := \
-L/usr/local/lib
DEPFLAGS = -MT $@ -MMD -MP -MF $(DEPDIR)/$*.Td

INCLUDES := \
-I. \
-I$(MAKEFILE_DIR)/downloads/ \
-I$(MAKEFILE_DIR)/downloads/eigen \
-I$(MAKEFILE_DIR)/downloads/gemmlowp \
-I$(MAKEFILE_DIR)/downloads/nsync/public \
-I$(MAKEFILE_DIR)/downloads/fft2d \
-I$(MAKEFILE_DIR)/downloads/double_conversion \
-I$(PROTOGENDIR) \
-I$(PBTGENDIR)
ifeq ($(HAS_GEN_HOST_PROTOC),true)
	INCLUDES += -I$(MAKEFILE_DIR)/gen/protobuf-host/include
endif
# This is at the end so any globally-installed frameworks like protobuf don't
# override local versions in the source tree.
INCLUDES += -I/usr/local/include

LIBS := \
$(TARGET_NSYNC_LIB) \
-lstdc++ \
-lprotobuf \
-lz \
-lm

ifeq ($(HAS_GEN_HOST_PROTOC),true)
	PROTOC := $(MAKEFILE_DIR)/gen/protobuf-host/bin/protoc
else
	PROTOC := protoc
endif

$(info PROTOC = "$(PROTOC)")
$(info CC_PREFIX = "$(CC_PREFIX)")

PROTOCFLAGS :=
AR := ar
ARFLAGS := -r
LIBFLAGS :=

# If we're on OS X, make sure that globals aren't stripped out.
ifeq ($(TARGET),OSX)
ifeq ($(HAS_GEN_HOST_PROTOC),true)
	LIBFLAGS += -L$(MAKEFILE_DIR)/gen/protobuf-host/lib
	export LD_LIBRARY_PATH=$(MAKEFILE_DIR)/gen/protobuf-host/lib
endif
	LDFLAGS += -all_load
endif
# Make sure that we don't strip global constructors on Linux.
ifeq ($(TARGET),LINUX)
ifeq ($(HAS_GEN_HOST_PROTOC),true)
	LIBFLAGS += -L$(MAKEFILE_DIR)/gen/protobuf-host/lib
	export LD_LIBRARY_PATH=$(MAKEFILE_DIR)/gen/protobuf-host/lib
endif
	CXXFLAGS += -fPIC
	LIBFLAGS += -Wl,--allow-multiple-definition -Wl,--whole-archive
	LDFLAGS := -Wl,--no-whole-archive
endif
# If we're on Linux, also link in the dl library.
ifeq ($(TARGET),LINUX)
	LIBS += -ldl -lpthread
endif
# If we're cross-compiling for the Raspberry Pi, use the right gcc.
ifeq ($(TARGET),PI)
	CXXFLAGS += $(ANDROID_TYPES) -DRASPBERRY_PI
	LDFLAGS := -Wl,--no-whole-archive
	LIBS += -ldl -lpthread
	LIBFLAGS += -Wl,--allow-multiple-definition -Wl,--whole-archive
endif

# Set up Android building
# LINT.IfChange
ifeq ($(TARGET),ANDROID)
# Override NDK_ROOT on the command line with your own NDK location, e.g.
# make -f tensorflow/contrib/makefile/Makefile TARGET=ANDROID \
# NDK_ROOT=/path/to/your/ndk
# You need to have an Android version of the protobuf libraries compiled to link
# in. The compile_android_protobuf.sh script may help.

	ANDROID_HOST_OS_ARCH :=
	ifeq ($(HOST_OS),LINUX)
		ANDROID_HOST_OS_ARCH=linux
	endif
	ifeq ($(HOST_OS),OSX)
		ANDROID_HOST_OS_ARCH=darwin
	endif
	ifeq ($(HOST_OS),WINDOWS)
    $(error "windows is not supported.")
	endif

	ifeq ($(HOST_ARCH),x86_32)
		ANDROID_HOST_OS_ARCH := $(ANDROID_HOST_OS_ARCH)-x86
	else
		ANDROID_HOST_OS_ARCH := $(ANDROID_HOST_OS_ARCH)-$(HOST_ARCH)
	endif
    
	ifndef ANDROID_ARCH
		ANDROID_ARCH := armeabi-v7a
	endif

	ifeq ($(ANDROID_ARCH),arm64-v8a)
		TOOLCHAIN := aarch64-linux-android-4.9
		SYSROOT_ARCH := arm64
		BIN_PREFIX := aarch64-linux-android
		MARCH_OPTION :=
	endif
	ifeq ($(ANDROID_ARCH),armeabi)
		TOOLCHAIN := arm-linux-androideabi-4.9
		SYSROOT_ARCH := arm
		BIN_PREFIX := arm-linux-androideabi
		MARCH_OPTION :=
	endif
	ifeq ($(ANDROID_ARCH),armeabi-v7a)
		TOOLCHAIN := arm-linux-androideabi-4.9
		SYSROOT_ARCH := arm
		BIN_PREFIX := arm-linux-androideabi
		MARCH_OPTION := -march=armv7-a -mfloat-abi=softfp -mfpu=neon
	endif
	ifeq ($(ANDROID_ARCH),mips)
		TOOLCHAIN := mipsel-linux-android-4.9
		SYSROOT_ARCH := mips
		BIN_PREFIX := mipsel-linux-android
		MARCH_OPTION :=
	endif
	ifeq ($(ANDROID_ARCH),mips64)
		TOOLCHAIN := mips64el-linux-android-4.9
		SYSROOT_ARCH := mips64
		BIN_PREFIX := mips64el-linux-android
		MARCH_OPTION :=
	endif
	ifeq ($(ANDROID_ARCH),x86)
		TOOLCHAIN := x86-4.9
		SYSROOT_ARCH := x86
		BIN_PREFIX := i686-linux-android
		MARCH_OPTION :=
	endif
	ifeq ($(ANDROID_ARCH),x86_64)
		TOOLCHAIN := x86_64-4.9
		SYSROOT_ARCH := x86_64
		BIN_PREFIX := x86_64-linux-android
		MARCH_OPTION :=
	endif
    
	ifndef NDK_ROOT
    $(error "NDK_ROOT is not defined.")
	endif
	CXX := $(CC_PREFIX) $(NDK_ROOT)/toolchains/$(TOOLCHAIN)/prebuilt/$(ANDROID_HOST_OS_ARCH)/bin/$(BIN_PREFIX)-g++
	CC := $(CC_PREFIX) $(NDK_ROOT)/toolchains/$(TOOLCHAIN)/prebuilt/$(ANDROID_HOST_OS_ARCH)/bin/$(BIN_PREFIX)-gcc
	CXXFLAGS +=\
--sysroot $(NDK_ROOT)/platforms/android-21/arch-$(SYSROOT_ARCH) \
-Wno-narrowing \
-fomit-frame-pointer \
$(MARCH_OPTION) \
-fPIE \
-fPIC
	INCLUDES = \
-I$(NDK_ROOT)/sources/android/support/include \
-I$(NDK_ROOT)/sources/cxx-stl/gnu-libstdc++/4.9/include \
-I$(NDK_ROOT)/sources/cxx-stl/gnu-libstdc++/4.9/libs/$(ANDROID_ARCH)/include \
-I. \
-I$(MAKEFILE_DIR)/downloads/ \
-I$(MAKEFILE_DIR)/downloads/eigen \
-I$(MAKEFILE_DIR)/downloads/gemmlowp \
-I$(MAKEFILE_DIR)/downloads/nsync/public \
-I$(MAKEFILE_DIR)/downloads/fft2d \
-I$(MAKEFILE_DIR)/downloads/double_conversion \
-I$(MAKEFILE_DIR)/gen/protobuf_android/$(ANDROID_ARCH)/include \
-I$(PROTOGENDIR) \
-I$(PBTGENDIR)

	LIBS := \
$(TARGET_NSYNC_LIB) \
-lgnustl_static \
-lprotobuf \
-llog \
-lz \
-lm \
-ldl \
-latomic

	LD := $(NDK_ROOT)/toolchains/$(TOOLCHAIN)/prebuilt/$(ANDROID_HOST_OS_ARCH)/$(BIN_PREFIX)/bin/ld

	LDFLAGS := \
$(MARCH_OPTION) \
-L$(MAKEFILE_DIR)/gen/protobuf_android/$(ANDROID_ARCH)/lib \
-L$(NDK_ROOT)/sources/cxx-stl/gnu-libstdc++/4.9/libs/$(ANDROID_ARCH) \
-fPIE \
-pie \
-v

	AR := $(NDK_ROOT)/toolchains/$(TOOLCHAIN)/prebuilt/$(ANDROID_HOST_OS_ARCH)/bin/$(BIN_PREFIX)-ar
	ARFLAGS := r
	LIBFLAGS += -Wl,--allow-multiple-definition -Wl,--whole-archive

	ifdef HEXAGON_LIBS
		INCLUDES += -I$(HEXAGON_INCLUDE)
		LIBS += -lhexagon_controller
		LDFLAGS += -L$(HEXAGON_LIBS)
		CXXFLAGS += -DUSE_HEXAGON_LIBS

# CAVEAT: We should disable TENSORFLOW_DISABLE_META while running
# quantized_matmul on Android because it crashes in
# MultiThreadGemm in tensorflow/core/kernels/meta_support.cc
# See http://b/33270149
# TODO(satok): Remove once it's fixed
		CXXFLAGS += -DTENSORFLOW_DISABLE_META

# Declare __ANDROID_TYPES_FULL__ to enable required types for hvx
		CXXFLAGS += -D__ANDROID_TYPES_FULL__
	endif

	ifdef ENABLE_EXPERIMENTAL_HEXNN_OPS
		CXXFLAGS += -DENABLE_EXPERIMENTAL_HEXNN_OPS
	endif

	ifeq ($(BUILD_FOR_TEGRA),1)
		NVCC := $(JETPACK)/cuda/bin/nvcc
		NVCCFLAGS := -x=cu -D__CUDACC__ -DNVCC -DANDROID_TEGRA -ccbin $(NDK_ROOT)/toolchains/$(TOOLCHAIN)/prebuilt/$(ANDROID_HOST_OS_ARCH)/bin/$(BIN_PREFIX)-g++ --std c++11 --expt-relaxed-constexpr -m64 -gencode arch=compute_53,\"code=sm_53\" -gencode arch=compute_62,\"code=sm_62\" -DEIGEN_AVOID_STL_ARRAY -DTENSORFLOW_USE_EIGEN_THREADPOOL -DLANG_CXX11 -DEIGEN_HAS_C99_MATH -DGOOGLE_CUDA=1 -DTF_EXTRA_CUDA_CAPABILITIES=5.3
		CXXFLAGS4NVCC =\
-DIS_SLIM_BUILD \
-DANDROID_TEGRA \
-fno-exceptions \
-DNDEBUG $(OPTFLAGS) \
-march=armv8-a \
-fPIE \
-D__ANDROID_TYPES_FULL__ \
--sysroot $(NDK_ROOT)/platforms/android-21/arch-arm64

		CXXFLAGS +=\
-DGOOGLE_CUDA=1 \
-D__ANDROID_TYPES_FULL__ \
-DANDROID_TEGRA \
-DEIGEN_AVOID_STL_ARRAY \
-DEIGEN_HAS_C99_MATH \
-DLANG_CXX11 -DTENSORFLOW_USE_EIGEN_THREADPOOL -DTF_EXTRA_CUDA_CAPABILITIES=5.3

		INCLUDES += \
-Itensorflow/core/kernels \
-I$(MAKEFILE_DIR)/downloads/cub \
-I$(MAKEFILE_DIR)/downloads/cub/cub_archive/cub/device \
-Ithird_party/toolchains/gpus/cuda \
-I$(JETPACK)/cuda/include \
-I$(JETPACK) \
-I$(JETPACK)/cuDNN/aarch64 \
-I$(JETPACK)/cuda/extras/CUPTI/include


		CUDA_LIBS := \
-ltfcuda \
-lcudart_static \
-lcudnn \
-lcublas_static \
-lcufftw_static \
-lcusolver_static \
-lcusparse_static \
-lcufft \
-lcuda \
-lculibos \
-lcurand_static

		OBJDIR := $(OBJDIR)android_arm64-v8a/
		LIBDIR := $(LIBDIR)android_arm64-v8a/
		BINDIR := $(BINDIR)android_arm64-v8a/
		DEPDIR := $(DEPDIR)android_arm64-v8a/

		TEGRA_LIBS := \
-L$(JETPACK)/cuda/targets/aarch64-linux-androideabi/lib \
-L$(JETPACK)/cuda/targets/aarch64-linux-androideabi/lib/stubs \
-L$(JETPACK)/cuda/targets/aarch64-linux-androideabi/lib64 \
-L$(JETPACK)/cuda/targets/aarch64-linux-androideabi/lib64/stubs \
-L$(JETPACK)/cuDNN/aarch64/cuda/lib64 \
-L$(LIBDIR)

		CUDA_LIB_DEPS := $(LIBDIR)libtfcuda.a
	else
		OBJDIR := $(OBJDIR)android_$(ANDROID_ARCH)/
		LIBDIR := $(LIBDIR)android_$(ANDROID_ARCH)/
		BINDIR := $(BINDIR)android_$(ANDROID_ARCH)/
		DEPDIR := $(DEPDIR)android_$(ANDROID_ARCH)/
	endif # ifeq ($(BUILD_FOR_TEGRA),1)
endif  # ANDROID
# LINT.ThenChange(//tensorflow/contrib/android/cmake/CMakeLists.txt)

# Settings for iOS.
ifeq ($(TARGET),IOS)
	IPHONEOS_PLATFORM := $(shell xcrun --sdk iphoneos --show-sdk-platform-path)
	IPHONEOS_SYSROOT := $(shell xcrun --sdk iphoneos --show-sdk-path)
	IPHONESIMULATOR_PLATFORM := $(shell xcrun --sdk iphonesimulator \
	--show-sdk-platform-path)
	IPHONESIMULATOR_SYSROOT := $(shell xcrun --sdk iphonesimulator \
	--show-sdk-path)
	IOS_SDK_VERSION := $(shell xcrun --sdk iphoneos --show-sdk-version)
	MIN_SDK_VERSION := 9.0
# Override IOS_ARCH with ARMV7, ARMV7S, ARM64, or I386.
	IOS_ARCH := X86_64
	ifeq ($(IOS_ARCH),ARMV7)
		CXXFLAGS += -miphoneos-version-min=$(MIN_SDK_VERSION) \
		-arch armv7 \
		-fembed-bitcode \
		-D__thread=thread_local \
		-DUSE_GEMM_FOR_CONV \
		-Wno-c++11-narrowing \
		-mno-thumb \
		-DTF_LEAN_BINARY \
		$(ANDROID_TYPES) \
		-fno-exceptions \
		-isysroot \
		${IPHONEOS_SYSROOT}
		LDFLAGS := -arch armv7 \
		-fembed-bitcode \
		-miphoneos-version-min=${MIN_SDK_VERSION} \
		-framework Accelerate \
		-Xlinker -S \
		-Xlinker -x \
		-Xlinker -dead_strip \
		-all_load \
		-L$(GENDIR)protobuf_ios/lib \
		-lz
	endif
	ifeq ($(IOS_ARCH),ARMV7S)
		CXXFLAGS += -miphoneos-version-min=$(MIN_SDK_VERSION) \
		-arch armv7s \
		-fembed-bitcode \
		-D__thread=thread_local \
		-DUSE_GEMM_FOR_CONV \
		-Wno-c++11-narrowing \
		-mno-thumb \
		-DTF_LEAN_BINARY \
		$(ANDROID_TYPES) \
		-fno-exceptions \
		-isysroot \
		${IPHONEOS_SYSROOT}
		LDFLAGS := -arch armv7s \
		-fembed-bitcode \
		-miphoneos-version-min=${MIN_SDK_VERSION} \
		-framework Accelerate \
		-Xlinker -S \
		-Xlinker -x \
		-Xlinker -dead_strip \
		-all_load \
		-L$(GENDIR)protobuf_ios/lib \
		-lz
	endif
	ifeq ($(IOS_ARCH),ARM64)
		CXXFLAGS += -miphoneos-version-min=$(MIN_SDK_VERSION) \
		-arch arm64 \
		-fembed-bitcode \
		-D__thread=thread_local \
		-DUSE_GEMM_FOR_CONV \
		-Wno-c++11-narrowing \
		-DTF_LEAN_BINARY \
		$(ANDROID_TYPES) \
		-fno-exceptions \
		-isysroot \
		${IPHONEOS_SYSROOT}
		LDFLAGS := -arch arm64 \
		-fembed-bitcode \
		-miphoneos-version-min=${MIN_SDK_VERSION} \
		-framework Accelerate \
		-Xlinker -S \
		-Xlinker -x \
		-Xlinker -dead_strip \
		-all_load \
		-L$(GENDIR)protobuf_ios/lib \
		-lz
	endif
	ifeq ($(IOS_ARCH),I386)
		CXXFLAGS += -mios-simulator-version-min=$(MIN_SDK_VERSION) \
		-arch i386 \
		-mno-sse \
		-fembed-bitcode \
		-D__thread=thread_local \
		-DUSE_GEMM_FOR_CONV \
		-Wno-c++11-narrowing \
		-DTF_LEAN_BINARY \
		$(ANDROID_TYPES) \
		-fno-exceptions \
		-isysroot \
		${IPHONESIMULATOR_SYSROOT}
		LDFLAGS := -arch i386 \
		-fembed-bitcode \
		-mios-simulator-version-min=${MIN_SDK_VERSION} \
		-framework Accelerate \
		-Xlinker -S \
		-Xlinker -x \
		-Xlinker -dead_strip \
		-all_load \
		-L$(GENDIR)protobuf_ios/lib \
		-lz
	endif
	ifeq ($(IOS_ARCH),X86_64)
		CXXFLAGS += -mios-simulator-version-min=$(MIN_SDK_VERSION) \
		-arch x86_64 \
		-fembed-bitcode \
		-D__thread=thread_local \
		-DUSE_GEMM_FOR_CONV \
		-Wno-c++11-narrowing \
		-DTF_LEAN_BINARY \
		$(ANDROID_TYPES) \
		-fno-exceptions \
		-isysroot \
		${IPHONESIMULATOR_SYSROOT}
		LDFLAGS := -arch x86_64 \
		-fembed-bitcode \
		-mios-simulator-version-min=${MIN_SDK_VERSION} \
		-framework Accelerate \
		-Xlinker -S \
		-Xlinker -x \
		-Xlinker -dead_strip \
		-all_load \
		-L$(GENDIR)protobuf_ios/lib \
		-lz
	endif
	OBJDIR := $(OBJDIR)ios_$(IOS_ARCH)/
	LIBDIR := $(LIBDIR)ios_$(IOS_ARCH)/
	BINDIR := $(BINDIR)ios_$(IOS_ARCH)/
	DEPDIR := $(DEPDIR)ios_$(IOS_ARCH)/
endif

# This library is the main target for this makefile. It will contain a minimal
# runtime that can be linked in to other programs.
LIB_NAME := libtensorflow-core.a
LIB_PATH := $(LIBDIR)$(LIB_NAME)

# A small example program that shows how to link against the library.
BENCHMARK_NAME := $(BINDIR)benchmark

# What sources we want to compile, derived from the main Bazel build using the
# gen_file_lists.sh script.

CORE_CC_ALL_SRCS := \
$(wildcard tensorflow/core/*.cc) \
$(wildcard tensorflow/core/common_runtime/*.cc) \
$(wildcard tensorflow/core/framework/*.cc) \
$(wildcard tensorflow/core/graph/*.cc) \
$(wildcard tensorflow/core/grappler/*.cc) \
$(wildcard tensorflow/core/grappler/*/*.cc) \
$(wildcard tensorflow/core/lib/*/*.cc) \
$(wildcard tensorflow/core/platform/*.cc) \
$(wildcard tensorflow/core/platform/*/*.cc) \
$(wildcard tensorflow/core/platform/*/*/*.cc) \
$(wildcard tensorflow/core/util/*.cc) \
$(wildcard tensorflow/core/util/*/*.cc) \
$(wildcard tensorflow/contrib/makefile/downloads/double_conversion/double-conversion/*.cc) \
tensorflow/core/util/version_info.cc
# Remove duplicates (for version_info.cc)
CORE_CC_ALL_SRCS := $(sort $(CORE_CC_ALL_SRCS))

CORE_CC_EXCLUDE_SRCS_NON_GPU := \
$(wildcard tensorflow/core/*/*test.cc) \
$(wildcard tensorflow/core/*/*testutil*) \
$(wildcard tensorflow/core/*/*testlib*) \
$(wildcard tensorflow/core/*/*main.cc) \
$(wildcard tensorflow/core/*/*/*test.cc) \
$(wildcard tensorflow/core/*/*/*testutil*) \
$(wildcard tensorflow/core/*/*/*testlib*) \
$(wildcard tensorflow/core/*/*/*main.cc) \
$(wildcard tensorflow/core/debug/*.cc) \
$(wildcard tensorflow/core/framework/op_gen_lib.cc) \
$(wildcard tensorflow/core/graph/dot.*) \
$(wildcard tensorflow/core/lib/db/*) \
$(wildcard tensorflow/core/lib/gif/*) \
$(wildcard tensorflow/core/lib/io/zlib*) \
$(wildcard tensorflow/core/lib/io/record*) \
$(wildcard tensorflow/core/lib/jpeg/*) \
$(wildcard tensorflow/core/lib/png/*) \
$(wildcard tensorflow/core/util/events_writer.*) \
$(wildcard tensorflow/core/util/reporter.*) \
$(wildcard tensorflow/core/platform/default/test_benchmark.*) \
$(wildcard tensorflow/core/platform/cloud/*) \
$(wildcard tensorflow/core/platform/google/*) \
$(wildcard tensorflow/core/platform/google/*/*) \
$(wildcard tensorflow/core/platform/jpeg.*) \
$(wildcard tensorflow/core/platform/png.*) \
$(wildcard tensorflow/core/platform/s3/*) \
$(wildcard tensorflow/core/platform/windows/*) \
$(wildcard tensorflow/core/grappler/inputs/trivial_test_graph_input_yielder.*) \
$(wildcard tensorflow/core/grappler/inputs/file_input_yielder.*) \
$(wildcard tensorflow/core/grappler/clusters/single_machine.*) \
tensorflow/core/util/cuda_kernel_helper_test.cu.cc

CORE_CC_EXCLUDE_SRCS := \
$(CORE_CC_EXCLUDE_SRCS_NON_GPU) \
$(wildcard tensorflow/core/platform/stream_executor.*) \
$(wildcard tensorflow/core/platform/default/cuda_libdevice_path.*) \
$(wildcard tensorflow/core/platform/cuda.h) \
$(wildcard tensorflow/core/platform/cuda_libdevice_path.*) \
$(wildcard tensorflow/core/user_ops/*.cu.cc) \
$(wildcard tensorflow/core/common_runtime/gpu/*) \
$(wildcard tensorflow/core/common_runtime/gpu_device_factory.*)

ifeq ($(BUILD_FOR_TEGRA),1)
CORE_CC_ALL_SRCS := $(CORE_CC_ALL_SRCS) \
tensorflow/core/kernels/concat_lib_gpu.cc \
tensorflow/core/kernels/cuda_solvers.cc \
tensorflow/core/kernels/cudnn_pooling_gpu.cc \
tensorflow/core/kernels/dense_update_functor.cc \
tensorflow/core/kernels/fractional_avg_pool_op.cc \
tensorflow/core/kernels/fractional_max_pool_op.cc \
tensorflow/core/kernels/fractional_pool_common.cc \
tensorflow/core/kernels/pooling_ops_3d.cc \
tensorflow/core/kernels/sparse_fill_empty_rows_op.cc \
tensorflow/core/kernels/list_kernels.cc \
$(wildcard tensorflow/core/common_runtime/gpu/*.cc) \
$(wildcard tensorflow/stream_executor/*.cc) \
$(wildcard tensorflow/stream_executor/*/*.cc)

CORE_CC_EXCLUDE_SRCS := \
$(CORE_CC_EXCLUDE_SRCS_NON_GPU)

CUDA_CC_SRCS := $(wildcard tensorflow/core/kernels/*.cu.cc)
CUDA_CC_OBJS := $(addprefix $(OBJDIR), $(CUDA_CC_SRCS:.cc=.o))
endif  # TEGRA

# Filter out all the excluded files.
TF_CC_SRCS := $(filter-out $(CORE_CC_EXCLUDE_SRCS), $(CORE_CC_ALL_SRCS))
# Add in any extra files that don't fit the patterns easily
TF_CC_SRCS += tensorflow/contrib/makefile/downloads/fft2d/fftsg.c
TF_CC_SRCS += tensorflow/core/common_runtime/gpu/gpu_id_manager.cc
# Also include the op and kernel definitions.
TF_CC_SRCS += $(shell cat $(MAKEFILE_DIR)/tf_op_files.txt)
PBT_CC_SRCS := $(shell cat $(MAKEFILE_DIR)/tf_pb_text_files.txt)
PROTO_SRCS := $(shell cat $(MAKEFILE_DIR)/tf_proto_files.txt)
BENCHMARK_SRCS := \
tensorflow/core/util/reporter.cc \
tensorflow/tools/benchmark/benchmark_model.cc \
tensorflow/tools/benchmark/benchmark_model_main.cc

ifdef HEXAGON_LIBS
	TF_CC_SRCS += \
tensorflow/cc/framework/scope.cc \
tensorflow/cc/framework/ops.cc \
tensorflow/cc/ops/const_op.cc \
tensorflow/core/kernels/hexagon/graph_transfer_utils.cc \
tensorflow/core/kernels/hexagon/graph_transferer.cc \
tensorflow/core/kernels/hexagon/hexagon_control_wrapper.cc \
tensorflow/core/kernels/hexagon/hexagon_ops_definitions.cc \
tensorflow/core/kernels/hexagon/hexagon_remote_fused_graph_executor_build.cc
endif

# File names of the intermediate files target compilation generates.
TF_CC_OBJS := $(addprefix $(OBJDIR), \
$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(TF_CC_SRCS))))
PBT_GEN_FILES := $(addprefix $(PBTGENDIR), $(PBT_CC_SRCS))
PBT_OBJS := $(addprefix $(OBJDIR), $(PBT_CC_SRCS:.cc=.o))
PROTO_CC_SRCS := $(addprefix $(PROTOGENDIR), $(PROTO_SRCS:.proto=.pb.cc))
PROTO_OBJS := $(addprefix $(OBJDIR), $(PROTO_SRCS:.proto=.pb.o))
LIB_OBJS := $(PROTO_OBJS) $(TF_CC_OBJS) $(PBT_OBJS)
BENCHMARK_OBJS := $(addprefix $(OBJDIR), $(BENCHMARK_SRCS:.cc=.o))

.PHONY: clean cleantarget

# The target that's compiled if there's no command-line arguments.
all: $(LIB_PATH) $(BENCHMARK_NAME)

# Rules for target compilation.


.phony_version_info:
tensorflow/core/util/version_info.cc: .phony_version_info
	tensorflow/tools/git/gen_git_source.sh $@

# Gathers together all the objects we've compiled into a single '.a' archive.
$(LIB_PATH): $(LIB_OBJS)
	@mkdir -p $(dir $@)
	$(AR) $(ARFLAGS) $(LIB_PATH) $(LIB_OBJS)

$(BENCHMARK_NAME): $(BENCHMARK_OBJS) $(LIB_PATH) $(CUDA_LIB_DEPS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
	-o $(BENCHMARK_NAME) $(BENCHMARK_OBJS) \
	$(LIBFLAGS) $(TEGRA_LIBS) $(LIB_PATH) $(LDFLAGS) $(LIBS) $(CUDA_LIBS)

# NVCC compilation rules for Tegra
ifeq ($(BUILD_FOR_TEGRA),1)
$(OBJDIR)%.cu.o: %.cu.cc
	@mkdir -p $(dir $@)
	@mkdir -p $(dir $(DEPDIR)$*)
	$(NVCC) $(NVCCFLAGS) -Xcompiler "$(CXXFLAGS4NVCC) $(DEPFLAGS)" $(INCLUDES) -c $< -o $@

$(LIBDIR)libtfcuda.a: $(CUDA_CC_OBJS)
	@mkdir -p $(dir $@)
	$(AR) $(ARFLAGS) $@ $(CUDA_CC_OBJS)
endif

# Matches on the normal hand-written TensorFlow C++ source files.
$(OBJDIR)%.o: %.cc | $(PBT_GEN_FILES)
	@mkdir -p $(dir $@)
	@mkdir -p $(dir $(DEPDIR)$*)
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) $(INCLUDES) -c $< -o $@
	@mv -f $(DEPDIR)/$*.Td $(DEPDIR)/$*.d

# Matches on plain C files.
$(OBJDIR)%.o: %.c
	@mkdir -p $(dir $@)
	@mkdir -p $(dir $(DEPDIR)$*)
	$(CXX) $(patsubst --std=c++11,--std=c99, $(CXXFLAGS)) -x c $(DEPFLAGS) \
$(INCLUDES) -c $< -o $@
	@mv -f $(DEPDIR)/$*.Td $(DEPDIR)/$*.d

# Compiles C++ source files that have been generated by protoc.
$(OBJDIR)%.pb.o: $(PROTOGENDIR)%.pb.cc
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Builds C++ code from proto files using protoc.
$(PROTOGENDIR)%.pb.cc $(PROTOGENDIR)%.pb.h: %.proto
	@mkdir -p $(dir $@)
	$(PROTOC) $(PROTOCFLAGS) $< --cpp_out $(PROTOGENDIR)

# Uses proto_text to generate minimal pb_text C++ files from protos.
$(PBTGENDIR)%.pb_text.cc $(PBTGENDIR)%.pb_text.h $(PBTGENDIR)%.pb_text-impl.h: %.proto | $(PROTO_TEXT)
	@mkdir -p $(dir $@)
	$(PROTO_TEXT) \
	$(PBTGENDIR)tensorflow/core \
	tensorflow/core/ \
	tensorflow/tools/proto_text/placeholder.txt \
	$<

# Compiles the C++ source files created by proto_text.
$(OBJDIR)%.pb_text.o: $(PBTGENDIR)%.pb_text.cc
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Makes sure that we don't compile the protoc-generated C++ sources before they
# and the proto_text files have been created.
$(PROTO_OBJS): $(PROTO_CC_SRCS) $(PBT_GEN_FILES)

# Host compilation rules.

# For normal manually-created TensorFlow C++ source files.
$(HOST_OBJDIR)%.o: %.cc
	@mkdir -p $(dir $@)
	$(HOST_CXX) $(HOST_CXXFLAGS) $(HOST_INCLUDES) -c $< -o $@

# Compiles object code from protoc-built C++ source files.
$(HOST_OBJDIR)%.pb.o: $(HOST_GENDIR)%.pb.cc
	@mkdir -p $(dir $@)
	$(HOST_CXX) $(HOST_CXXFLAGS) $(HOST_INCLUDES) -c $< -o $@

# Ensures we wait until proto_text has generated the .h files from protos before
# we compile the C++.
$(PROTO_TEXT_OBJS) : $(PROTO_TEXT_PB_H_FILES)

# Runs proto_text to generate C++ source files from protos.
$(PROTO_TEXT): $(PROTO_TEXT_OBJS) $(PROTO_TEXT_PB_H_FILES)
	@mkdir -p $(dir $@)
	$(HOST_CXX) $(HOST_CXXFLAGS) $(HOST_INCLUDES) \
	-o $(PROTO_TEXT) $(PROTO_TEXT_OBJS) $(HOST_LDOPTS) $(HOST_LIBS)

# Compiles the C++ source files from protos using protoc.
$(HOST_GENDIR)%.pb.cc $(HOST_GENDIR)%.pb.h: %.proto
	@mkdir -p $(dir $@)
	$(PROTOC) $(PROTOCFLAGS) $< --cpp_out $(HOST_GENDIR)

# Gets rid of all generated files.
clean:
	rm -rf $(MAKEFILE_DIR)/gen
	rm -rf tensorflow/core/util/version_info.cc

# Gets rid of all generated files except protobuf libs generated
# before calling make.  This allows users not to recompile proto libs everytime.
clean_except_protobuf_libs:
	find $(MAKEFILE_DIR)/gen -mindepth 1 -maxdepth 1 ! -name "protobuf*" -exec rm -r "{}" \;
	rm -rf tensorflow/core/util/version_info.cc

# Gets rid of target files only, leaving the host alone. Also leaves the lib
# directory untouched deliberately, so we can persist multiple architectures
# across builds for iOS and Android.
cleantarget:
	rm -rf $(OBJDIR)
	rm -rf $(BINDIR)
	rm -rf $(LIBDIR)

$(DEPDIR)/%.d: ;
.PRECIOUS: $(DEPDIR)/%.d

-include $(patsubst %,$(DEPDIR)/%.d,$(basename $(TF_CC_SRCS)))

ifdef SUB_MAKEFILES
  $(warning "include sub makefiles, must not contain white spaces in the path:" $(SUB_MAKEFILES))
  include $(SUB_MAKEFILES)
endif
