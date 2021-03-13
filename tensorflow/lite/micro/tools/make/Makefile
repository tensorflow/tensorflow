# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

ifneq (3.82,$(firstword $(sort $(MAKE_VERSION) 3.82)))
  $(error "Requires make version 3.82 or later (current is $(MAKE_VERSION))")
endif

# root directory of tensorflow
TENSORFLOW_ROOT :=
MAKEFILE_DIR := tensorflow/lite/micro/tools/make

#  Override this on make command line to parse third party downloads during project generation
#  make -f tensorflow/lite/micro/tools/make/Makefile PARSE_THIRD_PARTY=true TARGET=apollo3evb generate_hello_world_make_project
PARSE_THIRD_PARTY :=


# Pull in some convenience functions.
include $(MAKEFILE_DIR)/helper_functions.inc

# Try to figure out the host system
HOST_OS :=
ifeq ($(OS),Windows_NT)
	HOST_OS = windows
else
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Linux)
		HOST_OS := linux
	endif
	ifeq ($(UNAME_S),Darwin)
		HOST_OS := osx
	endif
endif

# Determine the host architecture, with any ix86 architecture being labelled x86_32
HOST_ARCH := $(shell if uname -m | grep -Eq 'i[345678]86'; then echo x86_32; else echo $(shell uname -m); fi)

# Override these on the make command line to target a specific architecture. For example:
# make -f tensorflow/lite/Makefile TARGET=rpi TARGET_ARCH=armv7l
TARGET := $(HOST_OS)
TARGET_ARCH := $(HOST_ARCH)

# Default compiler and tool names:
TOOLCHAIN:=gcc
CXX_TOOL := g++
CC_TOOL := gcc
AR_TOOL := ar

ifneq ($(TAGS),)
  $(error The TAGS command line option is no longer supported in the TFLM Makefile.)
endif

# Specify which specialized kernel implementation should be pulled in.
OPTIMIZED_KERNEL_DIR :=

# Specify which co-processor's kernel implementation should be pulled in.
# If the same kernel is implemented in both kernels/OPTIMIZED_KERNEL_DIR and
# kernels/CO_PROCESSOR, then the implementation from kernels/CO_PROCESSOR will
# be used.
CO_PROCESSOR :=

# This is obviously horrible.  We need to generate these 3 versions of the
# include directories from one source.
INCLUDES := \
-I. \
-I$(MAKEFILE_DIR)/downloads/gemmlowp \
-I$(MAKEFILE_DIR)/downloads/flatbuffers/include \
-I$(MAKEFILE_DIR)/downloads/ruy

# Same list of paths, but now relative to the generated project files.
GENERATED_PROJECT_INCLUDES := \
-I. \
-I./third_party/gemmlowp \
-I./third_party/flatbuffers/include \
-I./third_party/ruy

# Same list of paths, but now in the format the generate_keil_project.py
# script expects them.
PROJECT_INCLUDES := \
. \
third_party/gemmlowp \
third_party/flatbuffers/include \
third_party/ruy

TEST_SCRIPT :=

MICROLITE_LIBS := -lm

# For the target, optimized_kernel_dir, and co-processor as specified on the
# command line we add -D<tag> to the cflags to allow for #idefs in the code.
#
# We apply the following transformations (via the tr command):
#   1. Convert to uppercase (TARGET=xtensa -> -DXTENSA)

ADDITIONAL_DEFINES := -D$(shell echo $(TARGET) | tr [a-z] [A-Z])

ifneq ($(OPTIMIZED_KERNEL_DIR),)
  ADDITIONAL_DEFINES += -D$(shell echo $(OPTIMIZED_KERNEL_DIR) | tr [a-z] [A-Z])
endif

ifneq ($(CO_PROCESSOR),)
  ADDITIONAL_DEFINES += -D$(shell echo $(CO_PROCESSOR) | tr [a-z] [A-Z])
endif

OPTIMIZATION_LEVEL := -O3

CC_WARNINGS := \
  -Werror \
  -Wsign-compare \
  -Wdouble-promotion \
  -Wshadow \
  -Wunused-variable \
  -Wmissing-field-initializers \
  -Wunused-function \
  -Wswitch \
  -Wvla \
  -Wall \
  -Wextra \
  -Wstrict-aliasing \
  -Wno-unused-parameter

COMMON_FLAGS := \
  -fno-unwind-tables \
  -ffunction-sections \
  -fdata-sections \
  -fmessage-length=0 \
  -DTF_LITE_STATIC_MEMORY \
  -DTF_LITE_DISABLE_X86_NEON \
  $(OPTIMIZATION_LEVEL) \
  $(CC_WARNINGS) \
  $(ADDITIONAL_DEFINES)

ifeq ($(TARGET), $(HOST_OS))
  # If we are not doing a cross-compilation then -DTF_LITE_USE_CTIME is what we
  # want to have by default.
  COMMON_FLAGS += -DTF_LITE_USE_CTIME
endif

CXXFLAGS := \
  -std=c++11 \
  -fno-rtti \
  -fno-exceptions \
  -fno-threadsafe-statics \
  $(COMMON_FLAGS)

CCFLAGS := \
  -std=c11 \
  $(COMMON_FLAGS)

ARFLAGS := -r

ifeq ($(TOOLCHAIN), gcc)
  ifneq ($(TARGET), osx)
    # GCC on MacOS uses an LLVM backend so we avoid the additional linker flags
    # that are unsupported with LLVM.
    LDFLAGS += \
      -Wl,--fatal-warnings \
      -Wl,--gc-sections
  endif
endif

# override these in the makefile.inc for specific compiler targets
TARGET_TOOLCHAIN_PREFIX :=
TARGET_TOOLCHAIN_ROOT :=

# Specifying BUILD_TYPE=<blah> as part of the make command gives us a few
# options to choose from.
#
# If BUILD_TYPE is not specified, the default build (which should be suitable
# most of the time) has all of the error checking logic at the expense of a
# latency increase of ~5-10% relative to BUILD_TYPE=release_with_logs.
#
# This default build is most suited for usual development and testing as is
# highlighted by the discussion on this github pull request:
# https://github.com/tensorflow/tensorflow/pull/42314#issuecomment-694360567
BUILD_TYPE := default
ifeq ($(BUILD_TYPE), debug)
	# Specifying BUILD_TYPE=debug adds debug symbols to the binary (and makes it
	# larger) and should be used to run a binary with gdb.
	CXXFLAGS += -g
	CCFLAGS  += -g
else ifeq ($(BUILD_TYPE), release)
	# The 'release' build results in the smallest binary (by virtue of removing
	# strings from log messages, DCHECKs ...).
	#
	# The down-side is that we currently do not have a good mechanism to allow
	# for logging that is not related to errors (e.g. profiling information, or
	# logs that help determine if tests pass or fail). As a result, we are unable
	# to run tests or benchmarks with BUILD_TYPE=release (which is a bit
	# counter-intuitive). TODO(b/158205789): A global error reporter might help.
	#
	# For a close approximation of the release build use
	# BUILD_TYPE=release_with_logs.
	CXXFLAGS += -DNDEBUG -DTF_LITE_STRIP_ERROR_STRINGS
	CCFLAGS  += -DNDEBUG -DTF_LITE_STRIP_ERROR_STRINGS
else ifeq ($(BUILD_TYPE), release_with_logs)
	# The latency with BUILD_TYPE=release_with_logs will be close to the 'release'
	# build and there will still be error logs. This build type may be preferable
	# for profiling and benchmarking.
	CXXFLAGS += -DNDEBUG
	CCFLAGS  += -DNDEBUG
else ifeq ($(BUILD_TYPE), no_tf_lite_static_memory)
	# This build should not be used to run any binaries/tests since
	# TF_LITE_STATIC_MEMORY should be defined for all micro builds. However,
	# having a build without TF_LITE_STATIC_MEMORY is useful to catch errors in
	# code that is shared between TfLite Mobile and TfLite Micro. See this issue
	# for more details:
	# https://github.com/tensorflow/tensorflow/issues/43076
	CXXFLAGS := $(filter-out -DTF_LITE_STATIC_MEMORY, $(CXXFLAGS))
	CCFLAGS := $(filter-out -DTF_LITE_STATIC_MEMORY, $(CCFLAGS))
endif

# This library is the main target for this makefile. It will contain a minimal
# runtime that can be linked in to other programs.
MICROLITE_LIB_NAME := libtensorflow-microlite.a

# These two must be defined before we include the target specific Makefile.inc
# because we filter out the examples that are not supported for those targets.
# See targets/xtensa_xpg_makefile.inc for an example.
#
# We limit max depth of directories to search to not include target specific
# Makefiles that are included directly by the main example Makefile. See
# examples/micro_speech/Makefile.inc for an example. At the same time, we
# search till an arbitrary depth for files named Makefile_internal.inc as a way
# to bypass this check and allow for deeper directory structures.
MICRO_LITE_EXAMPLE_TESTS := $(shell find tensorflow/lite/micro/examples/ -maxdepth 2 -name Makefile.inc)
MICRO_LITE_EXAMPLE_TESTS += $(shell find tensorflow/lite/micro/examples/ -name Makefile_internal.inc)
MICRO_LITE_BENCHMARKS := $(wildcard tensorflow/lite/micro/benchmarks/Makefile.inc)

# TODO(b/152645559): move all benchmarks to benchmarks directory.
MICROLITE_BENCHMARK_SRCS := \
$(wildcard tensorflow/lite/micro/benchmarks/*benchmark.cc)

MICROLITE_TEST_SRCS := \
tensorflow/lite/micro/memory_arena_threshold_test.cc \
tensorflow/lite/micro/memory_helpers_test.cc \
tensorflow/lite/micro/micro_allocator_test.cc \
tensorflow/lite/micro/micro_error_reporter_test.cc \
tensorflow/lite/micro/micro_interpreter_test.cc \
tensorflow/lite/micro/micro_mutable_op_resolver_test.cc \
tensorflow/lite/micro/micro_string_test.cc \
tensorflow/lite/micro/micro_time_test.cc \
tensorflow/lite/micro/micro_utils_test.cc \
tensorflow/lite/micro/recording_micro_allocator_test.cc \
tensorflow/lite/micro/recording_simple_memory_allocator_test.cc \
tensorflow/lite/micro/simple_memory_allocator_test.cc \
tensorflow/lite/micro/testing_helpers_test.cc \
tensorflow/lite/micro/kernels/activations_test.cc \
tensorflow/lite/micro/kernels/add_test.cc \
tensorflow/lite/micro/kernels/add_n_test.cc \
tensorflow/lite/micro/kernels/arg_min_max_test.cc \
tensorflow/lite/micro/kernels/batch_to_space_nd_test.cc \
tensorflow/lite/micro/kernels/cast_test.cc \
tensorflow/lite/micro/kernels/ceil_test.cc \
tensorflow/lite/micro/kernels/circular_buffer_test.cc \
tensorflow/lite/micro/kernels/comparisons_test.cc \
tensorflow/lite/micro/kernels/concatenation_test.cc \
tensorflow/lite/micro/kernels/conv_test.cc \
tensorflow/lite/micro/kernels/depthwise_conv_test.cc \
tensorflow/lite/micro/kernels/dequantize_test.cc \
tensorflow/lite/micro/kernels/detection_postprocess_test.cc \
tensorflow/lite/micro/kernels/elementwise_test.cc \
tensorflow/lite/micro/kernels/elu_test.cc \
tensorflow/lite/micro/kernels/exp_test.cc \
tensorflow/lite/micro/kernels/expand_dims_test.cc \
tensorflow/lite/micro/kernels/fill_test.cc \
tensorflow/lite/micro/kernels/floor_test.cc \
tensorflow/lite/micro/kernels/fully_connected_test.cc \
tensorflow/lite/micro/kernels/hard_swish_test.cc \
tensorflow/lite/micro/kernels/l2norm_test.cc \
tensorflow/lite/micro/kernels/leaky_relu_test.cc \
tensorflow/lite/micro/kernels/logical_test.cc \
tensorflow/lite/micro/kernels/logistic_test.cc \
tensorflow/lite/micro/kernels/maximum_minimum_test.cc \
tensorflow/lite/micro/kernels/mul_test.cc \
tensorflow/lite/micro/kernels/neg_test.cc \
tensorflow/lite/micro/kernels/pack_test.cc \
tensorflow/lite/micro/kernels/pad_test.cc \
tensorflow/lite/micro/kernels/pooling_test.cc \
tensorflow/lite/micro/kernels/prelu_test.cc \
tensorflow/lite/micro/kernels/quantization_util_test.cc \
tensorflow/lite/micro/kernels/quantize_test.cc \
tensorflow/lite/micro/kernels/reduce_test.cc \
tensorflow/lite/micro/kernels/reshape_test.cc \
tensorflow/lite/micro/kernels/resize_nearest_neighbor_test.cc \
tensorflow/lite/micro/kernels/round_test.cc \
tensorflow/lite/micro/kernels/shape_test.cc \
tensorflow/lite/micro/kernels/softmax_test.cc \
tensorflow/lite/micro/kernels/space_to_batch_nd_test.cc \
tensorflow/lite/micro/kernels/split_test.cc \
tensorflow/lite/micro/kernels/split_v_test.cc \
tensorflow/lite/micro/kernels/squeeze_test.cc \
tensorflow/lite/micro/kernels/strided_slice_test.cc \
tensorflow/lite/micro/kernels/sub_test.cc \
tensorflow/lite/micro/kernels/svdf_test.cc \
tensorflow/lite/micro/kernels/tanh_test.cc \
tensorflow/lite/micro/kernels/transpose_conv_test.cc \
tensorflow/lite/micro/kernels/unpack_test.cc \
tensorflow/lite/micro/kernels/zeros_like_test.cc \
tensorflow/lite/micro/memory_planner/greedy_memory_planner_test.cc \
tensorflow/lite/micro/memory_planner/linear_memory_planner_test.cc

MICROLITE_CC_KERNEL_SRCS := \
tensorflow/lite/micro/kernels/activations.cc \
tensorflow/lite/micro/kernels/add.cc \
tensorflow/lite/micro/kernels/add_n.cc \
tensorflow/lite/micro/kernels/arg_min_max.cc \
tensorflow/lite/micro/kernels/batch_to_space_nd.cc \
tensorflow/lite/micro/kernels/cast.cc \
tensorflow/lite/micro/kernels/ceil.cc \
tensorflow/lite/micro/kernels/circular_buffer.cc \
tensorflow/lite/micro/kernels/comparisons.cc \
tensorflow/lite/micro/kernels/concatenation.cc \
tensorflow/lite/micro/kernels/conv.cc \
tensorflow/lite/micro/kernels/conv_common.cc \
tensorflow/lite/micro/kernels/depthwise_conv.cc \
tensorflow/lite/micro/kernels/depthwise_conv_common.cc \
tensorflow/lite/micro/kernels/dequantize.cc \
tensorflow/lite/micro/kernels/detection_postprocess.cc \
tensorflow/lite/micro/kernels/elementwise.cc \
tensorflow/lite/micro/kernels/elu.cc \
tensorflow/lite/micro/kernels/ethosu.cc \
tensorflow/lite/micro/kernels/exp.cc \
tensorflow/lite/micro/kernels/expand_dims.cc \
tensorflow/lite/micro/kernels/fill.cc \
tensorflow/lite/micro/kernels/floor.cc \
tensorflow/lite/micro/kernels/fully_connected.cc \
tensorflow/lite/micro/kernels/fully_connected_common.cc \
tensorflow/lite/micro/kernels/hard_swish.cc \
tensorflow/lite/micro/kernels/kernel_runner.cc \
tensorflow/lite/micro/kernels/kernel_util.cc \
tensorflow/lite/micro/kernels/l2norm.cc \
tensorflow/lite/micro/kernels/leaky_relu.cc \
tensorflow/lite/micro/kernels/logical.cc \
tensorflow/lite/micro/kernels/logistic.cc \
tensorflow/lite/micro/kernels/maximum_minimum.cc \
tensorflow/lite/micro/kernels/mul.cc \
tensorflow/lite/micro/kernels/neg.cc \
tensorflow/lite/micro/kernels/pack.cc \
tensorflow/lite/micro/kernels/pad.cc \
tensorflow/lite/micro/kernels/pooling.cc \
tensorflow/lite/micro/kernels/prelu.cc \
tensorflow/lite/micro/kernels/quantize.cc \
tensorflow/lite/micro/kernels/quantize_common.cc \
tensorflow/lite/micro/kernels/reduce.cc \
tensorflow/lite/micro/kernels/reshape.cc \
tensorflow/lite/micro/kernels/resize_nearest_neighbor.cc \
tensorflow/lite/micro/kernels/round.cc \
tensorflow/lite/micro/kernels/shape.cc \
tensorflow/lite/micro/kernels/softmax.cc \
tensorflow/lite/micro/kernels/softmax_common.cc \
tensorflow/lite/micro/kernels/space_to_batch_nd.cc \
tensorflow/lite/micro/kernels/split.cc \
tensorflow/lite/micro/kernels/split_v.cc \
tensorflow/lite/micro/kernels/squeeze.cc \
tensorflow/lite/micro/kernels/strided_slice.cc \
tensorflow/lite/micro/kernels/sub.cc \
tensorflow/lite/micro/kernels/svdf.cc \
tensorflow/lite/micro/kernels/svdf_common.cc \
tensorflow/lite/micro/kernels/tanh.cc \
tensorflow/lite/micro/kernels/transpose_conv.cc \
tensorflow/lite/micro/kernels/unpack.cc \
tensorflow/lite/micro/kernels/zeros_like.cc

MICROLITE_TEST_HDRS := \
$(wildcard tensorflow/lite/micro/testing/*.h)

MICROLITE_CC_BASE_SRCS := \
$(wildcard tensorflow/lite/micro/*.cc) \
$(wildcard tensorflow/lite/micro/memory_planner/*.cc) \
tensorflow/lite/c/common.c \
tensorflow/lite/core/api/error_reporter.cc \
tensorflow/lite/core/api/flatbuffer_conversions.cc \
tensorflow/lite/core/api/op_resolver.cc \
tensorflow/lite/core/api/tensor_utils.cc \
tensorflow/lite/kernels/internal/quantization_util.cc \
tensorflow/lite/kernels/kernel_util.cc \
tensorflow/lite/schema/schema_utils.cc

MICROLITE_CC_SRCS := $(filter-out $(MICROLITE_TEST_SRCS), $(MICROLITE_CC_BASE_SRCS))
MICROLITE_CC_SRCS := $(filter-out $(MICROLITE_BENCHMARK_SRCS), $(MICROLITE_CC_SRCS))

MICROLITE_CC_HDRS := \
$(wildcard tensorflow/lite/micro/*.h) \
$(wildcard tensorflow/lite/micro/benchmarks/*model_data.h) \
$(wildcard tensorflow/lite/micro/kernels/*.h) \
$(wildcard tensorflow/lite/micro/memory_planner/*.h) \
LICENSE \
tensorflow/lite/c/builtin_op_data.h \
tensorflow/lite/c/c_api_types.h \
tensorflow/lite/c/common.h \
tensorflow/lite/core/api/error_reporter.h \
tensorflow/lite/core/api/flatbuffer_conversions.h \
tensorflow/lite/core/api/op_resolver.h \
tensorflow/lite/core/api/tensor_utils.h \
tensorflow/lite/kernels/internal/common.h \
tensorflow/lite/kernels/internal/compatibility.h \
tensorflow/lite/kernels/internal/optimized/neon_check.h \
tensorflow/lite/kernels/internal/quantization_util.h \
tensorflow/lite/kernels/internal/reference/add.h \
tensorflow/lite/kernels/internal/reference/add_n.h \
tensorflow/lite/kernels/internal/reference/arg_min_max.h \
tensorflow/lite/kernels/internal/reference/batch_to_space_nd.h \
tensorflow/lite/kernels/internal/reference/binary_function.h \
tensorflow/lite/kernels/internal/reference/ceil.h \
tensorflow/lite/kernels/internal/reference/comparisons.h \
tensorflow/lite/kernels/internal/reference/concatenation.h \
tensorflow/lite/kernels/internal/reference/conv.h \
tensorflow/lite/kernels/internal/reference/depthwiseconv_float.h \
tensorflow/lite/kernels/internal/reference/depthwiseconv_uint8.h \
tensorflow/lite/kernels/internal/reference/dequantize.h \
tensorflow/lite/kernels/internal/reference/elu.h \
tensorflow/lite/kernels/internal/reference/exp.h \
tensorflow/lite/kernels/internal/reference/fill.h \
tensorflow/lite/kernels/internal/reference/floor.h \
tensorflow/lite/kernels/internal/reference/fully_connected.h \
tensorflow/lite/kernels/internal/reference/hard_swish.h \
tensorflow/lite/kernels/internal/reference/integer_ops/add.h \
tensorflow/lite/kernels/internal/reference/integer_ops/conv.h \
tensorflow/lite/kernels/internal/reference/integer_ops/depthwise_conv.h \
tensorflow/lite/kernels/internal/reference/integer_ops/fully_connected.h \
tensorflow/lite/kernels/internal/reference/integer_ops/logistic.h \
tensorflow/lite/kernels/internal/reference/integer_ops/l2normalization.h \
tensorflow/lite/kernels/internal/reference/integer_ops/mean.h \
tensorflow/lite/kernels/internal/reference/integer_ops/mul.h \
tensorflow/lite/kernels/internal/reference/integer_ops/pooling.h \
tensorflow/lite/kernels/internal/reference/integer_ops/tanh.h \
tensorflow/lite/kernels/internal/reference/integer_ops/transpose_conv.h \
tensorflow/lite/kernels/internal/reference/l2normalization.h \
tensorflow/lite/kernels/internal/reference/leaky_relu.h \
tensorflow/lite/kernels/internal/reference/maximum_minimum.h \
tensorflow/lite/kernels/internal/reference/mul.h \
tensorflow/lite/kernels/internal/reference/neg.h \
tensorflow/lite/kernels/internal/reference/pad.h \
tensorflow/lite/kernels/internal/reference/pooling.h \
tensorflow/lite/kernels/internal/reference/prelu.h \
tensorflow/lite/kernels/internal/reference/process_broadcast_shapes.h \
tensorflow/lite/kernels/internal/reference/quantize.h \
tensorflow/lite/kernels/internal/reference/reduce.h \
tensorflow/lite/kernels/internal/reference/requantize.h \
tensorflow/lite/kernels/internal/reference/resize_nearest_neighbor.h \
tensorflow/lite/kernels/internal/reference/round.h \
tensorflow/lite/kernels/internal/reference/softmax.h \
tensorflow/lite/kernels/internal/reference/space_to_batch_nd.h \
tensorflow/lite/kernels/internal/reference/sub.h \
tensorflow/lite/kernels/internal/reference/logistic.h \
tensorflow/lite/kernels/internal/reference/strided_slice.h \
tensorflow/lite/kernels/internal/reference/tanh.h \
tensorflow/lite/kernels/internal/reference/transpose_conv.h \
tensorflow/lite/kernels/internal/cppmath.h \
tensorflow/lite/kernels/internal/max.h \
tensorflow/lite/kernels/internal/min.h \
tensorflow/lite/kernels/internal/portable_tensor.h \
tensorflow/lite/kernels/internal/strided_slice_logic.h \
tensorflow/lite/kernels/internal/tensor_ctypes.h \
tensorflow/lite/kernels/internal/types.h \
tensorflow/lite/kernels/kernel_util.h \
tensorflow/lite/kernels/op_macros.h \
tensorflow/lite/kernels/padding.h \
tensorflow/lite/portable_type_to_tflitetype.h \
tensorflow/lite/schema/schema_generated.h \
tensorflow/lite/schema/schema_utils.h

# For project generation v1, the headers that are common to all targets need to
# have a third_party prefix. Other third_party headers (e.g. CMSIS) do not have
# this requirement and are added to THIRD_PARTY_CC_HDRS with full path from the
# tensorflow root. This inconsistency may also be the reason why (for
# example) these different third party libraries are fund in different paths in
# the Arduino output tree.
#
# The convention with the (under development) project generation v2 is for all
# third party paths to be relative to the root of the git repository. We are
# keeping backwards compatibility between v1 and v2 by having a
# THIRD_PARTY_CC_HDRS_BASE variable and adding in a third_party prefix to
# THIRD_PARTY_CC_HDRS later on in the Makefile logic.
# TODO(#47413): remove this additional logic once we are ready to switch over to
# project generation v2.
THIRD_PARTY_CC_HDRS :=

# TODO(b/165940489): Figure out how to avoid including fixed point
# platform-specific headers.
THIRD_PARTY_CC_HDRS_BASE := \
gemmlowp/fixedpoint/fixedpoint.h \
gemmlowp/fixedpoint/fixedpoint_neon.h \
gemmlowp/fixedpoint/fixedpoint_sse.h \
gemmlowp/internal/detect_platform.h \
gemmlowp/LICENSE \
flatbuffers/include/flatbuffers/base.h \
flatbuffers/include/flatbuffers/stl_emulation.h \
flatbuffers/include/flatbuffers/flatbuffers.h \
flatbuffers/include/flatbuffers/flexbuffers.h \
flatbuffers/include/flatbuffers/util.h \
flatbuffers/LICENSE.txt \
ruy/ruy/profiler/instrumentation.h

MAKE_PROJECT_FILES := \
  Makefile \
  README_MAKE.md \
  .vscode/tasks.json

MBED_PROJECT_FILES := \
  README_MBED.md \
  mbed-os.lib \
  mbed_app.json

KEIL_PROJECT_FILES := \
  README_KEIL.md \
  keil_project.uvprojx

ARDUINO_PROJECT_FILES := \
  library.properties

ESP_PROJECT_FILES := \
  README_ESP.md \
  CMakeLists.txt \
  main/CMakeLists.txt \
  components/tfmicro/CMakeLists.txt

ALL_PROJECT_TARGETS :=

ARDUINO_LIBRARY_TARGETS :=
ARDUINO_LIBRARY_ZIPS :=

# For some invocations of the makefile, it is useful to avoid downloads. This
# can be achieved by explicitly passing in DISABLE_DOWNLOADS=true on the command
# line. Note that for target-specific downloads (e.g. CMSIS) there will need to
# be corresponding checking in the respecitve included makefiles (e.g.
# ext_libs/cmsis_nn.inc)
DISABLE_DOWNLOADS :=

ifneq ($(DISABLE_DOWNLOADS), true)
  # The download scripts require that the downloads directory already exist for
  # improved error checking. To accomodate that, we first create a downloads
  # directory.
  $(shell mkdir -p ${MAKEFILE_DIR}/downloads)

  # Directly download the flatbuffers library.
  DOWNLOAD_RESULT := $(shell $(MAKEFILE_DIR)/flatbuffers_download.sh ${MAKEFILE_DIR}/downloads)
  ifneq ($(DOWNLOAD_RESULT), SUCCESS)
    $(error Something went wrong with the flatbuffers download: $(DOWNLOAD_RESULT))
  endif

  DOWNLOAD_RESULT := $(shell $(MAKEFILE_DIR)/pigweed_download.sh ${MAKEFILE_DIR}/downloads)
  ifneq ($(DOWNLOAD_RESULT), SUCCESS)
    $(error Something went wrong with the pigweed download: $(DOWNLOAD_RESULT))
  endif

  include $(MAKEFILE_DIR)/third_party_downloads.inc
  THIRD_PARTY_DOWNLOADS :=
  $(eval $(call add_third_party_download,$(GEMMLOWP_URL),$(GEMMLOWP_MD5),gemmlowp,))
  $(eval $(call add_third_party_download,$(RUY_URL),$(RUY_MD5),ruy,))
  $(eval $(call add_third_party_download,$(PERSON_MODEL_URL),$(PERSON_MODEL_MD5),person_model_grayscale,))
  $(eval $(call add_third_party_download,$(PERSON_MODEL_INT8_URL),$(PERSON_MODEL_INT8_MD5),person_model_int8,))
endif

# The target-specific makefile must have a name that is exactly
# TARGET_makefile.inc and is only needed for cross-compilation (i.e. when TARGET
# is different from the HOST_OS).
# There are also some other targets like arduino and CHRE that are also special
# in that they do no have a <target>_makefile but are still used to create a
# directory for the generated artifacts. We are using a workaround right now and
# will be separating the project generation from the Makefile in the future.
TARGETS_WITHOUT_MAKEFILES := \
$(HOST_OS) \
arduino

# This specific string needs to be outputted for a test to be recognized as
# having passed.
TEST_PASS_STRING:='~~~ALL TESTS PASSED~~~'

# ${TARGET}_makefile.inc can set this to true to allow it to defined a custom
# implementation for `make test`. See bluepill_makefile as an example.
TARGET_SPECIFIC_MAKE_TEST:=0

ifeq ($(findstring $(TARGET),$(TARGETS_WITHOUT_MAKEFILES)),)
  include $(MAKEFILE_DIR)/targets/$(TARGET)_makefile.inc
endif

ifneq ($(OPTIMIZED_KERNEL_DIR),)
  include $(MAKEFILE_DIR)/ext_libs/$(OPTIMIZED_KERNEL_DIR).inc
  # Specialize for the optimized kernels
  MICROLITE_CC_KERNEL_SRCS := $(call substitute_specialized_implementations,$(MICROLITE_CC_KERNEL_SRCS),$(OPTIMIZED_KERNEL_DIR))
endif

# If a co-processor is specified on the command line with
# CO_PROCESSOR=<co_processor> then we will include ext_libs/<co_processor>.inc
# and find additional kernel sources in kernels/<co_processor>/
#
# That the co-processor specialization of the kernel sources happens after the
# optimized_kernel_dir means that if there is an implementation of the same
# kernel in both directories, the one from co_processor will be used.
ifneq ($(CO_PROCESSOR),)
  include $(MAKEFILE_DIR)/ext_libs/$(CO_PROCESSOR).inc
  # Specialize for the coprocessor kernels.
  MICROLITE_CC_KERNEL_SRCS := $(call substitute_specialized_implementations,$(MICROLITE_CC_KERNEL_SRCS),$(CO_PROCESSOR))
endif

# TODO(#47413): Remove this logic once we are switched over to the newer version
# of project generation (v2). Project generation v1 needs the "base" third party
# headers to have a prefix of third_party/. In order to support v2 prototyping
# without a lot of changes with the v1 system, we are manually adding in this
# prefix, and also making a copy in THIRD_PARTY_CC_HDRS_V2 that will be used in
# the list_third_party_headers target.
THIRD_PARTY_CC_HDRS_V2 := $(THIRD_PARTY_CC_HDRS)
THIRD_PARTY_CC_HDRS += $(addprefix third_party/,$(THIRD_PARTY_CC_HDRS_BASE))

# Specialize for debug_log. micro_time etc.
MICROLITE_CC_SRCS := $(call substitute_specialized_implementations,$(MICROLITE_CC_SRCS),$(TARGET))
MICROLITE_CC_SRCS += $(MICROLITE_CC_KERNEL_SRCS)

ALL_SRCS := \
	$(MICROLITE_CC_SRCS) \
	$(MICROLITE_TEST_SRCS)

# Where compiled objects are stored.

GENDIR := $(MAKEFILE_DIR)/gen/$(TARGET)_$(TARGET_ARCH)_$(BUILD_TYPE)/
OBJDIR := $(GENDIR)obj/
BINDIR := $(GENDIR)bin/
LIBDIR := $(GENDIR)lib/
PRJDIR := $(GENDIR)prj/

MICROLITE_LIB_PATH := $(LIBDIR)$(MICROLITE_LIB_NAME)

CXX := $(TARGET_TOOLCHAIN_ROOT)${TARGET_TOOLCHAIN_PREFIX}${CXX_TOOL}
CC := $(TARGET_TOOLCHAIN_ROOT)${TARGET_TOOLCHAIN_PREFIX}${CC_TOOL}
AR := $(TARGET_TOOLCHAIN_ROOT)${TARGET_TOOLCHAIN_PREFIX}${AR_TOOL}

# The default Makefile target(all) must appear before any target,
# which is compiled if there's no command-line arguments.
all: $(MICROLITE_LIB_PATH)

# Load the examples.
include $(MICRO_LITE_EXAMPLE_TESTS)

# Load the benchmarks.
include $(MICRO_LITE_BENCHMARKS)

# Create rules for downloading third-party dependencies.
THIRD_PARTY_TARGETS :=
$(foreach DOWNLOAD,$(THIRD_PARTY_DOWNLOADS),$(eval $(call create_download_rule,$(DOWNLOAD))))
third_party_downloads: $(THIRD_PARTY_TARGETS)

MICROLITE_LIB_OBJS := $(addprefix $(OBJDIR), \
$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(MICROLITE_CC_SRCS))))

MICROLITE_LIB_OBJS += $(addprefix $(OBJDIR), \
$(patsubst %.S,%.o,$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(THIRD_PARTY_CC_SRCS)))))

# For normal manually-created TensorFlow C++ source files.
$(OBJDIR)%.o: %.cc $(THIRD_PARTY_TARGETS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# For normal manually-created TensorFlow C source files.
$(OBJDIR)%.o: %.c $(THIRD_PARTY_TARGETS)
	@mkdir -p $(dir $@)
	$(CC) $(CCFLAGS) $(INCLUDES) -c $< -o $@

# For normal manually-created TensorFlow ASM source files.
$(OBJDIR)%.o: %.S $(THIRD_PARTY_TARGETS)
	@mkdir -p $(dir $@)
	$(CC) $(CCFLAGS) $(INCLUDES) -c $< -o $@

microlite: $(MICROLITE_LIB_PATH)

# Hack for generating schema file bypassing flatbuffer parsing
tensorflow/lite/schema/schema_generated.h:
	@cp -u tensorflow/lite/schema/schema_generated.h.oss tensorflow/lite/schema/schema_generated.h

# Gathers together all the objects we've compiled into a single '.a' archive.
$(MICROLITE_LIB_PATH): tensorflow/lite/schema/schema_generated.h $(MICROLITE_LIB_OBJS)
	@mkdir -p $(dir $@)
	$(AR) $(ARFLAGS) $(MICROLITE_LIB_PATH) $(MICROLITE_LIB_OBJS)

$(BINDIR)%_test : $(OBJDIR)%_test.o $(MICROLITE_LIB_PATH)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
	-o $@ $< \
	$(MICROLITE_LIB_PATH) $(LDFLAGS) $(MICROLITE_LIBS)

$(BINDIR)%.test_target: $(BINDIR)%_test
	@test -f $(TEST_SCRIPT) || (echo 'Unable to find the test script. Is the software emulation available in $(TARGET)?'; exit 1)
	$(TEST_SCRIPT) $< $(TEST_PASS_STRING)

# snease: Add %.bin rule here since BINDIR is now defined
# These are microcontroller-specific rules for converting the ELF output
# of the linker into a binary image that can be loaded directly.
OBJCOPY := ${TARGET_TOOLCHAIN_ROOT}$(TARGET_TOOLCHAIN_PREFIX)objcopy
$(BINDIR)%.bin: $(BINDIR)%
	@mkdir -p $(dir $@)
	$(OBJCOPY) $< $@ -O binary


# Some tests have additional dependencies (beyond libtensorflow-microlite.a) and
# those need to be explicitly specified with their own individual call to the
# microlite_test helper function. For these tests, we also need to make sure to
# not add targets for them if they have been excluded as part of the target
# specific Makefile.
EXPLICITLY_SPECIFIED_TEST:= tensorflow/lite/micro/kernels/detection_postprocess_test.cc
ifneq ($(findstring $(EXPLICITLY_SPECIFIED_TEST),$(MICROLITE_TEST_SRCS)),)
  MICROLITE_TEST_SRCS := $(filter-out $(EXPLICITLY_SPECIFIED_TEST), $(MICROLITE_TEST_SRCS))
  EXPLICITLY_SPECIFIED_TEST_SRCS := \
  $(EXPLICITLY_SPECIFIED_TEST) \
  tensorflow/lite/micro/kernels/detection_postprocess_flexbuffers_generated_data.cc
  EXPLICITLY_SPECIFIED_TEST_HDRS := \
  tensorflow/lite/micro/kernels/detection_postprocess_flexbuffers_generated_data.h
  $(eval $(call microlite_test,kernel_detection_postprocess_test,\
  $(EXPLICITLY_SPECIFIED_TEST_SRCS),$(EXPLICITLY_SPECIFIED_TEST_HDRS)))
endif

EXPLICITLY_SPECIFIED_TEST:= tensorflow/lite/micro/kernels/circular_buffer_test.cc
ifneq ($(findstring $(EXPLICITLY_SPECIFIED_TEST),$(MICROLITE_TEST_SRCS)),)
  MICROLITE_TEST_SRCS := $(filter-out $(EXPLICITLY_SPECIFIED_TEST), $(MICROLITE_TEST_SRCS))
  EXPLICITLY_SPECIFIED_TEST_SRCS := \
  $(EXPLICITLY_SPECIFIED_TEST) \
  tensorflow/lite/micro/kernels/circular_buffer_flexbuffers_generated_data.cc
  EXPLICITLY_SPECIFIED_TEST_HDRS := \
  tensorflow/lite/micro/kernels/circular_buffer_flexbuffers_generated_data.h
  $(eval $(call microlite_test,kernel_circular_buffer_test,\
  $(EXPLICITLY_SPECIFIED_TEST_SRCS),$(EXPLICITLY_SPECIFIED_TEST_HDRS)))
endif

EXPLICITLY_SPECIFIED_TEST:= tensorflow/lite/micro/memory_arena_threshold_test.cc
ifneq ($(findstring $(EXPLICITLY_SPECIFIED_TEST),$(MICROLITE_TEST_SRCS)),)
  MICROLITE_TEST_SRCS := $(filter-out $(EXPLICITLY_SPECIFIED_TEST), $(MICROLITE_TEST_SRCS))
  EXPLICITLY_SPECIFIED_TEST_SRCS := \
  $(EXPLICITLY_SPECIFIED_TEST) \
  tensorflow/lite/micro/benchmarks/keyword_scrambled_model_data.cc \
  tensorflow/lite/micro/testing/test_conv_model.cc
  EXPLICITLY_SPECIFIED_TEST_HDRS := \
  tensorflow/lite/micro/benchmarks/keyword_scrambled_model_data.h \
  tensorflow/lite/micro/testing/test_conv_model.h
  $(eval $(call microlite_test,memory_arena_threshold_test,\
  $(EXPLICITLY_SPECIFIED_TEST_SRCS),$(EXPLICITLY_SPECIFIED_TEST_HDRS)))
endif

EXPLICITLY_SPECIFIED_TEST:= tensorflow/lite/micro/micro_allocator_test.cc
ifneq ($(findstring $(EXPLICITLY_SPECIFIED_TEST),$(MICROLITE_TEST_SRCS)),)
  MICROLITE_TEST_SRCS := $(filter-out $(EXPLICITLY_SPECIFIED_TEST), $(MICROLITE_TEST_SRCS))
  EXPLICITLY_SPECIFIED_TEST_SRCS := \
  $(EXPLICITLY_SPECIFIED_TEST) \
  tensorflow/lite/micro/testing/test_conv_model.cc
  EXPLICITLY_SPECIFIED_TEST_HDRS := \
  tensorflow/lite/micro/testing/test_conv_model.h
  $(eval $(call microlite_test,micro_allocator_test,\
  $(EXPLICITLY_SPECIFIED_TEST_SRCS),$(EXPLICITLY_SPECIFIED_TEST_HDRS)))
endif

EXPLICITLY_SPECIFIED_TEST:= tensorflow/lite/micro/recording_micro_allocator_test.cc
ifneq ($(findstring $(EXPLICITLY_SPECIFIED_TEST),$(MICROLITE_TEST_SRCS)),)
  MICROLITE_TEST_SRCS := $(filter-out $(EXPLICITLY_SPECIFIED_TEST), $(MICROLITE_TEST_SRCS))
  EXPLICITLY_SPECIFIED_TEST_SRCS := \
  $(EXPLICITLY_SPECIFIED_TEST) \
  tensorflow/lite/micro/testing/test_conv_model.cc
  EXPLICITLY_SPECIFIED_TEST_HDRS := \
  tensorflow/lite/micro/testing/test_conv_model.h
  $(eval $(call microlite_test,recording_micro_allocator_test,\
  $(EXPLICITLY_SPECIFIED_TEST_SRCS),$(EXPLICITLY_SPECIFIED_TEST_HDRS)))
endif

EXPLICITLY_SPECIFIED_TEST:= tensorflow/lite/micro/kernels/conv_test.cc
ifneq ($(findstring $(EXPLICITLY_SPECIFIED_TEST),$(MICROLITE_TEST_SRCS)),)
  MICROLITE_TEST_SRCS := $(filter-out $(EXPLICITLY_SPECIFIED_TEST), $(MICROLITE_TEST_SRCS))
  EXPLICITLY_SPECIFIED_TEST_SRCS := \
  $(EXPLICITLY_SPECIFIED_TEST) \
  tensorflow/lite/micro/kernels/conv_test_common.cc
  EXPLICITLY_SPECIFIED_TEST_HDRS := \
  tensorflow/lite/micro/kernels/conv_test.h
  $(eval $(call microlite_test,kernel_conv_test,\
  $(EXPLICITLY_SPECIFIED_TEST_SRCS),$(EXPLICITLY_SPECIFIED_TEST_HDRS)))
endif

EXPLICITLY_SPECIFIED_TEST:= tensorflow/lite/micro/kernels/transpose_conv_test.cc
ifneq ($(findstring $(EXPLICITLY_SPECIFIED_TEST),$(MICROLITE_TEST_SRCS)),)
  MICROLITE_TEST_SRCS := $(filter-out $(EXPLICITLY_SPECIFIED_TEST), $(MICROLITE_TEST_SRCS))
  EXPLICITLY_SPECIFIED_TEST_SRCS := \
  $(EXPLICITLY_SPECIFIED_TEST) \
  tensorflow/lite/micro/kernels/conv_test_common.cc
  EXPLICITLY_SPECIFIED_TEST_HDRS := \
  tensorflow/lite/micro/kernels/conv_test.h
  $(eval $(call microlite_test,kernel_transpose_conv_test,\
  $(EXPLICITLY_SPECIFIED_TEST_SRCS),$(EXPLICITLY_SPECIFIED_TEST_HDRS)))
endif


# For all the tests that do not have any additional dependencies, we can
# add a make target in a common way.
$(foreach TEST_TARGET,$(filter-out tensorflow/lite/micro/kernels/%,$(MICROLITE_TEST_SRCS)),\
$(eval $(call microlite_test,$(notdir $(basename $(TEST_TARGET))),$(TEST_TARGET))))

$(foreach TEST_TARGET,$(filter tensorflow/lite/micro/kernels/%,$(MICROLITE_TEST_SRCS)),\
$(eval $(call microlite_test,kernel_$(notdir $(basename $(TEST_TARGET))),$(TEST_TARGET))))

ifeq ($(TARGET_SPECIFIC_MAKE_TEST),0)
test: $(MICROLITE_TEST_TARGETS)
endif

# Just build the test targets
build: $(MICROLITE_BUILD_TARGETS)

generate_projects: $(ALL_PROJECT_TARGETS)

ARDUINO_PROJECT_TARGETS := $(foreach TARGET,$(ALL_PROJECT_TARGETS),$(if $(findstring _arduino,$(TARGET)),$(TARGET),))

generate_arduino_zip: $(ARDUINO_PROJECT_TARGETS) $(ARDUINO_LIBRARY_ZIPS)
	python tensorflow/lite/micro/tools/make/merge_arduino_zips.py $(PRJDIR)/tensorflow_lite.zip $(ARDUINO_LIBRARY_ZIPS)

list_library_sources:
	@echo $(MICROLITE_CC_SRCS)

list_library_headers:
	@echo $(MICROLITE_CC_HDRS)

list_third_party_sources:
	@echo $(THIRD_PARTY_CC_SRCS)

list_third_party_headers:
	@echo $(addprefix $(MAKEFILE_DIR)/downloads/,$(THIRD_PARTY_CC_HDRS_BASE)) $(THIRD_PARTY_CC_HDRS_V2)

# Gets rid of all generated files.
clean:
	rm -rf $(MAKEFILE_DIR)/gen

# Removes third-party downloads.
clean_downloads:
	rm -rf $(MAKEFILE_DIR)/downloads

$(DEPDIR)/%.d: ;
.PRECIOUS: $(DEPDIR)/%.d
.PRECIOUS: $(BINDIR)%_test

-include $(patsubst %,$(DEPDIR)/%.d,$(basename $(ALL_SRCS)))
