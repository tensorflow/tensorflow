ifneq (3.82,$(firstword $(sort $(MAKE_VERSION) 3.82)))
  $(error "Requires make version 3.82 or later (current is $(MAKE_VERSION))")
endif

# root directory of tensorflow
TENSORFLOW_ROOT :=
MAKEFILE_DIR := tensorflow/lite/micro/tools/make


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
CXX_TOOL := g++
CC_TOOL := gcc
AR_TOOL := ar

# Specify TAGS on the command line to add a particular set of specialized
# implementations, for example TAGS="CMSIS disco_f746ng" to target a Discovery
# STM32F746NG board, using the CMSIS library's implementations where possible.
ALL_TAGS := $(TAGS) $(TARGET)

# This is obviously horrible.  We need to generate these 3 versions of the
# include directories from one source.
INCLUDES := \
-I. \
-I$(MAKEFILE_DIR)/downloads/ \
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

TEST_SCRIPT := tensorflow/lite/micro/testing/test_linux_binary.sh

MICROLITE_LIBS := -lm

# TODO(b/150240249): Add in -fno-rtti once that works for the Xtensa toolchain.
CXXFLAGS := -std=c++11 -DTF_LITE_STATIC_MEMORY
CCFLAGS  := -std=c11   -DTF_LITE_STATIC_MEMORY
ARFLAGS := -r

# override these in the makefile.inc for specific compiler targets
TARGET_TOOLCHAIN_PREFIX :=
TARGET_TOOLCHAIN_ROOT :=

ifeq ($(BUILD_TYPE), debug)
	CXXFLAGS += -DDEBUG -g
	CCFLAGS  += -DDEBUG -g
else ifeq ($(BUILD_TYPE), release)
	CXXFLAGS += -DNDEBUG -O3 -DTF_LITE_STRIP_ERROR_STRINGS
	CCFLAGS  += -DNDEBUG -O3 -DTF_LITE_STRIP_ERROR_STRINGS
else
	CXXFLAGS += -DNDEBUG -O3
	CCFLAGS  += -DNDEBUG -O3
endif

# This library is the main target for this makefile. It will contain a minimal
# runtime that can be linked in to other programs.
MICROLITE_LIB_NAME := libtensorflow-microlite.a

# These two must be defined before we include the target specific Makefile.inc
# because we filter out the examples that are not supported for those targets.
# See targets/xtensa_xpg_makefile.inc for an example.
MICRO_LITE_EXAMPLE_TESTS := $(shell find tensorflow/lite/micro/examples/ -name Makefile.inc)
MICRO_LITE_BENCHMARKS := $(wildcard tensorflow/lite/micro/benchmarks/Makefile.inc)

MICROLITE_TEST_SRCS := \
$(wildcard tensorflow/lite/micro/*test.cc) \
$(wildcard tensorflow/lite/micro/kernels/*test.cc) \
$(wildcard tensorflow/lite/micro/memory_planner/*test.cc)

# TODO(b/152645559): move all benchmarks to benchmarks directory.
MICROLITE_BENCHMARK_SRCS := \
$(wildcard tensorflow/lite/micro/benchmarks/*.cc)

MICROLITE_TEST_HDRS := \
$(wildcard tensorflow/lite/micro/testing/*.h)

MICROLITE_CC_BASE_SRCS := \
$(wildcard tensorflow/lite/micro/*.cc) \
$(wildcard tensorflow/lite/micro/kernels/*.cc) \
$(wildcard tensorflow/lite/micro/memory_planner/*.cc) \
tensorflow/lite/c/common.c \
tensorflow/lite/core/api/error_reporter.cc \
tensorflow/lite/core/api/flatbuffer_conversions.cc \
tensorflow/lite/core/api/op_resolver.cc \
tensorflow/lite/core/api/tensor_utils.cc \
tensorflow/lite/kernels/internal/quantization_util.cc \
tensorflow/lite/kernels/kernel_util.cc \
tensorflow/lite/micro/testing/test_utils.cc

MICROLITE_CC_SRCS := $(filter-out $(MICROLITE_TEST_SRCS), $(MICROLITE_CC_BASE_SRCS))
MICROLITE_CC_SRCS := $(filter-out $(MICROLITE_BENCHMARK_SRCS), $(MICROLITE_CC_SRCS))

MICROLITE_CC_HDRS := \
$(wildcard tensorflow/lite/micro/*.h) \
$(wildcard tensorflow/lite/micro/kernels/*.h) \
$(wildcard tensorflow/lite/micro/memory_planner/*.h) \
LICENSE \
tensorflow/core/public/version.h \
tensorflow/lite/c/builtin_op_data.h \
tensorflow/lite/c/common.h \
tensorflow/lite/core/api/error_reporter.h \
tensorflow/lite/core/api/flatbuffer_conversions.h \
tensorflow/lite/core/api/op_resolver.h \
tensorflow/lite/core/api/profiler.h \
tensorflow/lite/core/api/tensor_utils.h \
tensorflow/lite/kernels/internal/common.h \
tensorflow/lite/kernels/internal/compatibility.h \
tensorflow/lite/kernels/internal/optimized/neon_check.h \
tensorflow/lite/kernels/internal/quantization_util.h \
tensorflow/lite/kernels/internal/reference/add.h \
tensorflow/lite/kernels/internal/reference/arg_min_max.h \
tensorflow/lite/kernels/internal/reference/binary_function.h \
tensorflow/lite/kernels/internal/reference/ceil.h \
tensorflow/lite/kernels/internal/reference/comparisons.h \
tensorflow/lite/kernels/internal/reference/concatenation.h \
tensorflow/lite/kernels/internal/reference/conv.h \
tensorflow/lite/kernels/internal/reference/depthwiseconv_float.h \
tensorflow/lite/kernels/internal/reference/depthwiseconv_uint8.h \
tensorflow/lite/kernels/internal/reference/dequantize.h \
tensorflow/lite/kernels/internal/reference/floor.h \
tensorflow/lite/kernels/internal/reference/fully_connected.h \
tensorflow/lite/kernels/internal/reference/integer_ops/add.h \
tensorflow/lite/kernels/internal/reference/integer_ops/conv.h \
tensorflow/lite/kernels/internal/reference/integer_ops/depthwise_conv.h \
tensorflow/lite/kernels/internal/reference/integer_ops/fully_connected.h \
tensorflow/lite/kernels/internal/reference/integer_ops/logistic.h \
tensorflow/lite/kernels/internal/reference/integer_ops/l2normalization.h \
tensorflow/lite/kernels/internal/reference/integer_ops/mul.h \
tensorflow/lite/kernels/internal/reference/l2normalization.h \
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
tensorflow/lite/kernels/internal/reference/sub.h \
tensorflow/lite/kernels/internal/reference/logistic.h \
tensorflow/lite/kernels/internal/reference/strided_slice.h \
tensorflow/lite/kernels/internal/reference/integer_ops/pooling.h \
tensorflow/lite/kernels/internal/cppmath.h \
tensorflow/lite/kernels/internal/strided_slice_logic.h \
tensorflow/lite/kernels/internal/tensor.h \
tensorflow/lite/kernels/internal/tensor_ctypes.h \
tensorflow/lite/kernels/internal/types.h \
tensorflow/lite/kernels/kernel_util.h \
tensorflow/lite/kernels/op_macros.h \
tensorflow/lite/kernels/padding.h \
tensorflow/lite/schema/schema_generated.h \
tensorflow/lite/string_type.h \
tensorflow/lite/string_util.h \
tensorflow/lite/type_to_tflitetype.h \
tensorflow/lite/version.h

THIRD_PARTY_CC_HDRS := \
third_party/gemmlowp/fixedpoint/fixedpoint.h \
third_party/gemmlowp/fixedpoint/fixedpoint_sse.h \
third_party/gemmlowp/internal/detect_platform.h \
third_party/gemmlowp/LICENSE \
third_party/flatbuffers/include/flatbuffers/base.h \
third_party/flatbuffers/include/flatbuffers/stl_emulation.h \
third_party/flatbuffers/include/flatbuffers/flatbuffers.h \
third_party/flatbuffers/LICENSE.txt \
third_party/ruy/ruy/profiler/instrumentation.h


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

include $(MAKEFILE_DIR)/third_party_downloads.inc
THIRD_PARTY_DOWNLOADS :=
$(eval $(call add_third_party_download,$(GEMMLOWP_URL),$(GEMMLOWP_MD5),gemmlowp,))
$(eval $(call add_third_party_download,$(FLATBUFFERS_URL),$(FLATBUFFERS_MD5),flatbuffers,))
$(eval $(call add_third_party_download,$(RUY_URL),$(RUY_MD5),ruy,))

# These target-specific makefiles should modify or replace options like
# CXXFLAGS or LIBS to work for a specific targeted architecture. All logic
# based on platforms or architectures should happen within these files, to
# keep this main makefile focused on the sources and dependencies.
include $(wildcard $(MAKEFILE_DIR)/targets/*_makefile.inc)

# Load dependencies for optimized kernel implementations.
include $(wildcard $(MAKEFILE_DIR)/ext_libs/*.inc)

# Call specialize here so that platform-specific tags can be taken into account.
MICROLITE_CC_SRCS := $(call specialize,$(MICROLITE_CC_SRCS))

ALL_TAGS += $(TARGET_ARCH)

ALL_SRCS := \
	$(MICROLITE_CC_SRCS) \
	$(MICROLITE_TEST_SRCS)

# Where compiled objects are stored.
GENDIR := $(MAKEFILE_DIR)/gen/$(TARGET)_$(TARGET_ARCH)/
OBJDIR := $(GENDIR)obj/
BINDIR := $(GENDIR)bin/
LIBDIR := $(GENDIR)lib/
PRJDIR := $(GENDIR)prj/

MICROLITE_LIB_PATH := $(LIBDIR)$(MICROLITE_LIB_NAME)

CXX := $(TARGET_TOOLCHAIN_ROOT)${TARGET_TOOLCHAIN_PREFIX}${CXX_TOOL}
CC := $(TARGET_TOOLCHAIN_ROOT)${TARGET_TOOLCHAIN_PREFIX}${CC_TOOL}
AR := $(TARGET_TOOLCHAIN_ROOT)${TARGET_TOOLCHAIN_PREFIX}${AR_TOOL}

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

# The target that's compiled if there's no command-line arguments.
all: $(MICROLITE_LIB_PATH)

microlite: $(MICROLITE_LIB_PATH)

# Hack for generating schema file bypassing flatbuffer parsing
tensorflow/lite/schema/schema_generated.h:
	@cp -u tensorflow/lite/schema/schema_generated.h.OPENSOURCE tensorflow/lite/schema/schema_generated.h

# Gathers together all the objects we've compiled into a single '.a' archive.
$(MICROLITE_LIB_PATH): tensorflow/lite/schema/schema_generated.h $(MICROLITE_LIB_OBJS)
	@mkdir -p $(dir $@)
	$(AR) $(ARFLAGS) $(MICROLITE_LIB_PATH) $(MICROLITE_LIB_OBJS)

$(BINDIR)%_test : $(OBJDIR)%_test.o $(MICROLITE_LIB_PATH)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
	-o $@ $< \
	$(LIBFLAGS) $(MICROLITE_LIB_PATH) $(LDFLAGS) $(MICROLITE_LIBS)

$(BINDIR)%.test_target: $(BINDIR)%_test
	@test -f $(TEST_SCRIPT) || (echo 'Unable to find the test script. Is the software emulation available in $(TARGET)?'; exit 1)
	$(TEST_SCRIPT) $< '~~~ALL TESTS PASSED~~~'

# snease: Add %.bin rule here since BINDIR is now defined
# These are microcontroller-specific rules for converting the ELF output
# of the linker into a binary image that can be loaded directly.
OBJCOPY := ${TARGET_TOOLCHAIN_ROOT}$(TARGET_TOOLCHAIN_PREFIX)objcopy
$(BINDIR)%.bin: $(BINDIR)%
	@mkdir -p $(dir $@)
	$(OBJCOPY) $< $@ -O binary

# Generate standalone makefile projects for all of the test targets.
$(foreach TEST_TARGET,$(filter-out tensorflow/lite/micro/kernels/%,$(MICROLITE_TEST_SRCS)),\
$(eval $(call microlite_test,$(notdir $(basename $(TEST_TARGET))),$(TEST_TARGET))))
$(foreach TEST_TARGET,$(filter tensorflow/lite/micro/kernels/%,$(MICROLITE_TEST_SRCS)),\
$(eval $(call microlite_test,kernel_$(notdir $(basename $(TEST_TARGET))),$(TEST_TARGET))))

test: $(MICROLITE_TEST_TARGETS)

# Just build the test targets
build: $(MICROLITE_BUILD_TARGETS)

generate_projects: $(ALL_PROJECT_TARGETS)

generate_non_kernel_projects: $(filter-out generate_kernel%,$(ALL_PROJECT_TARGETS))

generate_non_test_projects: $(filter-out %_test%,$(ALL_PROJECT_TARGETS))

generate_arduino_zip: generate_non_kernel_projects $(ARDUINO_LIBRARY_ZIPS)
	python tensorflow/lite/micro/tools/make/merge_arduino_zips.py $(PRJDIR)/tensorflow_lite.zip $(ARDUINO_LIBRARY_ZIPS)

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
