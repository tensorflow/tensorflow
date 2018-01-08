
# Find where we're running from, so we can store generated files here.
ifeq ($(origin MAKEFILE_DIR), undefined)
	MAKEFILE_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
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

ARCH := $(shell if [[ $(shell uname -m) =~ i[345678]86 ]]; then echo x86_32; else echo $(shell uname -m); fi)

# Where compiled objects are stored.
OBJDIR := $(MAKEFILE_DIR)/gen/obj/
BINDIR := $(MAKEFILE_DIR)/gen/bin/
LIBDIR := $(MAKEFILE_DIR)/gen/lib/
GENDIR := $(MAKEFILE_DIR)/gen/obj/

# Settings for the host compiler.
CXX := $(CC_PREFIX) gcc
CXXFLAGS := --std=c++11 -O3 -DNDEBUG
CC := $(CC_PREFIX) gcc
CFLAGS :=
LDOPTS :=
LDOPTS += -L/usr/local/lib
ARFLAGS := -r

INCLUDES := \
-I. \
-I$(MAKEFILE_DIR)/../../../ \
-I$(MAKEFILE_DIR)/downloads/ \
-I$(MAKEFILE_DIR)/downloads/eigen \
-I$(MAKEFILE_DIR)/downloads/gemmlowp \
-I$(MAKEFILE_DIR)/downloads/neon_2_sse \
-I$(MAKEFILE_DIR)/downloads/farmhash/src \
-I$(MAKEFILE_DIR)/downloads/flatbuffers/include \
-I$(GENDIR)
# This is at the end so any globally-installed frameworks like protobuf don't
# override local versions in the source tree.
INCLUDES += -I/usr/local/include

LIBS := \
-lstdc++ \
-lpthread \
-lm \
-lz

# If we're on Linux, also link in the dl library.
ifeq ($(OS),LINUX)
	LIBS += -ldl -lpthread
endif

include $(MAKEFILE_DIR)/ios_makefile.inc

# This library is the main target for this makefile. It will contain a minimal
# runtime that can be linked in to other programs.
LIB_NAME := libtensorflow-lite.a
LIB_PATH := $(LIBDIR)$(LIB_NAME)

# A small example program that shows how to link against the library.
BENCHMARK_PATH := $(BINDIR)benchmark_model

BENCHMARK_SRCS := \
tensorflow/contrib/lite/tools/benchmark_model.cc
BENCHMARK_OBJS := $(addprefix $(OBJDIR), \
$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(BENCHMARK_SRCS))))

# What sources we want to compile, must be kept in sync with the main Bazel
# build files.

CORE_CC_ALL_SRCS := \
$(wildcard tensorflow/contrib/lite/*.cc) \
$(wildcard tensorflow/contrib/lite/kernels/*.cc) \
$(wildcard tensorflow/contrib/lite/kernels/internal/*.cc) \
$(wildcard tensorflow/contrib/lite/kernels/internal/optimized/*.cc) \
$(wildcard tensorflow/contrib/lite/kernels/internal/reference/*.cc) \
$(wildcard tensorflow/contrib/lite/*.c) \
$(wildcard tensorflow/contrib/lite/kernels/*.c) \
$(wildcard tensorflow/contrib/lite/kernels/internal/*.c) \
$(wildcard tensorflow/contrib/lite/kernels/internal/optimized/*.c) \
$(wildcard tensorflow/contrib/lite/kernels/internal/reference/*.c) \
$(wildcard tensorflow/contrib/lite/downloads/farmhash/src/farmhash.cc)
# Remove any duplicates.
CORE_CC_ALL_SRCS := $(sort $(CORE_CC_ALL_SRCS))
CORE_CC_EXCLUDE_SRCS := \
$(wildcard tensorflow/contrib/lite/*test.cc) \
$(wildcard tensorflow/contrib/lite/*/*test.cc) \
$(wildcard tensorflow/contrib/lite/*/*/*test.cc) \
$(wildcard tensorflow/contrib/lite/*/*/*/*test.cc) \
$(wildcard tensorflow/contrib/lite/kernels/test_util.cc) \
$(BENCHMARK_SRCS)
# Filter out all the excluded files.
TF_LITE_CC_SRCS := $(filter-out $(CORE_CC_EXCLUDE_SRCS), $(CORE_CC_ALL_SRCS))
# File names of the intermediate files target compilation generates.
TF_LITE_CC_OBJS := $(addprefix $(OBJDIR), \
$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(TF_LITE_CC_SRCS))))
LIB_OBJS := $(TF_LITE_CC_OBJS)

# For normal manually-created TensorFlow C++ source files.
$(OBJDIR)%.o: %.cc
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# For normal manually-created TensorFlow C++ source files.
$(OBJDIR)%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CCFLAGS) $(INCLUDES) -c $< -o $@

# The target that's compiled if there's no command-line arguments.
all: $(LIB_PATH) $(BENCHMARK_PATH)

# Gathers together all the objects we've compiled into a single '.a' archive.
$(LIB_PATH): $(LIB_OBJS)
	@mkdir -p $(dir $@)
	$(AR) $(ARFLAGS) $(LIB_PATH) $(LIB_OBJS)

$(BENCHMARK_PATH): $(BENCHMARK_OBJS) $(LIB_PATH)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
	-o $(BENCHMARK_PATH) $(BENCHMARK_OBJS) \
	$(LIBFLAGS) $(LIB_PATH) $(LDFLAGS) $(LIBS)

# Gets rid of all generated files.
clean:
	rm -rf $(MAKEFILE_DIR)/gen

# Gets rid of target files only, leaving the host alone. Also leaves the lib
# directory untouched deliberately, so we can persist multiple architectures
# across builds for iOS and Android.
cleantarget:
	rm -rf $(OBJDIR)
	rm -rf $(BINDIR)

$(DEPDIR)/%.d: ;
.PRECIOUS: $(DEPDIR)/%.d

-include $(patsubst %,$(DEPDIR)/%.d,$(basename $(TF_CC_SRCS)))
