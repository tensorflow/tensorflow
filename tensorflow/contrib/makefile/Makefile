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

# Host compilation settings

# Find where we're running from, so we can store generated files here.
MAKEFILE_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

# Try to figure out the host system
HOST_OS =
ifeq ($(OS),Windows_NT)
	HOST_OS = WINDOWS
else
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Linux)
	        HOST_OS = LINUX
	endif
	ifeq ($(UNAME_S),Darwin)
		HOST_OS = OSX
	endif
endif

# Where compiled objects are stored.
HOST_OBJDIR = $(MAKEFILE_DIR)/gen/host_obj/
HOST_BINDIR = $(MAKEFILE_DIR)/gen/host_bin/
HOST_GENDIR = $(MAKEFILE_DIR)/gen/host_obj/

# Settings for the host compiler.
HOST_CXX = gcc
HOST_CXXFLAGS = --std=c++11
HOST_LDOPTS = \
-L/usr/local/lib
HOST_INCLUDES = \
-I/usr/local/include \
-I. \
-I$(MAKEFILE_DIR)/downloads/ \
-I$(MAKEFILE_DIR)/downloads/eigen-eigen-a5e9085a94e8 \
-I$(HOST_GENDIR)
HOST_LIBS = \
-lstdc++ \
-lprotobuf \
-lm

# If we're on Linux, also link in the dl library.
ifeq ($(HOST_OS),LINUX)
	HOST_LIBS += -ldl
endif

# proto_text is a tool that converts protobufs into a form we can use more
# compactly within TensorFlow. It's a bit like protoc, but is designed to
# produce a much more minimal result so we can save binary space.
# We have to build it on the host system first so that we can create files
# that are needed for the runtime building.
PROTO_TEXT = $(HOST_BINDIR)proto_text
# The list of dependencies is derived from the Bazel build file by running
# the gen_file_lists.sh script on a system with a working Bazel setup.
PROTO_TEXT_CC_FILES = $(shell cat $(MAKEFILE_DIR)/proto_text_cc_files.txt)
PROTO_TEXT_PB_CC_LIST = $(shell cat $(MAKEFILE_DIR)/proto_text_pb_cc_files.txt)
PROTO_TEXT_PB_H_LIST = $(shell cat $(MAKEFILE_DIR)/proto_text_pb_h_files.txt)

# Locations of the intermediate files proto_text generates.
PROTO_TEXT_PB_H_FILES = $(addprefix $(HOST_GENDIR), $(PROTO_TEXT_PB_H_LIST))
PROTO_TEXT_CC_OBJS = $(addprefix $(HOST_OBJDIR), $(PROTO_TEXT_CC_FILES:.cc=.o))
PROTO_TEXT_PB_OBJS = $(addprefix $(HOST_OBJDIR), $(PROTO_TEXT_PB_CC_LIST:.cc=.o))
PROTO_TEXT_OBJS = $(PROTO_TEXT_CC_OBJS) $(PROTO_TEXT_PB_OBJS)

# Target device settings.

# Default to running on the same system we're compiling on.
# You should override TARGET on the command line if you're cross-compiling, e.g.
# make -f tensorflow/contrib/makefile/Makefile TARGET=ANDROID
TARGET = $(HOST_OS)

# Where compiled objects are stored.
OBJDIR = $(MAKEFILE_DIR)/gen/obj/
LIBDIR = $(MAKEFILE_DIR)/gen/lib/
BINDIR = $(MAKEFILE_DIR)/gen/bin/
PBTGENDIR = $(MAKEFILE_DIR)/gen/proto_text/
PROTOGENDIR = $(MAKEFILE_DIR)/gen/proto/

# Settings for the target compiler.
CXX = gcc
CXXFLAGS = --std=c++11
LDFLAGS = \
-L/usr/local/lib
INCLUDES = \
-I/usr/local/include \
-I. \
-I$(MAKEFILE_DIR)/downloads/ \
-I$(MAKEFILE_DIR)/downloads/eigen-eigen-a5e9085a94e8 \
-I$(PROTOGENDIR) \
-I$(PBTGENDIR)
LIBS = \
-lstdc++ \
-lprotobuf \
-lm
PROTOC = protoc
PROTOCFLAGS =
AR = ar
ARFLAGS = -r
LIBFLAGS =

# If we're on OS X, make sure that globals aren't stripped out.
ifeq ($(TARGET),OSX)
	LDFLAGS += -all_load
endif
# Make sure that we don't strip global constructors on Linux.
ifeq ($(TARGET),LINUX)
	LIBFLAGS += -Wl,--allow-multiple-definition -Wl,--whole-archive
endif
# If we're on Linux, also link in the dl library.
ifeq ($(TARGET),LINUX)
	LIBS += -ldl
endif

# This library is the main target for this makefile. It will contain a minimal
# runtime that can be linked in to other programs.
LIB_NAME = $(LIBDIR)tf_lib.a

# A small example program that shows how to link against the library.
BENCHMARK_NAME = $(BINDIR)benchmark

# What sources we want to compile, derived from the main Bazel build using the
# gen_file_lists.sh script.
TF_CC_SRCS = $(shell cat $(MAKEFILE_DIR)/tf_cc_files.txt)
PBT_CC_SRCS = $(shell cat $(MAKEFILE_DIR)/tf_pb_text_files.txt)
PROTO_SRCS = $(shell cat $(MAKEFILE_DIR)/tf_proto_files.txt)
BENCHMARK_SRCS = \
tensorflow/tools/benchmark/benchmark_model.cc \
tensorflow/tools/benchmark/benchmark_model_main.cc

# File names of the intermediate files target compilation generates.
TF_CC_OBJS = $(addprefix $(OBJDIR), $(TF_CC_SRCS:.cc=.o))
PBT_GEN_FILES = $(addprefix $(PBTGENDIR), $(PBT_CC_SRCS))
PBT_OBJS = $(addprefix $(OBJDIR), $(PBT_CC_SRCS:.cc=.o))
PROTO_CC_SRCS = $(addprefix $(PROTOGENDIR), $(PROTO_SRCS:.proto=.pb.cc))
PROTO_OBJS = $(addprefix $(OBJDIR), $(PROTO_SRCS:.proto=.pb.o))
LIB_OBJS = $(PROTO_OBJS) $(TF_CC_OBJS) $(PBT_OBJS)
BENCHMARK_OBJS = $(addprefix $(OBJDIR), $(BENCHMARK_SRCS:.cc=.o))

.PHONY: clean

# The target that's compiled if there's no command-line arguments.
all: $(LIB_NAME) $(BENCHMARK_NAME)

# Rules for target compilation.

# Gathers together all the objects we've compiled into a single '.a' archive.
$(LIB_NAME): $(LIB_OBJS)
	@mkdir -p $(dir $@)
	$(AR) $(ARFLAGS) $(LIB_NAME) $(LIB_OBJS)

$(BENCHMARK_NAME): $(BENCHMARK_OBJS) $(LIB_NAME)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
	-o $(BENCHMARK_NAME) $(BENCHMARK_OBJS) \
	$(LIBFLAGS) $(LIB_NAME) $(LDFLAGS) $(LIBS)

# Matches on the normal hand-written TensorFlow C++ source files.
$(OBJDIR)%.o: %.cc 
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

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
