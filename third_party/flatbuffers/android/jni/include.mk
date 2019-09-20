# Copyright 2014 Google Inc. All rights reserved.
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

# This file contains utility functions for Android projects using Flatbuffers.
# To use this file, include it in your project's Android.mk by calling near the
# top of your android makefile like so:
#
#     include $(FLATBUFFERS_DIR)/android/jni/include.mk
#
# You will also need to import the flatbuffers module using the standard
# import-module function.
#
# The main functionality this file provides are the following functions:
# flatbuffers_fbs_to_h: Converts flatbuffer schema paths to header paths.
# flatbuffers_header_build_rule:
#     Creates a build rule for a schema's generated header. This build rule
#     has a dependency on the flatc compiler which will be built if necessary.
# flatbuffers_header_build_rules:
#     Creates build rules for generated headers for each schema listed and sets
#     up depenedendies.
#
# More information and example usage can be found in the comments preceeding
# each function.

# Targets to build the Flatbuffers compiler as well as some utility definitions
ifeq (,$(FLATBUFFERS_INCLUDE_MK_))
FLATBUFFERS_INCLUDE_MK_ := 1

# Portable version of $(realpath) that omits drive letters on Windows.
realpath-portable = $(join $(filter %:,$(subst :,: ,$1)),\
                      $(realpath $(filter-out %:,$(subst :,: ,$1))))

PROJECT_OS := $(OS)
ifeq (,$(OS))
PROJECT_OS := $(shell uname -s)
else
ifneq ($(findstring Windows,$(PROJECT_OS)),)
PROJECT_OS := Windows
endif
endif

# The following block generates build rules which result in headers being
# rebuilt from flatbuffers schemas.

FLATBUFFERS_CMAKELISTS_DIR := \
  $(call realpath-portable,$(dir $(lastword $(MAKEFILE_LIST)))/../..)

# Directory that contains the FlatBuffers compiler.
ifeq (Windows,$(PROJECT_OS))
FLATBUFFERS_FLATC_PATH?=$(FLATBUFFERS_CMAKELISTS_DIR)
FLATBUFFERS_FLATC := $(lastword \
                       $(wildcard $(FLATBUFFERS_FLATC_PATH)/*/flatc.exe) \
                       $(wildcard $(FLATBUFFERS_FLATC_PATH)/flatc.exe))
endif
ifeq (Linux,$(PROJECT_OS))
FLATBUFFERS_FLATC_PATH?=$(FLATBUFFERS_CMAKELISTS_DIR)
FLATBUFFERS_FLATC := $(FLATBUFFERS_FLATC_PATH)/flatc
endif
ifeq (Darwin,$(PROJECT_OS))
FLATBUFFERS_FLATC_PATH?=$(FLATBUFFERS_CMAKELISTS_DIR)
FLATBUFFERS_FLATC := $(lastword \
                       $(wildcard $(FLATBUFFERS_FLATC_PATH)/*/flatc) \
                       $(wildcard $(FLATBUFFERS_FLATC_PATH)/flatc))
endif

FLATBUFFERS_FLATC_ARGS?=

# Search for cmake.
CMAKE_ROOT := \
  $(call realpath-portable,$(LOCAL_PATH)/../../../../../../prebuilts/cmake)
ifeq (,$(CMAKE))
ifeq (Linux,$(PROJECT_OS))
CMAKE := $(wildcard $(CMAKE_ROOT)/linux-x86/current/bin/cmake*)
endif
ifeq (Darwin,$(PROJECT_OS))
CMAKE := \
  $(wildcard $(CMAKE_ROOT)/darwin-x86_64/current/*.app/Contents/bin/cmake)
endif
ifeq (Windows,$(PROJECT_OS))
CMAKE := $(wildcard $(CMAKE_ROOT)/windows/current/bin/cmake*)
endif
endif
ifeq (,$(CMAKE))
CMAKE := cmake
endif

# Windows friendly portable local path.
# GNU-make doesn't like : in paths, must use relative paths on Windows.
ifeq (Windows,$(PROJECT_OS))
PORTABLE_LOCAL_PATH =
else
PORTABLE_LOCAL_PATH = $(LOCAL_PATH)/
endif

# Generate a host build rule for the flatbuffers compiler.
ifeq (Windows,$(PROJECT_OS))
define build_flatc_recipe
	$(FLATBUFFERS_CMAKELISTS_DIR)\android\jni\build_flatc.bat \
        $(CMAKE) $(FLATBUFFERS_CMAKELISTS_DIR)
endef
endif
ifeq (Linux,$(PROJECT_OS))
define build_flatc_recipe
	+cd $(FLATBUFFERS_CMAKELISTS_DIR) && \
      $(CMAKE) . && \
      $(MAKE) flatc
endef
endif
ifeq (Darwin,$(PROJECT_OS))
define build_flatc_recipe
	cd $(FLATBUFFERS_CMAKELISTS_DIR) && "$(CMAKE)" -GXcode . && \
        xcodebuild -target flatc
endef
endif
ifeq (,$(build_flatc_recipe))
ifeq (,$(FLATBUFFERS_FLATC))
$(error flatc binary not found!)
endif
endif

# Generate a build rule for flatc.
ifeq ($(strip $(FLATBUFFERS_FLATC)),)
flatc_target := build_flatc
.PHONY: $(flatc_target)
FLATBUFFERS_FLATC := \
  python $(FLATBUFFERS_CMAKELISTS_DIR)/android/jni/run_flatc.py \
    $(FLATBUFFERS_CMAKELISTS_DIR)
else
flatc_target := $(FLATBUFFERS_FLATC)
endif
$(flatc_target):
	$(call build_flatc_recipe)

# $(flatbuffers_fbs_to_h schema_dir,output_dir,path)
#
# Convert the specified schema path to a Flatbuffers generated header path.
# For example:
#
# $(call flatbuffers_fbs_to_h,$(MY_PROJ_DIR)/schemas,\
#   $(MY_PROJ_DIR)/gen/include,$(MY_PROJ_DIR)/schemas/example.fbs)
#
# This will convert the file path `$(MY_PROJ_DIR)/schemas/example.fbs)` to
# `$(MY_PROJ_DIR)/gen/include/example_generated.h`
define flatbuffers_fbs_to_h
$(subst $(1),$(2),$(patsubst %.fbs,%_generated.h,$(3)))
endef

# $(flatbuffers_header_build_rule schema_file,schema_dir,output_dir,\
#   schema_include_dirs)
#
# Generate a build rule that will convert a Flatbuffers schema to a generated
# header derived from the schema filename using flatbuffers_fbs_to_h. For
# example:
#
# $(call flatbuffers_header_build_rule,$(MY_PROJ_DIR)/schemas/example.fbs,\
#   $(MY_PROJ_DIR)/schemas,$(MY_PROJ_DIR)/gen/include)
#
# The final argument, schema_include_dirs, is optional and is only needed when
# the schema files depend on other schema files outside their own directory.
define flatbuffers_header_build_rule
$(eval \
  $(call flatbuffers_fbs_to_h,$(2),$(3),$(1)): $(1) $(flatc_target)
	$(call host-echo-build-step,generic,Generate) \
      $(subst $(LOCAL_PATH)/,,$(call flatbuffers_fbs_to_h,$(2),$(3),$(1)))
	$(hide) $$(FLATBUFFERS_FLATC) $(FLATBUFFERS_FLATC_ARGS) \
      $(foreach include,$(4),-I $(include)) -o $$(dir $$@) -c $$<)
endef

# TODO: Remove when the LOCAL_PATH expansion bug in the NDK is fixed.
# Override the default behavior of local-source-file-path to workaround
# a bug which prevents the build of deeply nested projects when NDK_OUT is
# set.
local-source-file-path=\
$(if $(call host-path-is-absolute,$1),$1,$(call \
    realpath-portable,$(LOCAL_PATH)/$1))


# $(flatbuffers_header_build_rules schema_files,schema_dir,output_dir,\
#   schema_include_dirs,src_files,[build_target],[dependencies]))
#
# $(1) schema_files: Space separated list of flatbuffer schema files.
# $(2) schema_dir: Directory containing the flatbuffer schemas.
# $(3) output_dir: Where to place the generated files.
# $(4) schema_include_dirs: Directories to include when generating schemas.
# $(5) src_files: Files that should depend upon the headers generated from the
#   flatbuffer schemas.
# $(6) build_target: Name of a build target that depends upon all generated
#   headers.
# $(7) dependencies: Space seperated list of additional build targets src_files
#   should depend upon.
#
# Use this in your own Android.mk file to generate build rules that will
# generate header files for your flatbuffer schemas as well as automatically
# set your source files to be dependent on the generated headers. For example:
#
# $(call flatbuffers_header_build_rules,$(MY_PROJ_SCHEMA_FILES),\
#   $(MY_PROJ_SCHEMA_DIR),$(MY_PROJ_GENERATED_OUTPUT_DIR),
#   $(MY_PROJ_SCHEMA_INCLUDE_DIRS),$(LOCAL_SRC_FILES))
#
# NOTE: Due problesm with path processing in ndk-build when presented with
# deeply nested projects must redefine LOCAL_PATH after include this makefile
# using:
#
# LOCAL_PATH := $(call realpath-portable,$(LOCAL_PATH))
#
define flatbuffers_header_build_rules
$(foreach schema,$(1),\
  $(call flatbuffers_header_build_rule,\
    $(schema),$(strip $(2)),$(strip $(3)),$(strip $(4))))\
$(foreach src,$(strip $(5)),\
  $(eval $(call local-source-file-path,$(src)): \
    $(foreach schema,$(strip $(1)),\
      $(call flatbuffers_fbs_to_h,$(strip $(2)),$(strip $(3)),$(schema)))))\
$(if $(6),\
  $(foreach schema,$(strip $(1)),\
    $(eval $(6): \
      $(call flatbuffers_fbs_to_h,$(strip $(2)),$(strip $(3)),$(schema)))),)\
$(if $(7),\
  $(foreach src,$(strip $(5)),\
      $(eval $(call local-source-file-path,$(src)): $(strip $(7)))),)\
$(if $(7),\
  $(foreach dependency,$(strip $(7)),\
      $(eval $(6): $(dependency))),)
endef

endif  # FLATBUFFERS_INCLUDE_MK_
