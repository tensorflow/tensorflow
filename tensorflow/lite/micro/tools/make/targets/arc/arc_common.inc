# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Common Settings for ARC platform and its projects. 
# Might be reused across different targets

ifeq ($(TARGET_ARCH), arc)

  DLR := $$$$

  # List of pairs <dst>!<src>. Each of pairs declares destination file in generated project tree,
  # and source file in user environment. Destination and source are separated by '!' symbol 
  # likewise to "add_third_party_download" define in helper_functions.inc
  ARC_TARGET_COPY_FILES ?=

  # For the following variables see arc_app_makefile.tpl for usage

  # Additional text into application settings section of arc makefile project 
  ARC_EXTRA_APP_SETTINGS ?=

  # Additional text into application general rules of arc makefile project 
  ARC_EXTRA_APP_RULES ?=
  
  # additional arguments for RM command of "clean" target rule ("make clean" command)
  ARC_EXTRA_RM_TARGETS ?=

  # Dependencies of "flash" target rule ("make flash" command)
  ARC_BIN_DEPEND ?=
  
  # Commands in "flash" target rule ("make flash" command)
  ARC_BIN_RULE ?= \t$(DLR)\(error Flash rule isnt defined for this ARC target\)
  
  # Command to run app on "make run" command of generated project
  ARC_APP_RUN_CMD ?= 
  
  # Command to run app on "make debug" command of generated project
  ARC_APP_DEBUG_CMD ?= 
  
  # Additional text into application execution rules of arc makefile project 
  ARC_EXTRA_EXECUTE_RULES ?= 

# We overwrite project generator to exclude everything not relevant to ARC platform.
# ARC targets cannot work with non-ARC development tools.
# Basic make project is updated to be applicable for general ARC platform
define generate_microlite_projects
$(call generate_project,make,$(MAKE_PROJECT_FILES),$(1),$(MICROLITE_CC_SRCS) $(THIRD_PARTY_CC_SRCS) $(2),$(MICROLITE_CC_HDRS) $(THIRD_PARTY_CC_HDRS) $(MICROLITE_TEST_HDRS) $(3),$(LDFLAGS) $(MICROLITE_LIBS),$(CXXFLAGS) $(GENERATED_PROJECT_INCLUDES), $(CCFLAGS) $(GENERATED_PROJECT_INCLUDES),$(TARGET_TOOLCHAIN_ROOT),$(TARGET_TOOLCHAIN_PREFIX))
$(call generate_arc_project,make,$(MAKE_PROJECT_FILES),$(1),$(MICROLITE_CC_SRCS) $(THIRD_PARTY_CC_SRCS) $(2),$(MICROLITE_CC_HDRS) $(THIRD_PARTY_CC_HDRS) $(MICROLITE_TEST_HDRS) $(3),$(LDFLAGS) $(GENERATED_PROJECT_LIBS),$(CXXFLAGS) $(GENERATED_PROJECT_INCLUDES), $(CCFLAGS) $(GENERATED_PROJECT_INCLUDES))
endef

# Copy rule generator to do file copies with changing paths in generated project
# Arguments are:
# 1 - Path to file in generated project (destination).
# 2 - Path to files in the source repo (source).
# Used in helper_functions.inc for arc projects to copy files
define path_changing_copy_file
$(1) : $(2) third_party_downloads
	@mkdir -p $$(dir $$@)
	@cp $$< $$@
endef

# These are microcontroller-specific rules for converting the ELF output
# of the linker into a binary image that can be loaded directly.
# Not applicable for ARC, leaving it empty.
$(BINDIR)%.bin:


ifeq ($(ARC_TOOLCHAIN), mwdt)
  CC_TOOL := ccac
  AR_TOOL := arac
  CXX_TOOL := ccac
  LD_TOOL := ccac

  ARC_APP_RUN_CMD = mdb -run -jit -tcf=$(TCF_FILE_NAME) $(DLR)\(DBG_ARGS\)
  ARC_APP_DEBUG_CMD = mdb -OK -jit -tcf=$(TCF_FILE_NAME) $(DLR)\(DBG_ARGS\)

  # The variable TCF_FILE stores path to Tool Configuration File (*.tcf). 
  # This file is used by MWDT toolchain to properly compile/run code
  TCF_FILE ?= 

  LCF_FILE ?= 

  BUILD_ARC_MLI ?= true

# The variable TCF_FILE_NAME stores the TCF file name (including .tcf extension), 
# this variable is used later to add the option to the linker/compiler flags.
# This condition also handles the case when the user/makefile specifies 
# the configuration bundled with MWDT (usually without .tcf extension) and that doesn't require copying.
ifneq (,$(findstring .tcf,$(TCF_FILE)))
  TCF_FILE_NAME = $(notdir $(TCF_FILE))
  ARC_TARGET_COPY_FILES += $(notdir $(TCF_FILE))!$(TCF_FILE)
  MAKE_PROJECT_FILES += $(TCF_FILE_NAME)
else
  TCF_FILE_NAME = $(TCF_FILE)
endif

  PLATFORM_FLAGS = -tcf=$(TCF_FILE_NAME) -tcf_core_config
  
  PLATFORM_FLAGS += -Hnocopyr -Hpurge -Hdense_prologue -Hon=Long_enums -fslp-vectorize-aggressive -ffunction-sections -fdata-sections 
  
  # Use compact CRT. It requires pre-defined heap size
  PLATFORM_FLAGS += -Hcl -Hcrt_fast_memcpy -Hcrt_fast_memset 
  
  PLATFORM_LDFLAGS = -tcf=$(TCF_FILE_NAME) 
  
  PLATFORM_LDFLAGS += -Hnocopyr -m -Hldopt=-Coutput=memory.map -Hheap=24K 

ifneq ($(LCF_FILE), )
  PLATFORM_LDFLAGS += $(notdir $(LCF_FILE))
  ARC_TARGET_COPY_FILES += $(notdir $(LCF_FILE))!$(LCF_FILE)
  MAKE_PROJECT_FILES += $(notdir $(LCF_FILE))
endif

  CXXFLAGS := $(filter-out -std=c++11,$(CXXFLAGS))
  CCFLAGS := $(filter-out -std=c11,$(CCFLAGS))

  ldflags_to_remove = -Wl,--fatal-warnings -Wl,--gc-sections
  LDFLAGS := $(filter-out $(ldflags_to_remove),$(LDFLAGS))
  
  MICROLITE_LIBS := $(filter-out -lm,$(MICROLITE_LIBS))

  CXXFLAGS += $(PLATFORM_FLAGS)
  CCFLAGS += $(PLATFORM_FLAGS)
  LDFLAGS += $(PLATFORM_LDFLAGS)

endif # ARC_TOOLCHAIN

else
  $(error "Only ARC target architecture supported (TARGET_ARCH=arc)")

endif  # TARGET_ARCH
