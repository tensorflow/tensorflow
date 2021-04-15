#******************************************************************************
# Copyright (c) 2019 Cadence Design Systems, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including without
# limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to
# whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#*****************************************************************************/

# To build the code using Xtensa tools set following environment variables: 
#
# - PATH : make sure to include extensa tools in your path 
# - XTENSA_BASE  must be set to location of the Xtensa developer tools 
#       installation directory. 
#       for example: /scratch/ivp-source/xtensa/XtDevTools/install 
# - XTENSA_TOOLS_VERSION: For example: RI-2019.2-linux 
# - XTENSA_CORE: The name of the Xtensa core to use 
#       for example: visionp6_ao

ifeq ($(TARGET), xtensa_vision)
  TARGET_ARCH := visionp6_ao

  ifndef XTENSA_BASE
    $(error XTENSA_BASE is undefined)
  endif

  ifndef XTENSA_TOOLS_VERSION
    $(error XTENSA_TOOLS_VERSION is undefined)
  endif

  ifndef XTENSA_CORE
    $(error XTENSA_CORE is undefined)
  endif

#TODO  $(eval $(call add_third_party_download,$(XTENSA_HIFI4_URL),$(XTENSA_HIFI4_MD5),xa_nnlib_hifi4,))

   PLATFORM_ARGS = \
      -mcoproc     \
      -DXTENSA  \
      -DMAX_RFFT_PWR=9 \
      -DMIN_RFFT_PWR=MAX_RFFT_PWR \
      -fdata-sections \
      -ffunction-sections \
      -fno-exceptions \
      -fno-unwind-tables \
      -fno-use-cxa-atexit \
      -fmessage-length=0 \
      -fno-threadsafe-statics \
      -Wmacro-redefined \
      -mlongcalls  \

  TF_LITE_MICRO_FLAGS = \
    -DTF_LITE_STATIC_MEMORY\

  export PATH := $(XTENSA_BASE)/tools/$(XTENSA_TOOLS_VERSION)/XtensaTools/bin:$(PATH)
  TARGET_TOOLCHAIN_PREFIX := xt-
  CXX_TOOL := clang++
  CC_TOOL := clang

  CXXFLAGS = -O3 $(PLATFORM_ARGS) -std=c++11 $(TF_LITE_MICRO_FLAGS)
  #TODO: Use -std=c11 ?
  CCFLAGS = -O3 $(PLATFORM_ARGS) $(TF_LITE_MICRO_FLAGS)

  LDFLAGS += -mlsp=sim-stacklocal -lidma-xtos

  CC_WARNINGS += -Wno-strict-overflow \
                 -Wno-type-limits

  TEST_SCRIPT := tensorflow/lite/micro/testing/test_xtensa_vision_binary.sh

  # tests to exclude

  EXCLUDED_TESTS := \
    tensorflow/lite/micro/micro_interpreter_test.cc \
    tensorflow/lite/micro/memory_arena_threshold_test.cc
  MICROLITE_TEST_SRCS := $(filter-out $(EXCLUDED_TESTS), $(MICROLITE_TEST_SRCS))

  # These are microcontroller-specific rules for converting the ELF output
  # of the linker into a binary image that can be loaded directly.
  OBJCOPY := $(TARGET_TOOLCHAIN_PREFIX)objcopy

  $(BINDIR)/%.bin: $(BINDIR)/%
	  echo "here"
	  @mkdir -p $(dir $@)
	  $(OBJCOPY) $< $@ -O binary
endif
