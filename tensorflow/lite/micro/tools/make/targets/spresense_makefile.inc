# Settings for Spresense based platforms
# For spresense, Tensorflow lite Microcontroller is used as a library.

# This setting makefile accept 4 optional parameters on the make command line.
# These options below are needed for build an example of Tensorflow Microcontroller.
# But just build a library, no need to add those options.
#
#   SPRESENSE_DEFS       : This is the file path to Make.defs which includes configuration
#                          parameters of spresense.
#   SPRESENSE_CONFIG_H   : This is the file path to config.h which includes configuration
#                          parameters for source code.
#   SPRESENSE_CURDIR     : This is the directory path of externals/tensorflow in spresense
#                          source repository.
#   SPRESENSE_APP_TFMAKE : This is the file path to makefile.inc for additional source code
#                          in spresense to use tensorflow.

# Evacuate Compiler flags to avoid override them with loading Spresense Config
TMP_CXXFLAGS := $(CXXFLAGS)
TMP_CCLAGS := $(CCFLAGS)

# Define empty variable for add spresense specific settings
SPRESENSE_PLATFORM_FLAGS :=

ifneq ($(SPRESENSE_DEFS),)

# Load Spresense Config
include $(SPRESENSE_DEFS)

SPRESENSE_PLATFORM_FLAGS := \
  -DSPRESENSE_CONFIG_H="\"$(SPRESENSE_CONFIG_H)\"" \
  -I$(SPRESENSE_CURDIR)/wrapper_include

# Load application for Tensorflow lite micro in Spresense
ifneq ($(SPRESENSE_APP_TFMAKE),)
ifeq ($(CONFIG_EXTERNALS_TENSORFLOW_EXAMPLE_NONE),y)
-include $(SPRESENSE_APP_TFMAKE)
endif
endif

endif

TARGET_ARCH := cortex-m4
TARGET_TOOLCHAIN_PREFIX := arm-none-eabi-

PLATFORM_FLAGS = \
  $(SPRESENSE_PLATFORM_FLAGS) \
  -DGEMMLOWP_ALLOW_SLOW_SCALAR_FALLBACK \
  -DTF_LITE_STATIC_MEMORY \
  -DTF_LITE_MCU_DEBUG_LOG \
  -fmessage-length=0 \
  -fno-exceptions \
  -fno-unwind-tables \
  -ffunction-sections \
  -fdata-sections \
  -funsigned-char \
  -MMD \
  -mcpu=cortex-m4 \
  -mabi=aapcs \
  -mthumb \
  -mfpu=fpv4-sp-d16 \
  -mfloat-abi=hard \
  -Wall \
  -Wextra \
  -Wno-shadow \
  -Wno-vla \
  -Wno-strict-aliasing \
  -Wno-type-limits \
  -Wno-unused-parameter \
  -Wno-missing-field-initializers \
  -Wno-write-strings \
  -Wno-sign-compare \
  -Wunused-function \
  -fno-delete-null-pointer-checks \
  -fomit-frame-pointer \
  -Os

CXXFLAGS := $(TMP_CXXFLAGS) $(PLATFORM_FLAGS) -std=gnu++11 -fno-rtti -fno-use-cxa-atexit
CCFLAGS := $(TMP_CCFLAGS) $(PLATFORM_FLAGS)

BUILD_TYPE := micro

INCLUDES +=  -isystem$(MAKEFILE_DIR)/downloads/cmsis/CMSIS/Core/Include/

THIRD_PARTY_CC_SRCS := \
    $(THIRD_PARTY_CC_SRCS) \
    $(MAKEFILE_DIR)/../../spresense/compiler_specific.cc \

# TODO: Now Spresense environment is not support tests.
#       So remove every tests.
MICROLITE_TEST_SRCS := 
MICRO_LITE_EXAMPLE_TESTS :=

# These are microcontroller-specific rules for converting the ELF output
# of the linker into a binary image that can be loaded directly.
OBJCOPY := $(TARGET_TOOLCHAIN_PREFIX)objcopy

$(BINDIR)/%.bin: $(BINDIR)/%
	@mkdir -p $(dir $@)
	$(OBJCOPY) $< $@ -O binary

