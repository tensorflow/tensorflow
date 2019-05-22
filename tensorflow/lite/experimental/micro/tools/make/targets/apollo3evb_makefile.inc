# Settings for apollo3 evb and SparkFun Edge platforms.
ifeq ($(TARGET),$(filter $(TARGET),apollo3evb sparkfun_edge))
  export PATH := $(MAKEFILE_DIR)/downloads/gcc_embedded/bin/:$(PATH)
  TARGET_ARCH := cortex-m4
  TARGET_TOOLCHAIN_PREFIX := arm-none-eabi-
  # Download the Ambiq Apollo3 SDK and set this variable to find the header
  # files:
  APOLLO3_SDK := $(MAKEFILE_DIR)/downloads/AmbiqSuite-Rel2.0.0
  # Need a pointer to the GNU ARM toolchain for crtbegin.o for the fp functions
  # with the hard interfaces.
  GCC_ARM := $(MAKEFILE_DIR)/downloads/gcc_embedded/

  $(eval $(call add_third_party_download,$(GCC_EMBEDDED_URL),$(GCC_EMBEDDED_MD5),gcc_embedded,))
  $(eval $(call add_third_party_download,$(CMSIS_URL),$(CMSIS_MD5),cmsis,))
  $(eval $(call add_third_party_download,$(AM_SDK_URL),$(AM_SDK_MD5),AmbiqSuite-Rel2.0.0,patch_am_sdk))
  $(eval $(call add_third_party_download,$(AP3_URL),$(AP3_MD5),apollo3_ext,))
  $(eval $(call add_third_party_download,$(CUST_CMSIS_URL),$(CUST_CMSIS_MD5),CMSIS_ext,))

  ifeq ($(TARGET), sparkfun_edge)
    $(eval $(call add_third_party_download,$(SPARKFUN_EDGE_BSP_URL),$(SPARKFUN_EDGE_BSP_MD5),AmbiqSuite-Rel2.0.0/boards/SparkFun_TensorFlow_Apollo3_BSP,))
    # Make sure that we download the full Ambiq SDK before the SparkFun one.
$(MAKEFILE_DIR)/downloads/AmbiqSuite-Rel2.0.0/boards/SparkFun_TensorFlow_Apollo3_BSP: $(MAKEFILE_DIR)/downloads/AmbiqSuite-Rel2.0.0
  endif

  # Use the faster depthwise conv implementation.
  ALL_TAGS += portable_optimized

  PLATFORM_FLAGS = \
    -DPART_apollo3 \
    -DAM_PACKAGE_BGA \
    -DAM_PART_APOLLO3 \
    -DGEMMLOWP_ALLOW_SLOW_SCALAR_FALLBACK \
    -DTF_LITE_STATIC_MEMORY \
    -DNDEBUG \
    -DTF_LITE_MCU_DEBUG_LOG \
    -D __FPU_PRESENT=1 \
    -DARM_MATH_CM4 \
    -fno-rtti \
    -fmessage-length=0 \
    -fno-exceptions \
    -fno-unwind-tables \
    -fno-builtin \
    -ffunction-sections \
    -fdata-sections \
    -funsigned-char \
    -MMD \
    -mcpu=cortex-m4 \
    -mthumb \
    -mfpu=fpv4-sp-d16 \
    -mfloat-abi=hard \
    -std=gnu++11 \
    -Wvla \
    -Wall \
    -Wextra \
    -Wno-unused-parameter \
    -Wno-missing-field-initializers \
    -Wno-write-strings \
    -Wno-sign-compare \
    -fno-delete-null-pointer-checks \
    -fomit-frame-pointer \
    -fpermissive \
    -nostdlib \
    -ggdb \
    -O3
  CXXFLAGS += $(PLATFORM_FLAGS)
  CCFLAGS += $(PLATFORM_FLAGS)
  LDFLAGS += \
    -mthumb -mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=hard \
    -nostartfiles -static \
    -Wl,--gc-sections -Wl,--entry,Reset_Handler \
    -Wl,--start-group -lm -lc -lgcc -Wl,--end-group \
    -fno-exceptions \
    -nostdlib --specs=nano.specs -t -lstdc++ -lc -lnosys -lm \
    -Wl,-T,$(APOLLO3_SDK)/boards/apollo3_evb/examples/hello_world/gcc_patched/apollo3evb.ld \
    -Wl,-Map=$(MAKEFILE_DIR)/gen/$(TARGET).map,--cref
  BUILD_TYPE := micro
  ifeq ($(TARGET), apollo3evb)
    BOARD_BSP_PATH := $(APOLLO3_SDK)/boards/apollo3_evb/bsp
  endif
  ifeq ($(TARGET), sparkfun_edge)
    BOARD_BSP_PATH := $(APOLLO3_SDK)/boards/SparkFun_TensorFlow_Apollo3_BSP/bsp
  endif
  MICROLITE_LIBS := \
    $(BOARD_BSP_PATH)/gcc/bin/libam_bsp.a \
    $(APOLLO3_SDK)/mcu/apollo3/hal/gcc/bin/libam_hal.a \
    $(GCC_ARM)/lib/gcc/arm-none-eabi/7.3.1/thumb/v7e-m/fpv4-sp/hard/crtbegin.o \
    -lm
  INCLUDES += \
    -isystem$(MAKEFILE_DIR)/downloads/cmsis/CMSIS/Core/Include/ \
    -isystem$(MAKEFILE_DIR)/downloads/cmsis/CMSIS/DSP/Include/ \
    -I$(MAKEFILE_DIR)/downloads/CMSIS_ext/ \
    -I$(GCC_ARM)/arm-none-eabi/ \
    -I$(APOLLO3_SDK)/mcu/apollo3/ \
    -I$(APOLLO3_SDK)/CMSIS/AmbiqMicro/Include/ \
    -I$(BOARD_BSP_PATH) \
    -I$(APOLLO3_SDK)/devices/ \
    -I$(APOLLO3_SDK)/utils/

  # The startup_gcc.c file is an altered version of the examples/hello_world/gcc/startup_gcc.c
  # file from Ambiq:
  #   - Increase the stack size from 1k to 20k
  #   - Change the application entry call from main() to _main()
  # The am_*.c files should be copied from the Ambiq Apollo3 SDK
  # _main.c contains application and target specific initialization, like
  # setting clock speed, default uart setups, etc. and an implementation
  # of the DebugLog interfaces.
  MICROLITE_CC_SRCS += \
    $(APOLLO3_SDK)/boards/apollo3_evb/examples/hello_world/gcc_patched/startup_gcc.c \
    $(APOLLO3_SDK)/utils/am_util_delay.c \
    $(APOLLO3_SDK)/utils/am_util_faultisr.c \
    $(APOLLO3_SDK)/utils/am_util_id.c \
    $(APOLLO3_SDK)/utils/am_util_stdio.c

  ifeq ($(TARGET), apollo3evb)
    MICROLITE_CC_SRCS += \
      $(APOLLO3_SDK)/devices/am_devices_led.c
  endif

  CMSIS_SRC_DIR := tensorflow/lite/experimental/micro/tools/make/downloads/cmsis/CMSIS/DSP/Source
  CMSIS_SRCS := \
  $(CMSIS_SRC_DIR)/BasicMathFunctions/arm_mult_q15.c \
  $(CMSIS_SRC_DIR)/TransformFunctions/arm_rfft_init_q15.c \
  $(CMSIS_SRC_DIR)/TransformFunctions/arm_rfft_q15.c \
  $(CMSIS_SRC_DIR)/TransformFunctions/arm_cfft_q15.c \
  $(CMSIS_SRC_DIR)/TransformFunctions/arm_cfft_radix4_q15.c \
  $(CMSIS_SRC_DIR)/CommonTables/arm_const_structs.c \
  $(CMSIS_SRC_DIR)/CommonTables/arm_common_tables.c \
  $(CMSIS_SRC_DIR)/StatisticsFunctions/arm_mean_q15.c \
  $(CMSIS_SRC_DIR)/StatisticsFunctions/arm_max_q7.c

  AP3_EXT_MICRO_DIR := $(MAKEFILE_DIR)/downloads/apollo3_ext
  AP3_MICRO_DIR := tensorflow/lite/experimental/micro/examples/micro_speech/apollo3
  CMSIS_DIR := tensorflow/lite/experimental/micro/examples/micro_speech/CMSIS
  CMSIS_EXT_DIR := $(MAKEFILE_DIR)/downloads/CMSIS_ext

  MICRO_SPEECH_TEST_SRCS += \
    $(AP3_MICRO_DIR)/_main.c

  TEST_SCRIPT := tensorflow/lite/experimental/log_test/test_apollo3evb_binary.sh
  # These are tests that don't currently work on the Apollo3 board.
  EXCLUDED_TESTS := \
    tensorflow/lite/experimental/micro/micro_interpreter_test.cc \
    tensorflow/lite/experimental/micro/simple_tensor_allocator_test.cc
  MICROLITE_TEST_SRCS := $(filter-out $(EXCLUDED_TESTS), $(MICROLITE_TEST_SRCS))

endif
