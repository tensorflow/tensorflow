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

# ARM Cortex M makefile targeted for a FVP based on Arm Corstone-300 software.
# For more info see: tensorflow/lite/micro/cortex_m_corstone_300/README.md

export PATH := $(MAKEFILE_DIR)/downloads/corstone300/models/Linux64_GCC-6.4:$(PATH)
DOWNLOAD_RESULT := $(shell $(MAKEFILE_DIR)/corstone_300_download.sh ${MAKEFILE_DIR}/downloads)
ifneq ($(DOWNLOAD_RESULT), SUCCESS)
  $(error Something went wrong with the Arm Corstone-300 software download: $(DOWNLOAD_RESULT))
endif

ETHOS_U_CORE_PLATFORM := ${PWD}/$(MAKEFILE_DIR)/downloads/ethos_u_core_platform/targets/corstone-300
DOWNLOAD_RESULT := $(shell $(MAKEFILE_DIR)/ethos_u_core_platform_download.sh ${MAKEFILE_DIR}/downloads)
ifneq ($(DOWNLOAD_RESULT), SUCCESS)
  $(error Something went wrong with the Ethos-U Core Platform software download: $(DOWNLOAD_RESULT))
endif

# This target has dependencies to CMSIS-Device so just in case running without OPTIMIZED_KERNEL_DIR=cmsis_nn.
CMSIS_DEFAULT_DOWNLOAD_PATH := $(MAKEFILE_DIR)/downloads/cmsis
CMSIS_PATH := $(CMSIS_DEFAULT_DOWNLOAD_PATH)
ifeq ($(CMSIS_PATH), $(CMSIS_DEFAULT_DOWNLOAD_PATH))
  DOWNLOAD_RESULT := $(shell $(MAKEFILE_DIR)/ext_libs/cmsis_download.sh ${MAKEFILE_DIR}/downloads)
  ifneq ($(DOWNLOAD_RESULT), SUCCESS)
    $(error Something went wrong with the CMSIS download: $(DOWNLOAD_RESULT))
  endif
endif

# Convert downloaded person detect int8 model.
DOWNLOAD_RESULT := $(shell $(MAKEFILE_DIR)/ext_libs/person_detection_int8_vela_convert.sh ${MAKEFILE_DIR}/downloads)
ifneq ($(DOWNLOAD_RESULT), SUCCESS)
  $(error Something went wrong with the person detection int8 model conversion: $(DOWNLOAD_RESULT))
endif

FLOAT := soft
GCC_TARGET_ARCH := $(TARGET_ARCH)

ifeq ($(TARGET_ARCH), cortex-m0)
  CORE=M0

else ifeq ($(TARGET_ARCH), cortex-m3)
  CORE=M3

else ifeq ($(TARGET_ARCH), cortex-m33)
  CORE=M33
  FLOAT=hard
  CMSIS_ARM_FEATURES := _DSP_DP

else ifeq ($(TARGET_ARCH), cortex-m33+nodsp)
  CORE=M33

else ifeq ($(TARGET_ARCH), cortex-m4)
  CORE=M4
  GCC_TARGET_ARCH := cortex-m4+nofp

else ifeq ($(TARGET_ARCH), cortex-m4+fp)
  CORE=M4
  FLOAT=hard
  GCC_TARGET_ARCH := cortex-m4
  CMSIS_ARM_FEATURES := _FP

else ifeq ($(TARGET_ARCH), cortex-m55)
  CORE=M55
  FLOAT=hard

else ifeq ($(TARGET_ARCH), cortex-m55+nodsp+nofp)
  CORE=M55

else ifeq ($(TARGET_ARCH), cortex-m55+nofp)
  CORE=M55

else ifeq ($(TARGET_ARCH), cortex-m7)
  CORE=M7
  GCC_TARGET_ARCH := cortex-m7+nofp

else ifeq ($(TARGET_ARCH), cortex-m7+fp)
  CORE=M7
  FLOAT=hard
  GCC_TARGET_ARCH := cortex-m7
  CMSIS_ARM_FEATURES := _DP

else
  $(error "TARGET_ARCH=$(TARGET_ARCH) is not supported")
endif

ifneq ($(filter cortex-m55%,$(TARGET_ARCH)),)
  # soft-abi=soft disables MVE - use softfp instead for M55.
  ifeq ($(FLOAT),soft)
    FLOAT=softfp
  endif
endif

ifeq ($(TOOLCHAIN), gcc)
  TARGET_DEFAULT_TOOLCHAIN_ROOT := $(MAKEFILE_DIR)/downloads/gcc_embedded/bin/
  TARGET_TOOLCHAIN_ROOT := $(TARGET_DEFAULT_TOOLCHAIN_ROOT)
  ifeq ($(TARGET_TOOLCHAIN_ROOT), $(TARGET_DEFAULT_TOOLCHAIN_ROOT))
    DOWNLOAD_RESULT := $(shell $(MAKEFILE_DIR)/arm_gcc_download.sh ${MAKEFILE_DIR}/downloads)
    ifneq ($(DOWNLOAD_RESULT), SUCCESS)
      $(error Something went wrong with the GCC download: $(DOWNLOAD_RESULT))
    endif
  endif
  TARGET_TOOLCHAIN_PREFIX := arm-none-eabi-

  FLAGS_GCC = -mcpu=$(GCC_TARGET_ARCH) -mfpu=auto
  CXXFLAGS += $(FLAGS_GCC)
  CCFLAGS += $(FLAGS_GCC)

  LDFLAGS += \
    --specs=nosys.specs \
    -T $(ETHOS_U_CORE_PLATFORM)/platform_parsed.ld \
    -Wl,-Map=${TENSORFLOW_ROOT}$(MAKEFILE_DIR)/gen/$(TARGET).map,--cref \
    -Wl,--gc-sections \
    --entry Reset_Handler

else
  $(error "TOOLCHAIN=$(TOOLCHAIN) is not supported.")
endif

# TODO(#47718): resolve warnings.
OMIT_ERRORS = \
  -Wno-implicit-fallthrough \
  -Wno-strict-aliasing

PLATFORM_FLAGS = \
  -DTF_LITE_MCU_DEBUG_LOG \
  -mthumb \
  -mfloat-abi=$(FLOAT) \
  -funsigned-char \
  -mlittle-endian \
  ${OMIT_ERRORS} \
  -fomit-frame-pointer \
  -MD \
  -DCPU_$(CORE)=1

# Common + C/C++ flags
CXXFLAGS += $(PLATFORM_FLAGS)
CCFLAGS += $(PLATFORM_FLAGS)

ARM_CPU := $(subst cortex-m,ARMCM,$(GCC_TARGET_ARCH))
ARM_CPU := $(subst +nofp,,$(ARM_CPU))
CXXFLAGS += -D$(ARM_CPU)$(CMSIS_ARM_FEATURES)
CCFLAGS += -D$(ARM_CPU)$(CMSIS_ARM_FEATURES)

# For Ethos-U Core Driver. Header file name is depending on target architecture.
CXXFLAGS += -DCMSIS_DEVICE_ARM_CORTEX_M_XX_HEADER_FILE=\"$(ARM_CPU).h\"

THIRD_PARTY_CC_SRCS += \
  $(ETHOS_U_CORE_PLATFORM)/retarget.c \
  $(ETHOS_U_CORE_PLATFORM)/uart.c

CMSIS_DEFAULT_DOWNLOAD_PATH := $(MAKEFILE_DIR)/downloads/cmsis
CMSIS_PATH := $(CMSIS_DEFAULT_DOWNLOAD_PATH)
THIRD_PARTY_CC_SRCS += \
  $(CMSIS_PATH)/Device/ARM/$(ARM_CPU)/Source/system_$(ARM_CPU).c \
  $(CMSIS_PATH)/Device/ARM/$(ARM_CPU)/Source/startup_$(ARM_CPU).c
INCLUDES += \
  -I$(CMSIS_PATH)/Device/ARM/$(ARM_CPU)/Include \
  -I$(CMSIS_PATH)/CMSIS/Core/Include

# TODO(#47071): Examine why Micro benchmarks fails.
MICRO_LITE_BENCHMARKS := $(filter-out tensorflow/lite/micro/benchmarks/Makefile.inc, $(MICRO_LITE_BENCHMARKS))

# TODO(#47070): Examine why some tests fail here.
EXCLUDED_TESTS := \
  tensorflow/lite/micro/micro_interpreter_test.cc \
  tensorflow/lite/micro/micro_allocator_test.cc \
  tensorflow/lite/micro/memory_helpers_test.cc \
  tensorflow/lite/micro/micro_error_reporter_test.cc \
  tensorflow/lite/micro/output_handler_test.cc \
  tensorflow/lite/micro/memory_arena_threshold_test.cc \
  tensorflow/lite/micro/recording_micro_allocator_test.cc \
  tensorflow/lite/micro/kernels/circular_buffer_test.cc
MICROLITE_TEST_SRCS := $(filter-out $(EXCLUDED_TESTS), $(MICROLITE_TEST_SRCS))
EXCLUDED_EXAMPLE_TESTS := \
  tensorflow/lite/micro/examples/magic_wand/Makefile.inc \
  tensorflow/lite/micro/examples/micro_speech/Makefile.inc \
  tensorflow/lite/micro/examples/hello_world/Makefile.inc \
  tensorflow/lite/micro/examples/image_recognition_experimental/Makefile.inc
MICRO_LITE_EXAMPLE_TESTS := $(filter-out $(EXCLUDED_EXAMPLE_TESTS), $(MICRO_LITE_EXAMPLE_TESTS))

TEST_SCRIPT := tensorflow/lite/micro/testing/test_with_arm_corstone_300.sh
