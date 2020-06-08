# Settings for Silicon Labs EFM32/EFX32 platforms
ifneq ($(filter $(TARGET),stk3701a),)
 
  GCC_ARM := $(MAKEFILE_DIR)/downloads/gcc_embedded/

  export PATH := $(GCC_ARM):$(PATH)
  TARGET_ARCH := cortex-m4
  TARGET_TOOLCHAIN_PREFIX := arm-none-eabi-
  # Download the EFx32 TF-Lite Micro library and set this variable to find the header
  # files:
  SILABS_EFX32_LIB := $(MAKEFILE_DIR)/downloads/$(SILABS_EFX32_LIB_DEST)
  GECKO_SDK_DIR := $(SILABS_EFX32_LIB)/components/gecko_sdk_suite/v27/
  BINDIR := $(MAKEFILE_DIR)/gen/$(TARGET)_$(TARGET_ARCH)/bin
  APP_NAME := $(subst _bin,,$(firstword $(MAKECMDGOALS)))

  $(eval $(call add_third_party_download,$(GCC_EMBEDDED_URL),$(GCC_EMBEDDED_MD5),gcc_embedded,))
  $(eval $(call add_third_party_download,$(SILABS_EFX32_LIB_URL),$(SILABS_EFX32_LIB_MD5),$(SILABS_EFX32_LIB_DEST),))

  # Use the CMSIS-NN optimized kernels
  ALL_TAGS += silabs efx32 cmsis-nn

  COMMON_FLAGS := \
    -mthumb -mno-thumb-interwork \
     --specs $(GCC_ARM)/arm-none-eabi/lib/thumb/v7e-m/fpv5/hard/nano.specs \
     --specs $(GCC_ARM)/arm-none-eabi/lib/thumb/v7e-m/fpv5/hard/nosys.specs \
    -nostartfiles -ffunction-sections -fdata-sections -ggdb -ffreestanding \
    -Wall -Wextra -fno-common -fno-delete-null-pointer-checks \
    -Wno-unused-parameter \
    -D__PROGRAM_START -D__STARTUP_CLEAR_BSS \
    -D__STACK_SIZE=10240 -D__HEAP_SIZE=262144
    
  # Cortex-CM4 common flags
  ifneq ($(filter $(TARGET), stk3701a),)
    PLATFORM_FLAGS += \
      -mfloat-abi=hard -mfpu=fpv4-sp-d16 -mcpu=cortex-m4 \
      -D__ARM_FEATURE_DSP=1  -DARM_MATH_CM4
    PLATFORM_LDFLAGS += -mfloat-abi=hard -mfpu=fpv4-sp-d16 -mcpu=cortex-m4
  endif
    

  # STK3701A config
  ifeq ($(TARGET), stk3701a)
    PLATFORM_FLAGS += \
      -DEFM32GG11B820F2048GM64 -DEFM32GG11BxxxF2048 \
      -DEFM32GG11B -D_SILICON_LABS_32B_SERIES_1_CONFIG=1 \
      -DRETARGET_VCOM
    PLATFORM_LDFLAGS += -Wl,-T$(GECKO_SDK_DIR)platform/Device/SiliconLabs/EFM32GG11B/Source/GCC/efm32gg11b.ld 
    PLATFORM_INCLUDES += \
      -I$(GECKO_SDK_DIR)platform/Device/SiliconLabs/EFM32GG11B/Include \
      -I$(GECKO_SDK_DIR)hardware/kit/SLSTK3701A_EFM32GG11/config
      
    PLATFORM_CC_SRCS += $(addprefix $(GECKO_SDK_DIR)platform/Device/SiliconLabs/EFM32GG11B/,\
      Source/system_efm32gg11b.c \
      Source/GCC/startup_efm32gg11b.c)
  endif

  CXXFLAGS += \
    $(COMMON_FLAGS) $(PLATFORM_FLAGS) -fno-threadsafe-statics \
     -Wno-sign-compare -Wno-unused-but-set-variable -Wno-comment -Wno-type-limits -Wno-strict-aliasing \
     -Wno-missing-field-initializers -Wno-deprecated-declarations \
     -Wno-maybe-uninitialized -Wno-unused-variable -Wno-unused-value
  CCFLAGS += $(COMMON_FLAGS) $(PLATFORM_FLAGS)
  LDFLAGS += $(PLATFORM_LDFLAGS) \
    -Wl,--cref -mthumb -Wl,--no-wchar-size-warning \
     -ffunction-sections -fdata-sections -ggdb \
    -u _sbrk \
    -ffreestanding -nodefaultlibs -flto -Wl,--gc-sections \
    -Wl,-Map=$(BINDIR)/$(APP_NAME).map,--cref
  
  
  BUILD_TYPE := micro


  MICROLITE_LIBS := -lstdc++ -lm -lc -lgcc
  INCLUDES += \
    -isystem$(MAKEFILE_DIR)/downloads/cmsis/CMSIS/Core/Include/ \
    -isystem$(MAKEFILE_DIR)/downloads/cmsis/CMSIS/DSP/Include/ \
    $(PLATFORM_INCLUDES) \
    -I$(GECKO_SDK_DIR)platform/emlib/inc \
    -I$(GECKO_SDK_DIR)hardware/kit/common/bsp \
    -I$(GECKO_SDK_DIR)hardware/kit/common/drivers

  # emlib
  PLATFORM_CC_SRCS += $(addprefix $(GECKO_SDK_DIR)platform/emlib/,\
     src/em_assert.c \
     src/em_core.c \
     src/em_cmu.c \
     src/em_emu.c \
     src/em_gpio.c \
     src/em_ldma.c \
     src/em_system.c \
     src/em_usart.c)
     
   # Microphone
   PLATFORM_CC_SRCS += $(addprefix $(SILABS_EFX32_LIB)/components/microphone/,\
     microphone.c platform_dma.c)
   INCLUDES += \
     -I$(SILABS_EFX32_LIB)/components/microphone \
     -I$(SILABS_EFX32_LIB)/components/microphone/config 

     
   # BSP
   PLATFORM_CC_SRCS += $(addprefix $(GECKO_SDK_DIR)hardware/kit/common/bsp/,\
     bsp_bcc.c bsp_stk_leds.c bsp_stk.c bsp_trace.c)
   PLATFORM_CC_SRCS += $(addprefix $(GECKO_SDK_DIR)hardware/kit/common/drivers/,\
     retargetio.c retargetserial.c)

  # Newlib Nano overrides
  MICROLITE_CC_SRCS +=\
    $(PLATFORM_CC_SRCS) \
    $(SILABS_EFX32_LIB)/tools/toolchains/gcc/arm/newlib_nano_cpp.cc \
    $(SILABS_EFX32_LIB)/tools/toolchains/gcc/arm/newlib_nano.c
  
  # Currently only the micro_speech example is supported
  MICRO_LITE_EXAMPLE_TESTS := tensorflow/lite/micro/examples/micro_speech/Makefile.inc









OBJCOPY := ${TARGET_TOOLCHAIN_ROOT}$(TARGET_TOOLCHAIN_PREFIX)objcopy
ELFSIZE := ${TARGET_TOOLCHAIN_ROOT}$(TARGET_TOOLCHAIN_PREFIX)size


print_size: $(BINDIR)/$(APP_NAME)
	$(ELFSIZE) $< 

$(BINDIR)/$(APP_NAME).bin: $(BINDIR)/$(APP_NAME) print_size
	@mkdir -p $(dir $@)
	$(OBJCOPY) $< $@ -O binary
	
$(BINDIR)/$(APP_NAME).s37: $(BINDIR)/$(APP_NAME)
	@mkdir -p $(dir $@)
	$(OBJCOPY) $< $@ -O srec


ifeq ($(HOST_OS),windows)
COMMANDER := $(SILABS_EFX32_LIB)/tools/commander/windows/commander.exe
endif
ifeq ($(HOST_OS),osx)
COMMANDER := $(SILABS_EFX32_LIB)/tools/commander/osx/MacOS/commander
endif
ifeq ($(HOST_OS),linux)
COMMANDER := $(SILABS_EFX32_LIB)/tools/commander/linux/commander/commander
endif

download_commander:
	$(eval IS_WINDOWS_UBUNTU=$(shell $(SILABS_EFX32_LIB)/tools/check_if_windows_ubuntu.sh))
	$(if $(filter $(IS_WINDOWS_UBUNTU),true),\
	  $(eval COMMANDER=$(SILABS_EFX32_LIB)/tools/commander/windows/commander.exe)\
	  $(eval CMDER_HOST_OS=windows))
	@python --version >/dev/null || echo "'python' not found in environment PATH"
	@python $(SILABS_EFX32_LIB)/tools/python/download_commander.py \
	        --host_os $(if $(CMDER_HOST_OS),$(CMDER_HOST_OS),$(HOST_OS))\
	        --output $(dir $(COMMANDER))/download_details.txt

download: $(BINDIR)/$(APP_NAME).s37 download_commander
	$(info Downloading $< ...)
	$(COMMANDER) flash --device EFM32 --force $<


reset: download_commander
	$(COMMANDER) device reset --device EFM32

endif