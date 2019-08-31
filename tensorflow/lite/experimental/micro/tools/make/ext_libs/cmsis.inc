ifneq ($(filter cmsis-nn,$(ALL_TAGS)),)
    # Enable u-arch specfic behaviours
    ifneq (,$(filter $(TARGET_ARCH), cortex-m3))
        # CMSIS-NN optimizations not supported
    endif
    ifneq (,$(filter $(TARGET_ARCH), cortex-m4))
        CCFLAGS += -DARM_MATH_DSP
        CXXFLAGS += -DARM_MATH_DSP
    endif
    ifneq (,$(filter $(TARGET_ARCH), cortex-m7))
        CCFLAGS += -DARM_MATH_DSP
        CXXFLAGS += -DARM_MATH_DSP
    endif
    ifneq (,$(filter $(TARGET_ARCH), x86_64))
        # CMSIS-NN optimizations not supported
    endif

    # Setup CMSIS-NN lib and add required header files to microlite lib INCLUDE
    CMSIS_PATH = $(MAKEFILE_DIR)/downloads/cmsis/
    THIRD_PARTY_CC_SRCS += $(shell find $(CMSIS_PATH)/CMSIS/NN/Source/ -name *.c)
    THIRD_PARTY_CC_HDRS += $(shell find $(CMSIS_PATH)/CMSIS/Core/Include/ -name *.h) \
                           $(shell find $(CMSIS_PATH)/CMSIS/NN/Include/ -name *.h) \
                           $(shell find $(CMSIS_PATH)/CMSIS/DSP/Include/ -name *.h)
    INCLUDES += -I$(CMSIS_PATH)/CMSIS/Core/Include \
                -I$(CMSIS_PATH)/CMSIS/NN/Include \
                -I$(CMSIS_PATH)/CMSIS/DSP/Include
endif
