ifneq ($(filter cmsis-nn,$(ALL_TAGS)),)
    # Enable u-arch specfic behaviours
    ifneq (,$(filter $(TARGET_ARCH), cortex-m3))
        CCFLAGS += -DARM_MATH_CM3
        CXXFLAGS += -DARM_CMSIS_NN_M3
    endif
    ifneq (,$(filter $(TARGET_ARCH), cortex-m4))
        CCFLAGS += -DARM_MATH_CM4
        CXXFLAGS += -DARM_CMSIS_NN_M4
    endif
    ifneq (,$(filter $(TARGET_ARCH), cortex-m7))
        CCFLAGS += -DARM_MATH_CM7
        CXXFLAGS += -DARM_CMSIS_NN_M7
    endif
    ifneq (,$(filter $(TARGET_ARCH), x86_64))
        # For development purposes
        CCFLAGS += -DARM_MATH_CM4
        CXXFLAGS += -DARM_CMSIS_NN_X86_64
    endif

    # Setup CMSIS-NN lib and add required header files to microlite lib INCLUDE
    CMSIS_PATH = $(MAKEFILE_DIR)/downloads/cmsis/
    MICROLITE_CC_SRCS += $(shell find $(CMSIS_PATH)/CMSIS/NN/Source/ -name *.c)
    INCLUDES += -I$(CMSIS_PATH)/CMSIS/Core/Include \
                -I$(CMSIS_PATH)/CMSIS/NN/Include \
                -I$(CMSIS_PATH)/CMSIS/DSP/Include
endif
