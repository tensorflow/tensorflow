ifneq ($(filter cmsis-nn,$(ALL_TAGS)),)
    # Enable u-arch specfic behaviours
    ifneq (,$(filter $(TARGET_ARCH), x86_64))
        # CMSIS-NN optimizations not supported
    endif

    # Setup CMSIS-NN lib and add required header files to microlite lib INCLUDE
    THIRD_PARTY_DOWNLOADS += \
      $(eval $(call add_third_party_download,$(CMSIS_URL),$(CMSIS_MD5),cmsis,))

    CMSIS_PATH = $(MAKEFILE_DIR)/downloads/cmsis/

    # Include CMSIS-NN files
    THIRD_PARTY_CC_SRCS += \
      $(call recursive_find,$(CMSIS_PATH)/CMSIS/NN/Source,*.c)

    THIRD_PARTY_CC_HDRS += \
      $(call recursive_find,$(CMSIS_PATH)/CMSIS/NN/Include,*.h)
    THIRD_PARTY_CC_HDRS += \
      $(call recursive_find,$(CMSIS_PATH)/CMSIS/DSP/Include,*.h)
    THIRD_PARTY_CC_HDRS += \
      $(call recursive_find,$(CMSIS_PATH)/CMSIS/Core/Include,*.h)

    INCLUDES += -I$(CMSIS_PATH)/CMSIS/Core/Include \
                -I$(CMSIS_PATH)/CMSIS/NN/Include \
                -I$(CMSIS_PATH)/CMSIS/DSP/Include
endif
