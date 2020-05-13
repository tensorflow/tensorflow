ifneq ($(filter ethos-u,$(ALL_TAGS)),)
    # Don't want -lm flag
    MICROLITE_LIBS :=

    ifneq (,$(filter $(TARGET_ARCH), x86_64))
        $(error target architecture x86_64 not supported)
    endif

    THIRD_PARTY_DOWNLOADS += \
      $(eval $(call add_third_party_download,$(ETHOSU_URL),$(ETHOSU_MD5),ethosu,))
    ETHOSU_DRIVER_PATH = $(MAKEFILE_DIR)/downloads/ethosu

    # Currently there is a dependency to CMSIS-NN
    THIRD_PARTY_DOWNLOADS += \
        $(eval $(call add_third_party_download,$(CMSIS_URL),$(CMSIS_MD5),cmsis,))
    CMSIS_PATH = $(MAKEFILE_DIR)/downloads/cmsis/
    THIRD_PARTY_CC_HDRS += $(call recursive_find,$(CMSIS_PATH)/CMSIS/Core/Include,*.h)

    THIRD_PARTY_CC_HDRS += $(call recursive_find,$(ETHOSU_DRIVER_PATH)/include,*.h)
    ifeq (,$(ETHOSU_DRIVER_LIBS))
        THIRD_PARTY_CC_SRCS += $(call recursive_find,$(ETHOSU_DRIVER_PATH)/src,*.c)
    else
        MICROLITE_LIBS += $(ETHOSU_DRIVER_LIBS)
    endif

    INCLUDES += -I$(ETHOSU_DRIVER_PATH)/include \
                -I$(CMSIS_PATH)/CMSIS/Core/Include
    GENERATED_PROJECT_INCLUDES += -I./$(ETHOSU_DRIVER_PATH)/include
endif
