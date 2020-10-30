ifneq ($(filter ethos-u,$(ALL_TAGS)),)
    # Arm Compiler will not link the Math library (see below), therefore we're filtering it out.
    # See Fatal error: L6450U: Cannot find library m:
    # "Arm Compiler is designed to run in a bare metal environment,
    # and automatically includes implementations of these functions,
    # and so no such flag is necessary."
    # https://developer.arm.com/documentation/100891/0611/troubleshooting/general-troubleshooting-advice
    MICROLITE_LIBS := $(filter-out -lm,$(MICROLITE_LIBS))

    ifneq (,$(filter $(TARGET_ARCH), x86_64))
        $(error target architecture x86_64 not supported)
    endif

    # Do not download if an external PATH is provided.
    ETHOSU_DEFAULT_DOWNLOAD_DRIVER_PATH := $(MAKEFILE_DIR)/downloads/ethosu
    ETHOSU_DRIVER_PATH := $(ETHOSU_DEFAULT_DOWNLOAD_DRIVER_PATH)
    ifeq ($(ETHOSU_DRIVER_PATH), $(ETHOSU_DEFAULT_DOWNLOAD_DRIVER_PATH))
      $(eval $(call download_third_party_download,$(ETHOSU_URL),$(ETHOSU_MD5),ethosu))
    endif
    CMSIS_DEFAULT_DOWNLOAD_PATH := $(MAKEFILE_DIR)/downloads/cmsis
    CMSIS_PATH := $(CMSIS_DEFAULT_DOWNLOAD_PATH)
    ifeq ($(CMSIS_PATH), $(CMSIS_DEFAULT_DOWNLOAD_PATH))
      $(eval $(call download_third_party_download,$(CMSIS_URL),$(CMSIS_MD5),cmsis,))
    endif

    # Currently there is a dependency to CMSIS even without TAGS=cmsis-nn.
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
