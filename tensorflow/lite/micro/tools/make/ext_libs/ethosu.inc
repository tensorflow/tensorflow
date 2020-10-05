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

    ETHOSU_DRIVER_PATH = $(MAKEFILE_DIR)/downloads/ethosu

    # The driver need to be downloaded before the recursive_find below.
    # That won't happen with the standard way of downloading by generating a
    # target(call add_third_party_download), so instead use the shell function.
    NEED_DOWNLOAD := YES
    ifeq ($(NEED_DOWNLOAD),$(shell test -d $(ETHOSU_DRIVER_PATH) || echo $(NEED_DOWNLOAD)))
        DOWNLOAD_SCRIPT := ./tensorflow/lite/micro/tools/make/download_and_extract.sh
        DOWNLOAD_OK := OK
        DOWNLOAD_STATUS := $(shell $(DOWNLOAD_SCRIPT) $(ETHOSU_URL) $(ETHOSU_MD5) $(ETHOSU_DRIVER_PATH) >&2 && echo $(DOWNLOAD_OK))
        ifneq ($(DOWNLOAD_OK),$(DOWNLOAD_STATUS))
            $(error $(DOWNLOAD_SCRIPT) failed)
        endif
    endif

    # Currently there is a dependency to CMSIS-NN
    THIRD_PARTY_DOWNLOADS += \
        $(eval $(call add_third_party_download,$(CMSIS_URL),$(CMSIS_MD5),cmsis,patch_cmsis))
    ifeq ($(CMSIS_PATH),)
      CMSIS_PATH = $(MAKEFILE_DIR)/downloads/cmsis/
    endif
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
