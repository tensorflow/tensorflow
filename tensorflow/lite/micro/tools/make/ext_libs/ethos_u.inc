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

# Unless an external path is provided we force a download during the first phase of make so
# that the files exist prior to the call to recursive_find below. add_third_party_download
# prevents the use of wildcards and recursive_find in selecting which files to add to THIRD_PARTY_SRCS.
ETHOSU_DEFAULT_DOWNLOAD_DRIVER_PATH := $(MAKEFILE_DIR)/downloads/ethos_u_core_driver
ETHOSU_DRIVER_PATH := $(ETHOSU_DEFAULT_DOWNLOAD_DRIVER_PATH)
ifeq ($(ETHOSU_DRIVER_PATH), $(ETHOSU_DEFAULT_DOWNLOAD_DRIVER_PATH))
  $(call $(or $(shell $(DOWNLOAD_SCRIPT) $(ETHOSU_URL) $(ETHOSU_MD5) $(ETHOSU_DRIVER_PATH) >&2 && echo SUCCESS), $(error $(DOWNLOAD_SCRIPT) failed)))
endif

THIRD_PARTY_CC_HDRS += $(call recursive_find,$(ETHOSU_DRIVER_PATH)/include,*.h)
ifeq (,$(ETHOSU_DRIVER_LIBS))
    THIRD_PARTY_CC_SRCS += $(call recursive_find,$(ETHOSU_DRIVER_PATH)/src,*.c)
else
    MICROLITE_LIBS += $(ETHOSU_DRIVER_LIBS)
endif

# Currently there is a dependency to CMSIS even without OPTIMIZED_KERNEL_DIR=cmsis_nn.
CMSIS_DEFAULT_DOWNLOAD_PATH := $(MAKEFILE_DIR)/downloads/cmsis
CMSIS_PATH := $(CMSIS_DEFAULT_DOWNLOAD_PATH)
ifeq ($(CMSIS_PATH), $(CMSIS_DEFAULT_DOWNLOAD_PATH))
  DOWNLOAD_RESULT := $(shell $(MAKEFILE_DIR)/ext_libs/cmsis_download.sh ${MAKEFILE_DIR}/downloads)
  ifneq ($(DOWNLOAD_RESULT), SUCCESS)
    $(error Something went wrong with the CMSIS download: $(DOWNLOAD_RESULT))
  endif
endif

THIRD_PARTY_CC_HDRS += $(CMSIS_PATH)/CMSIS/Core/Include/cmsis_compiler.h

INCLUDES += -I$(ETHOSU_DRIVER_PATH)/include \
            -I$(CMSIS_PATH)/CMSIS/Core/Include
GENERATED_PROJECT_INCLUDES += -I./$(ETHOSU_DRIVER_PATH)/include
