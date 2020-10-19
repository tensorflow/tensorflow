ifneq ($(filter cmsis-nn,$(ALL_TAGS)),)
  # Enable u-arch specfic behaviours
  ifneq (,$(filter $(TARGET_ARCH), x86_64))
    # CMSIS-NN optimizations not supported
  endif

  # This should not be overriden.
  CMSIS_DEFAULT_DOWNLOAD_PATH := $(MAKEFILE_DIR)/downloads/cmsis

  # This may be overriden.
  CMSIS_PATH := $(CMSIS_DEFAULT_DOWNLOAD_PATH)

  # Download and optionally patch CMSIS repository.
  # This needs to be done before recursive_find below as it won't work with the standard
  # way of downloading by generating a target (call add_third_party_download).
  # So instead use the shell function, which will immediately download it.
  # Optional argument is: patch_cmsis
  DOWNLOAD_SCRIPT := $(MAKEFILE_DIR)/download_and_extract.sh
  DOWNLOAD_CMSIS = $(or $(shell $(DOWNLOAD_SCRIPT) $(CMSIS_URL) $(CMSIS_MD5) $(CMSIS_PATH) $(1) >&2 && echo SUCCESS), $(error $(DOWNLOAD_SCRIPT) failed))
  DOWNLOAD_CMSIS_IF_NEEDED = $(if $(shell test -d $(CMSIS_PATH) || echo NEED_DOWNLOAD), $(call DOWNLOAD_CMSIS, $(1)))

  ifeq ($(findstring $(TARGET),$(TARGETS_USING_QUALIFIED_CMSIS_NN_INCLUDE_PATH)), $(TARGET))
    # Download and patch CMSIS repository.
    # For targets using a qualified path to CMSIS, the headers need to be patched in
    # patch_cmsis() in download_and_extract.sh. See the RFC document for more details:
    # https://docs.google.com/document/d/14GRxeVEgSKgKBKAijO7oxnI49nLoTYBFQmPok-rG0cw
    result := $(call DOWNLOAD_CMSIS_IF_NEEDED, patch_cmsis)
  else
    # Download CMSIS repository unless an external CMSIS_PATH is provided from the command line.
    ifeq ($(CMSIS_PATH), $(CMSIS_DEFAULT_DOWNLOAD_PATH))
      result := $(call DOWNLOAD_CMSIS_IF_NEEDED)
    endif
  endif

  # Include CMSIS-NN files
  THIRD_PARTY_CC_SRCS += \
    $(call recursive_find,$(CMSIS_PATH)/CMSIS/NN/Source,*.c)
  THIRD_PARTY_CC_HDRS += \
    $(call recursive_find,$(CMSIS_PATH)/CMSIS/NN/Include,*.h)
  THIRD_PARTY_CC_HDRS += \
    $(call recursive_find,$(CMSIS_PATH)/CMSIS/DSP/Include,*.h)
  THIRD_PARTY_CC_HDRS += \
    $(call recursive_find,$(CMSIS_PATH)/CMSIS/Core/Include,*.h)

  # Need to add the CMSIS Core includes path.
  # All other CMSIS header files are included with their relative path
  # in the CMSIS-NN micro kernel source files in
  # tensorflow/lite/micro/kernels/cmsis-nn
  INCLUDES += \
    -I$(CMSIS_PATH)/CMSIS/Core/Include \
    -I$(CMSIS_PATH)/CMSIS/DSP/Include \
    -I$(CMSIS_PATH)/CMSIS/NN/Include

endif
