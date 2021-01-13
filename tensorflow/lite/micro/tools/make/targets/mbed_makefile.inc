# Settings for mbed platforms.
ifeq ($(TARGET), mbed)
  TARGET_ARCH := cortex-m4
  $(eval $(call add_third_party_download,$(CMSIS_URL),$(CMSIS_MD5),cmsis,))
  $(eval $(call add_third_party_download,$(CUST_CMSIS_URL),$(CUST_CMSIS_MD5),CMSIS_ext,))

  DOWNLOAD_RESULT := $(shell $(MAKEFILE_DIR)/arm_gcc_download.sh ${MAKEFILE_DIR}/downloads)
  ifneq ($(DOWNLOAD_RESULT), SUCCESS)
    $(error Something went wrong with the GCC download: $(DOWNLOAD_RESULT))
  endif
endif
