TARGET_ARCH := cortex-m4

DOWNLOAD_RESULT := $(shell $(MAKEFILE_DIR)/arm_gcc_download.sh ${MAKEFILE_DIR}/downloads)
export PATH := $(MAKEFILE_DIR)/downloads/gcc_embedded/bin/:$(PATH)
ifneq ($(DOWNLOAD_RESULT), SUCCESS)
  $(error Something went wrong with the GCC download: $(DOWNLOAD_RESULT))
endif
