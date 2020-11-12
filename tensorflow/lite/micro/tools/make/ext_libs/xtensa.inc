ifeq ($(TARGET_ARCH), hifi4)

  DOWNLOAD_RESULT := $(shell $(MAKEFILE_DIR)/ext_libs/xtensa_download.sh ${MAKEFILE_DIR}/downloads)
  ifneq ($(DOWNLOAD_RESULT), SUCCESS)
    $(error Something went wrong with the xtensa download: $(DOWNLOAD_RESULT))
  endif

  NNLIB_PATH := $(MAKEFILE_DIR)/downloads/xa_nnlib_hifi4

  THIRD_PARTY_CC_SRCS += \
    $(shell find $(NNLIB_PATH) -name "*.c")

  $(info THIRD_PARTY_CC_SRCS: $(THIRD_PARTY_CC_SRCS))

  INCLUDES += \
    -I$(NNLIB_PATH)/algo/kernels/ \
    -I$(NNLIB_PATH)/include/nnlib/ \
    -I$(NNLIB_PATH)/include/ \
    -I$(NNLIB_PATH)/algo/common/include/ \
    -I$(NNLIB_PATH)/algo/ndsp/hifi4/include/

endif
