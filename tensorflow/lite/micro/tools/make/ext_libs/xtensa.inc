ifeq ($(TARGET_ARCH), hifi4)

  LIBRARY_URL := "http://mirror.tensorflow.org/github.com/foss-xtensa/nnlib-hifi4/raw/master/archive/xa_nnlib_06_27.zip"
  LIBRARY_DIRNAME := "xa_nnlib_hifi4"
  LIBRARY_MD5 := "45fdc1209a8da62ab568aa6040f7eabf"

  DOWNLOAD_RESULT := $(shell $(MAKEFILE_DIR)/ext_libs/xtensa_download.sh ${MAKEFILE_DIR}/downloads $(LIBRARY_URL) $(LIBRARY_DIRNAME) $(LIBRARY_MD5))
  ifneq ($(DOWNLOAD_RESULT), SUCCESS)
    $(error Something went wrong with the xtensa download: $(DOWNLOAD_RESULT))
  endif

  NNLIB_PATH := $(MAKEFILE_DIR)/downloads/$(LIBRARY_DIRNAME)

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
