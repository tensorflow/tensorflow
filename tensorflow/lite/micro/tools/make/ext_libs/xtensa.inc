# Explicitly add kernel sources specific to the Xtensa optimized
# implementations.
MICROLITE_CC_KERNEL_SRCS += \
  tensorflow/lite/micro/kernels/xtensa/softmax_int8_int16.cc

ifeq ($(TARGET_ARCH), $(findstring $(TARGET_ARCH), "hifi5"))

  DOWNLOAD_RESULT := $(shell $(MAKEFILE_DIR)/ext_libs/xtensa_download.sh ${MAKEFILE_DIR}/downloads hifi5)
  ifneq ($(DOWNLOAD_RESULT), SUCCESS)
    $(error Something went wrong with the xtensa download: $(DOWNLOAD_RESULT))
  endif

  # TODO(b/161489252): -Wno-shadow is only needed for xannlib. But since we do
  # not have separate cflags (or the concept of modular build targets) with the
  # Makefile, -Wno-shadow will be used for everything.

  PLATFORM_FLAGS = \
    -DNNLIB_HIFI5 \
    -Wno-shadow

  CCFLAGS += $(PLATFORM_FLAGS)
  CXXFLAGS += $(PLATFORM_FLAGS)

  NNLIB_PATH := $(MAKEFILE_DIR)/downloads/xa_nnlib_hifi5

  THIRD_PARTY_CC_SRCS += \
    $(NNLIB_PATH)/algo/kernels/matXvec/hifi5/xa_nn_matmul_8x8.c \
    $(NNLIB_PATH)/algo/kernels/matXvec/hifi5/xa_nn_matmul_asym8xasym8.c \
    $(NNLIB_PATH)/algo/kernels/matXvec/hifi5/xa_nn_matXvec_sym8sxasym8s.c \
    $(NNLIB_PATH)/algo/kernels/matXvec/hifi5/xa_nn_matXvec_16x16.c \
    $(NNLIB_PATH)/algo/kernels/matXvec/hifi5/xa_nn_matXvec_8x16.c \
    $(NNLIB_PATH)/algo/kernels/matXvec/hifi5/xa_nn_matXvec_8x8.c \
    $(NNLIB_PATH)/algo/kernels/activations/hifi5/xa_nn_activations_8_8.c \
    $(NNLIB_PATH)/algo/kernels/fc/hifi4/xa_nn_fully_connected.c \
    $(NNLIB_PATH)/algo/kernels/activations/hifi5/xa_nn_softmax_asym8_asym8.c \
    $(NNLIB_PATH)/algo/kernels/cnn/hifi5/xa_nn_matXvec_sym8sxasym8s_asym8s_circ.c \
    $(NNLIB_PATH)/algo/kernels/cnn/hifi5/xa_nn_circ_buf.c \
    $(NNLIB_PATH)/algo/kernels/cnn/hifi5/xa_nn_conv2d_std_circ_buf.c \
    $(NNLIB_PATH)/algo/kernels/cnn/hifi5/xa_nn_conv2d_std_sym8sxasym8s.c \
    $(NNLIB_PATH)/algo/kernels/cnn/hifi5/xa_nn_conv2d_depthwise.c \
    $(NNLIB_PATH)/algo/kernels/cnn/hifi5/xa_nn_conv2d_depthwise_sym8sxasym8s.c \
    $(NNLIB_PATH)/algo/kernels/cnn/hifi5/xa_nn_conv2d_pointwise_sym8sxasym8s.c \
    $(NNLIB_PATH)/algo/kernels/matXvec/hifi5/xa_nn_matmul_sym8sxasym8s.c

  INCLUDES += \
    -I$(NNLIB_PATH)/ \
    -I$(NNLIB_PATH)/algo/kernels/ \
    -I$(NNLIB_PATH)/include/nnlib/ \
    -I$(NNLIB_PATH)/include/ \
    -I$(NNLIB_PATH)/algo/common/include/ \
    -I$(NNLIB_PATH)/algo/ndsp/hifi5/include/

endif

ifeq ($(TARGET_ARCH), $(findstring $(TARGET_ARCH), "fusion_f1 hifi4"))

  DOWNLOAD_RESULT := $(shell $(MAKEFILE_DIR)/ext_libs/xtensa_download.sh ${MAKEFILE_DIR}/downloads hifi4)
  ifneq ($(DOWNLOAD_RESULT), SUCCESS)
    $(error Something went wrong with the xtensa download: $(DOWNLOAD_RESULT))
  endif

  # TODO(b/161489252): -Wno-shadow is only needed for xannlib. But since we do
  # not have separate cflags (or the concept of modular build targets) with the
  # Makefile, -Wno-shadow will be used for everything.

  PLATFORM_FLAGS = \
    -DNNLIB_V2 \
    -Wno-shadow

  CCFLAGS += $(PLATFORM_FLAGS)
  CXXFLAGS += $(PLATFORM_FLAGS)

  NNLIB_PATH := $(MAKEFILE_DIR)/downloads/xa_nnlib_hifi4

  THIRD_PARTY_CC_SRCS += \
    $(shell find $(NNLIB_PATH) -name "*.c")

  EXCLUDED_NNLIB_SRCS = \
    $(NNLIB_PATH)/algo/layers/cnn/src/xa_nn_cnn_api.c \
    $(NNLIB_PATH)/algo/layers/gru/src/xa_nn_gru_api.c \
    $(NNLIB_PATH)/algo/layers/lstm/src/xa_nn_lstm_api.c

  THIRD_PARTY_CC_SRCS := $(filter-out $(EXCLUDED_NNLIB_SRCS), $(THIRD_PARTY_CC_SRCS))

  INCLUDES += \
    -I$(NNLIB_PATH)/ \
    -I$(NNLIB_PATH)/algo/kernels/ \
    -I$(NNLIB_PATH)/include/nnlib/ \
    -I$(NNLIB_PATH)/include/ \
    -I$(NNLIB_PATH)/algo/common/include/ \
    -I$(NNLIB_PATH)/algo/ndsp/hifi4/include/

endif
