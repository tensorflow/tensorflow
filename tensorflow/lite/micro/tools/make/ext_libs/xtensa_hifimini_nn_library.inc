ifneq ($(filter xtensa-xpg, $(ALL_TAGS)),)

    XTENSA_PATH = ./tensorflow/lite/micro/kernels/xtensa_hifimini

    ifneq (,$(filter xtensa_hifimini%, $(ALL_TAGS)))

        CCFLAGS += -DNNLIB_V2 \
                   -DHIFI_MINI_NNLIB_OPT \
                   -DDISABLE_NNLIB_UNALIGNED_SUPPORT \
                   -DXTENSA_NNLIB_MAX_SCRATCH_SIZE=70*1024

        CXXFLAGS += -DNNLIB_V2 \
                   -DHIFI_MINI_NNLIB_OPT \
                   -DDISABLE_NNLIB_UNALIGNED_SUPPORT \
                    -DXTENSA_NNLIB_MAX_SCRATCH_SIZE=70*1024

        MICROLITE_CC_SRCS += \
			              $(XTENSA_PATH)/xa_nnlib/algo/common/src/xa_nnlib_common_api.c \
                    $(XTENSA_PATH)/xa_nnlib/algo/kernels/activations/hifi_mini/xa_nn_activations_asym8s_asym8s.c \
                    $(XTENSA_PATH)/xa_nnlib/algo/kernels/activations/hifi_mini/xa_nn_activations_asym8u_asym8u.c \
                    $(XTENSA_PATH)/xa_nnlib/algo/kernels/activations/hifi_mini/xa_nn_softmax_asym8_asym8.c \
                    $(XTENSA_PATH)/xa_nnlib/algo/kernels/basic/hifi_mini/xa_nn_dot_prod_16x16.c \
                    $(XTENSA_PATH)/xa_nnlib/algo/kernels/cnn/hifimini/xa_nn_circ_buf.c \
                    $(XTENSA_PATH)/xa_nnlib/algo/kernels/cnn/hifimini/xa_nn_conv2d_depthwise_asym8uxasym8u.c \
                    $(XTENSA_PATH)/xa_nnlib/algo/kernels/cnn/hifimini/xa_nn_conv2d_depthwise.c \
                    $(XTENSA_PATH)/xa_nnlib/algo/kernels/cnn/hifimini/xa_nn_conv2d_depthwise_sym8sxasym8s.c \
                    $(XTENSA_PATH)/xa_nnlib/algo/kernels/cnn/hifimini/xa_nn_conv2d_std_asym8uxasym8u.c \
                    $(XTENSA_PATH)/xa_nnlib/algo/kernels/cnn/hifimini/xa_nn_conv2d_std_circ_buf.c \
                    $(XTENSA_PATH)/xa_nnlib/algo/kernels/cnn/hifimini/xa_nn_conv2d_std_sym8sxasym8s.c \
                    $(XTENSA_PATH)/xa_nnlib/algo/kernels/cnn/hifimini/xa_nn_matXvec_asym8xasym8_asym8_circ.c \
                    $(XTENSA_PATH)/xa_nnlib/algo/kernels/cnn/hifimini/xa_nn_matXvec_sym8xsym8_sym8_circ.c \
                    $(XTENSA_PATH)/xa_nnlib/algo/kernels/fc/hifi_mini/xa_nn_fully_connected.c \
                    $(XTENSA_PATH)/xa_nnlib/algo/kernels/matXvec/hifi_mini/xa_nn_matXvec_asym8uxasym8u.c \
                    $(XTENSA_PATH)/xa_nnlib/algo/kernels/matXvec/hifi_mini/xa_nn_matXvec_sym8sxasym8s.c \


        INCLUDES += -I$(XTENSA_PATH)/xa_nnlib/algo/kernels/ \
                    -I$(XTENSA_PATH)/xa_nnlib/include/nnlib/ \
                    -I$(XTENSA_PATH)/xa_nnlib/include/ \
                    -I$(XTENSA_PATH)/xa_nnlib/algo/common/include/ \
                    -I$(XTENSA_PATH)/xa_nnlib/algo/ndsp/hifimini/include/ \

    endif

endif
