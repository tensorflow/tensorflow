ifneq ($(filter xtensa_hifi, $(ALL_TAGS)),)

    XTENSA_PATH = $(MAKEFILE_DIR)/downloads

    ifneq (,$(filter hifi4%, $(TARGET_ARCH)))

        CCFLAGS += -DNNLIB_V2 \
                   -DXTENSA_NNLIB_MAX_SCRATCH_SIZE=70*1024

        CXXFLAGS += -DNNLIB_V2 \
                    -DXTENSA_NNLIB_MAX_SCRATCH_SIZE=70*1024

        MICROLITE_CC_SRCS += \
                             $(XTENSA_PATH)/xa_nnlib/algo/kernels/activations/hifi4/xa_nn_activations_f32_f32.c \
                             $(XTENSA_PATH)/xa_nnlib/algo/kernels/activations/hifi4/xa_nn_activations_asym8_asym8.c \
                             $(XTENSA_PATH)/xa_nnlib/algo/kernels/activations/hifi4/xa_nn_activations_32_16.c \
                             $(XTENSA_PATH)/xa_nnlib/algo/kernels/activations/hifi4/xa_nn_activations_32_8.c  \
                             $(XTENSA_PATH)/xa_nnlib/algo/kernels/activations/hifi4/xa_nn_softmax_asym8_asym8.c \
                             $(XTENSA_PATH)/xa_nnlib/algo/kernels/basic/hifi4/xa_nn_floor_f32.c \
                             $(XTENSA_PATH)/xa_nnlib/algo/kernels/cnn/hifi4/xa_nn_conv2d_std_circ_buf.c \
                             $(XTENSA_PATH)/xa_nnlib/algo/kernels/cnn/hifi4/xa_nn_conv2d_std_asym8xasym8.c \
                             $(XTENSA_PATH)/xa_nnlib/algo/kernels/cnn/hifi4/xa_nn_conv2d_std_f32.c \
                             $(XTENSA_PATH)/xa_nnlib/algo/kernels/cnn/hifi4/xa_nn_matXvec_asym8xasym8_asym8_circ.c \
                             $(XTENSA_PATH)/xa_nnlib/algo/kernels/cnn/hifi4/xa_nn_matXvec_f32_circ.c \
                             $(XTENSA_PATH)/xa_nnlib/algo/kernels/cnn/hifi4/xa_nn_conv2d_depthwise.c \
                             $(XTENSA_PATH)/xa_nnlib/algo/kernels/cnn/hifi4/xa_nn_conv2d_depthwise_f32.c \
                             $(XTENSA_PATH)/xa_nnlib/algo/kernels/cnn/hifi4/xa_nn_conv2d_depthwise_asym8xasym8.c \
                             $(XTENSA_PATH)/xa_nnlib/algo/kernels/cnn/hifi4/xa_nn_circ_buf.c \
                             $(XTENSA_PATH)/xa_nnlib/algo/kernels/fc/hifi4/xa_nn_fully_connected.c \
                             $(XTENSA_PATH)/xa_nnlib/algo/kernels/matXvec/hifi4/xa_nn_matXvec_f32.c \
                             $(XTENSA_PATH)/xa_nnlib/algo/kernels/matXvec/hifi4/xa_nn_matXvec_16x16.c \
                             $(XTENSA_PATH)/xa_nnlib/algo/kernels/matXvec/hifi4/xa_nn_matXvec_8x16.c \
                             $(XTENSA_PATH)/xa_nnlib/algo/kernels/matXvec/hifi4/xa_nn_matXvec_8x8.c \
                             $(XTENSA_PATH)/xa_nnlib/algo/kernels/matXvec/hifi4/xa_nn_matXvec_asym8xasym8.c \
                             $(XTENSA_PATH)/xa_nnlib/algo/kernels/pool/hifi4/xa_nn_avgpool.c \
                             $(XTENSA_PATH)/xa_nnlib/algo/kernels/pool/hifi4/xa_nn_avgpool_f32.c \
                             $(XTENSA_PATH)/xa_nnlib/algo/kernels/pool/hifi4/xa_nn_avgpool_asym8.c \
                             $(XTENSA_PATH)/xa_nnlib/algo/kernels/pool/hifi4/xa_nn_maxpool.c \
                             $(XTENSA_PATH)/xa_nnlib/algo/kernels/pool/hifi4/xa_nn_maxpool_f32.c \
                             $(XTENSA_PATH)/xa_nnlib/algo/kernels/pool/hifi4/xa_nn_maxpool_asym8.c \
                             $(XTENSA_PATH)/xa_nnlib/algo/kernels/pool/hifi4/xa_nn_avgpool_f32_nhwc.c \
                             $(XTENSA_PATH)/xa_nnlib/algo/kernels/pool/hifi4/xa_nn_avgpool_asym8_nhwc.c \
                             $(XTENSA_PATH)/xa_nnlib/algo/kernels/pool/hifi4/xa_nn_maxpool_f32_nhwc.c \
                             $(XTENSA_PATH)/xa_nnlib/algo/kernels/pool/hifi4/xa_nn_maxpool_asym8_nhwc.c \
                             $(XTENSA_PATH)/xa_nnlib/algo/kernels/pool/hifi4/xa_nn_inv_256_tbl.c \
                             $(XTENSA_PATH)/xa_nnlib/algo/ndsp/hifi4/src/vec_sigmoidf_hifi4.c \
                             $(XTENSA_PATH)/xa_nnlib/algo/ndsp/hifi4/src/vec_tanhf_hifi4.c \
                             $(XTENSA_PATH)/xa_nnlib/algo/ndsp/hifi4/src/vec_reluf_hifi4.c \
                             $(XTENSA_PATH)/xa_nnlib/algo/ndsp/hifi4/src/vec_softmaxf_hifi4.c \
                             $(XTENSA_PATH)/xa_nnlib/algo/ndsp/hifi4/src/vec_alognf_hifi4.c \
                             $(XTENSA_PATH)/xa_nnlib/algo/ndsp/hifi4/src/scl_sigmoidf_hifi4.c \
                             $(XTENSA_PATH)/xa_nnlib/algo/ndsp/hifi4/src/scl_tanhf_hifi4.c \
                             $(XTENSA_PATH)/xa_nnlib/algo/ndsp/hifi4/src/expf_tbl.c \
                             $(XTENSA_PATH)/xa_nnlib/algo/ndsp/hifi4/src/pow2f_tbl.c \
                             $(XTENSA_PATH)/xa_nnlib/algo/ndsp/hifi4/src/inff_tbl.c \
                             $(XTENSA_PATH)/xa_nnlib/algo/ndsp/hifi4/src/tanhf_tbl.c \
                             $(XTENSA_PATH)/xa_nnlib/algo/ndsp/hifi4/src/nanf_tbl.c \

        INCLUDES += -I$(XTENSA_PATH)/xa_nnlib/algo/kernels/ \
                    -I$(XTENSA_PATH)/xa_nnlib/include/nnlib/ \
                    -I$(XTENSA_PATH)/xa_nnlib/include/ \
                    -I$(XTENSA_PATH)/xa_nnlib/algo/common/include/ \
                    -I$(XTENSA_PATH)/xa_nnlib/algo/ndsp/hifi4/include/ \

    endif

endif
