ifneq ($(filter xtensa-xpg, $(ALL_TAGS)),)

    XTENSA_PATH = $(MAKEFILE_DIR)/../../kernels/xtensa_hifimini

    ifneq (,$(filter xtensa_hifimini%, $(ALL_TAGS)))

        CCFLAGS += -DHIFI_MINI_NNLIB_OPT \
                   -DDISABLE_NNLIB_UNALIGNED_SUPPORT \
                   -DXTENSA_NNLIB_MAX_SCRATCH_SIZE=1024

        CXXFLAGS += -DHIFI_MINI_NNLIB_OPT \
                   -DDISABLE_NNLIB_UNALIGNED_SUPPORT \
                   -DXTENSA_NNLIB_MAX_SCRATCH_SIZE=1024

        MICROLITE_CC_SRCS += \
                    $(XTENSA_PATH)/xa_nnlib/algo/kernels/activations/hifi_mini/xa_nn_activations_asym8s_asym8s.c \
                    $(XTENSA_PATH)/xa_nnlib/algo/kernels/activations/hifi_mini/xa_nn_softmax_asym8_asym8.c \
                    $(XTENSA_PATH)/xa_nnlib/algo/kernels/basic/hifi_mini/xa_nn_dot_prod_16x16.c \
                    $(XTENSA_PATH)/xa_nnlib/algo/kernels/fc/hifi_mini/xa_nn_fully_connected.c \
                    $(XTENSA_PATH)/xa_nnlib/algo/kernels/matXvec/hifi_mini/xa_nn_matXvec_sym8sxasym8s.c \


        INCLUDES += -I$(XTENSA_PATH)/xa_nnlib/algo/kernels/ \
                    -I$(XTENSA_PATH)/xa_nnlib/include/nnlib/ \
                    -I$(XTENSA_PATH)/xa_nnlib/include/ \
                    -I$(XTENSA_PATH)/xa_nnlib/algo/common/include/ \

    endif

endif
