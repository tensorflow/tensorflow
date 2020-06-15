ifneq ($(filter cmsis-nn,$(ALL_TAGS)),)
    # Enable u-arch specfic behaviours
    ifneq (,$(filter $(TARGET_ARCH), x86_64))
        # CMSIS-NN optimizations not supported
    endif

    # Setup CMSIS-NN lib and add required header files to microlite lib INCLUDE
    THIRD_PARTY_DOWNLOADS += \
      $(eval $(call add_third_party_download,$(CMSIS_URL),$(CMSIS_MD5),cmsis,patch_cmsis))

    ifneq (,$(filter $(TARGET_ARCH), cortex-m55))
      CCFLAGS += -DARM_MATH_MVEI
      CXXFLAGS += -DARM_MATH_MVEI
    endif

    CMSIS_PATH = $(MAKEFILE_DIR)/downloads/cmsis/

    # List of files generated with:
    # find tensorflow/lite/micro/tools/make/downloads/cmsis/CMSIS/NN/Source/ -iname "*.c"
    # These can't be generated at runtime because the library hasn't been
    # downloaded when the dependencies are calculated by make. See the RFC at
    # https://docs.google.com/document/d/14GRxeVEgSKgKBKAijO7oxnI49nLoTYBFQmPok-rG0cw
    # for more details on how this list was created and why it's needed.
    THIRD_PARTY_CC_SRCS +=                                              \
      $(CMSIS_PATH)/CMSIS/NN/Source/BasicMathFunctions/arm_elementwise_mul_s8.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/BasicMathFunctions/arm_elementwise_add_s8.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/FullyConnectedFunctions/arm_fully_connected_q7_opt.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/FullyConnectedFunctions/arm_fully_connected_s8.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/FullyConnectedFunctions/arm_fully_connected_q15_opt.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/FullyConnectedFunctions/arm_fully_connected_q15.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/FullyConnectedFunctions/arm_fully_connected_mat_q7_vec_q15.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/FullyConnectedFunctions/arm_fully_connected_q7.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/FullyConnectedFunctions/arm_fully_connected_mat_q7_vec_q15_opt.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/ConcatenationFunctions/arm_concatenation_s8_z.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/ConcatenationFunctions/arm_concatenation_s8_x.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/ConcatenationFunctions/arm_concatenation_s8_w.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/ConcatenationFunctions/arm_concatenation_s8_y.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/ConvolutionFunctions/arm_nn_depthwise_conv_s8_core.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/ConvolutionFunctions/arm_depthwise_conv_u8_basic_ver1.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_s8.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_1x1_s8_fast.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q7_fast.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_q7_q15_reordered.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_q7_q15.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q7_RGB.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/ConvolutionFunctions/arm_nn_mat_mult_s8.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_1x1_HWC_q7_fast_nonsquare.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q7_basic.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/ConvolutionFunctions/arm_depthwise_separable_conv_HWC_q7_nonsquare.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q15_fast_nonsquare.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/ConvolutionFunctions/arm_depthwise_conv_s8_opt.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/ConvolutionFunctions/arm_depthwise_conv_3x3_s8.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_s8_s16.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q15_basic.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q15_fast.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q7_fast_nonsquare.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_s8_s16_reordered.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/ConvolutionFunctions/arm_depthwise_separable_conv_HWC_q7.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/ConvolutionFunctions/arm_depthwise_conv_s8.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_1_x_n_s8.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q7_basic_nonsquare.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_wrapper_s8.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/PoolingFunctions/arm_pool_q7_HWC.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/PoolingFunctions/arm_max_pool_s8_opt.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/PoolingFunctions/arm_avgpool_s8.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/PoolingFunctions/arm_max_pool_s8.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/ActivationFunctions/arm_relu_q15.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/ActivationFunctions/arm_relu_q7.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/ActivationFunctions/arm_nn_activations_q7.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/ActivationFunctions/arm_relu6_s8.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/ActivationFunctions/arm_nn_activations_q15.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/NNSupportFunctions/arm_nn_add_q7.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/NNSupportFunctions/arm_nn_depthwise_conv_nt_t_padded_s8.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/NNSupportFunctions/arm_q7_to_q15_reordered_no_shift.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/NNSupportFunctions/arm_nn_depthwise_conv_nt_t_s8.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/NNSupportFunctions/arm_nntables.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/NNSupportFunctions/arm_nn_accumulate_q7_to_q15.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/NNSupportFunctions/arm_nn_mult_q7.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/NNSupportFunctions/arm_nn_mult_q15.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/NNSupportFunctions/arm_q7_to_q15_reordered_with_offset.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/NNSupportFunctions/arm_q7_to_q15_with_offset.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/NNSupportFunctions/arm_nn_mat_mul_core_4x_s8.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/NNSupportFunctions/arm_nn_mat_mult_nt_t_s8.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_s8.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/NNSupportFunctions/arm_nn_mat_mul_core_1x_s8.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/NNSupportFunctions/arm_q7_to_q15_no_shift.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/SoftmaxFunctions/arm_softmax_s8.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/SoftmaxFunctions/arm_softmax_q15.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/SoftmaxFunctions/arm_softmax_u8.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/SoftmaxFunctions/arm_softmax_q7.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/SoftmaxFunctions/arm_softmax_with_batch_q7.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/ReshapeFunctions/arm_reshape_s8.c

    # Cherry-picked list of headers that are needed to compile the CMSIS-NN
    # optimized kernels. We don't include all the possible CMSIS headers because
    # of their large number. See the RFC document for more details:
    # https://docs.google.com/document/d/14GRxeVEgSKgKBKAijO7oxnI49nLoTYBFQmPok-rG0cw
    THIRD_PARTY_CC_HDRS += \
      $(CMSIS_PATH)CMSIS/NN/Include/arm_nnfunctions.h \
      $(CMSIS_PATH)CMSIS/NN/Include/arm_nnsupportfunctions.h \
      $(CMSIS_PATH)CMSIS/NN/Include/arm_nn_tables.h \
      $(CMSIS_PATH)CMSIS/NN/Include/arm_nn_types.h \
      $(CMSIS_PATH)CMSIS/DSP/Include/arm_math.h \
      $(CMSIS_PATH)CMSIS/DSP/Include/arm_common_tables.h

endif
