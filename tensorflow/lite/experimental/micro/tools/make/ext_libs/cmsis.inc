ifneq ($(filter cmsis-nn,$(ALL_TAGS)),)
    # Enable u-arch specfic behaviours
    ifneq (,$(filter $(TARGET_ARCH), cortex-m3))
        # CMSIS-NN optimizations not supported
    endif
    ifneq (,$(filter $(TARGET_ARCH), cortex-m4))
        CCFLAGS += -DARM_MATH_DSP
        CXXFLAGS += -DARM_MATH_DSP
    endif
    ifneq (,$(filter $(TARGET_ARCH), cortex-m7))
        CCFLAGS += -DARM_MATH_DSP
        CXXFLAGS += -DARM_MATH_DSP
    endif
    ifneq (,$(filter $(TARGET_ARCH), x86_64))
        # CMSIS-NN optimizations not supported
    endif

    # Setup CMSIS-NN lib and add required header files to microlite lib INCLUDE
    THIRD_PARTY_DOWNLOADS += \
      $(eval $(call add_third_party_download,$(CMSIS_URL),$(CMSIS_MD5),cmsis,))

    CMSIS_PATH = $(MAKEFILE_DIR)/downloads/cmsis/
    # List created by running:
    # find tensorflow/lite/experimental/micro/tools/make/downloads/cmsis/CMSIS/NN/Source/ -name *.c | sed -E 's#tensorflow/lite/experimental/micro/tools/make/downloads/cmsis(.*)$#      ${CMSIS_PATH}\1 \\#g'
    THIRD_PARTY_CC_SRCS += \
      $(CMSIS_PATH)/CMSIS/NN/Source/BasicMathFunctions/arm_elementwise_mul_s8.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/BasicMathFunctions/arm_elementwise_add_s8.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/FullyConnectedFunctions/arm_fully_connected_q7_opt.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/FullyConnectedFunctions/arm_fully_connected_s8.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/FullyConnectedFunctions/arm_fully_connected_q15_opt.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/FullyConnectedFunctions/arm_fully_connected_q15.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/FullyConnectedFunctions/arm_fully_connected_mat_q7_vec_q15.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/FullyConnectedFunctions/arm_fully_connected_q7.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/FullyConnectedFunctions/arm_fully_connected_mat_q7_vec_q15_opt.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/ConvolutionFunctions/arm_depthwise_conv_u8_basic_ver1.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_s8.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_1x1_s8_fast.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q7_fast.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_q7_q15_reordered.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_q7_q15.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q7_RGB.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_1x1_HWC_q7_fast_nonsquare.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q7_basic.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/ConvolutionFunctions/arm_depthwise_separable_conv_HWC_q7_nonsquare.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q15_fast_nonsquare.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/ConvolutionFunctions/arm_depthwise_conv_s8_opt.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_s8_s16.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q15_basic.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q15_fast.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q7_fast_nonsquare.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_s8_s16_reordered.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/ConvolutionFunctions/arm_depthwise_separable_conv_HWC_q7.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/ConvolutionFunctions/arm_depthwise_conv_s8.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q7_basic_nonsquare.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/PoolingFunctions/arm_pool_q7_HWC.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/PoolingFunctions/arm_avgpool_s8.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/ActivationFunctions/arm_relu_q15.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/ActivationFunctions/arm_relu_q7.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/ActivationFunctions/arm_nn_activations_q7.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/ActivationFunctions/arm_nn_activations_q15.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/NNSupportFunctions/arm_nn_add_q7.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/NNSupportFunctions/arm_q7_to_q15_reordered_no_shift.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/NNSupportFunctions/arm_nntables.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/NNSupportFunctions/arm_nn_accumulate_q7_to_q15.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/NNSupportFunctions/arm_nn_mult_q7.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/NNSupportFunctions/arm_nn_mult_q15.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/NNSupportFunctions/arm_q7_to_q15_reordered_with_offset.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/NNSupportFunctions/arm_q7_to_q15_with_offset.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/NNSupportFunctions/arm_q7_to_q15_no_shift.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/SoftmaxFunctions/arm_softmax_q15.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/SoftmaxFunctions/arm_softmax_q7.c \
      $(CMSIS_PATH)/CMSIS/NN/Source/SoftmaxFunctions/arm_softmax_with_batch_q7.c

    # List created by running:
    # find tensorflow/lite/experimental/micro/tools/make/downloads/cmsis/CMSIS/{Core,NN,DSP}/Include -name *.h | sed -E 's#tensorflow/lite/experimental/micro/tools/make/downloads/cmsis(.*)$#      ${CMSIS_PATH}\1 \\#g'
    THIRD_PARTY_CC_HDRS += \
      ${CMSIS_PATH}/CMSIS/Core/Include/cmsis_compiler.h \
      ${CMSIS_PATH}/CMSIS/Core/Include/cmsis_armclang.h \
      ${CMSIS_PATH}/CMSIS/Core/Include/mpu_armv7.h \
      ${CMSIS_PATH}/CMSIS/Core/Include/mpu_armv8.h \
      ${CMSIS_PATH}/CMSIS/Core/Include/cmsis_gcc.h \
      ${CMSIS_PATH}/CMSIS/Core/Include/core_armv8mbl.h \
      ${CMSIS_PATH}/CMSIS/Core/Include/cmsis_version.h \
      ${CMSIS_PATH}/CMSIS/Core/Include/core_cm33.h \
      ${CMSIS_PATH}/CMSIS/Core/Include/core_cm0.h \
      ${CMSIS_PATH}/CMSIS/Core/Include/core_armv8mml.h \
      ${CMSIS_PATH}/CMSIS/Core/Include/core_cm3.h \
      ${CMSIS_PATH}/CMSIS/Core/Include/core_cm7.h \
      ${CMSIS_PATH}/CMSIS/Core/Include/cmsis_armcc.h \
      ${CMSIS_PATH}/CMSIS/Core/Include/core_cm4.h \
      ${CMSIS_PATH}/CMSIS/Core/Include/core_cm0plus.h \
      ${CMSIS_PATH}/CMSIS/Core/Include/tz_context.h \
      ${CMSIS_PATH}/CMSIS/Core/Include/core_cm23.h \
      ${CMSIS_PATH}/CMSIS/Core/Include/cmsis_iccarm.h \
      ${CMSIS_PATH}/CMSIS/Core/Include/core_sc300.h \
      ${CMSIS_PATH}/CMSIS/Core/Include/core_sc000.h \
      ${CMSIS_PATH}/CMSIS/NN/Include/arm_nnsupportfunctions.h \
      ${CMSIS_PATH}/CMSIS/NN/Include/arm_nn_tables.h \
      ${CMSIS_PATH}/CMSIS/NN/Include/arm_nnfunctions.h \
      ${CMSIS_PATH}/CMSIS/DSP/Include/arm_common_tables.h \
      ${CMSIS_PATH}/CMSIS/DSP/Include/arm_math.h \
      ${CMSIS_PATH}/CMSIS/DSP/Include/arm_const_structs.h

    INCLUDES += -I$(CMSIS_PATH)/CMSIS/Core/Include \
                -I$(CMSIS_PATH)/CMSIS/NN/Include \
                -I$(CMSIS_PATH)/CMSIS/DSP/Include
endif
