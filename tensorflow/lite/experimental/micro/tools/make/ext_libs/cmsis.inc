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
    # find tensorflow/lite/experimental/micro/tools/make/downloads/cmsis/CMSIS/{Core,NN,DSP} -name *.h | sed -E 's#tensorflow/lite/experimental/micro/tools/make/downloads/cmsis(.*)$#      ${CMSIS_PATH}\1 \\#g'
    THIRD_PARTY_CC_HDRS += \
      ${CMSIS_PATH}/CMSIS/Core/Include/core_cm3.h \
      ${CMSIS_PATH}/CMSIS/Core/Include/core_armv81mml.h \
      ${CMSIS_PATH}/CMSIS/Core/Include/cmsis_compiler.h \
      ${CMSIS_PATH}/CMSIS/Core/Include/core_cm4.h \
      ${CMSIS_PATH}/CMSIS/Core/Include/core_cm1.h \
      ${CMSIS_PATH}/CMSIS/Core/Include/cmsis_iccarm.h \
      ${CMSIS_PATH}/CMSIS/Core/Include/cmsis_version.h \
      ${CMSIS_PATH}/CMSIS/Core/Include/core_armv8mml.h \
      ${CMSIS_PATH}/CMSIS/Core/Include/core_sc300.h \
      ${CMSIS_PATH}/CMSIS/Core/Include/cmsis_armclang_ltm.h \
      ${CMSIS_PATH}/CMSIS/Core/Include/cmsis_armcc.h \
      ${CMSIS_PATH}/CMSIS/Core/Include/core_cm0plus.h \
      ${CMSIS_PATH}/CMSIS/Core/Include/core_cm33.h \
      ${CMSIS_PATH}/CMSIS/Core/Include/cmsis_armclang.h \
      ${CMSIS_PATH}/CMSIS/Core/Include/tz_context.h \
      ${CMSIS_PATH}/CMSIS/Core/Include/core_armv8mbl.h \
      ${CMSIS_PATH}/CMSIS/Core/Include/core_sc000.h \
      ${CMSIS_PATH}/CMSIS/Core/Include/core_cm23.h \
      ${CMSIS_PATH}/CMSIS/Core/Include/core_cm35p.h \
      ${CMSIS_PATH}/CMSIS/Core/Include/core_cm7.h \
      ${CMSIS_PATH}/CMSIS/Core/Include/cmsis_gcc.h \
      ${CMSIS_PATH}/CMSIS/Core/Include/mpu_armv7.h \
      ${CMSIS_PATH}/CMSIS/Core/Include/core_cm0.h \
      ${CMSIS_PATH}/CMSIS/Core/Include/mpu_armv8.h \
      ${CMSIS_PATH}/CMSIS/NN/Include/arm_nnfunctions.h \
      ${CMSIS_PATH}/CMSIS/NN/Include/arm_nnsupportfunctions.h \
      ${CMSIS_PATH}/CMSIS/NN/Include/arm_nn_tables.h \
      ${CMSIS_PATH}/CMSIS/NN/Examples/ARM/arm_nn_examples/gru/arm_nnexamples_gru_test_data.h \
      ${CMSIS_PATH}/CMSIS/NN/Examples/ARM/arm_nn_examples/gru/RTE/_ARMCM0/RTE_Components.h \
      ${CMSIS_PATH}/CMSIS/NN/Examples/ARM/arm_nn_examples/gru/RTE/_ARMCM7_SP/RTE_Components.h \
      ${CMSIS_PATH}/CMSIS/NN/Examples/ARM/arm_nn_examples/gru/RTE/_ARMCM4_FP/RTE_Components.h \
      ${CMSIS_PATH}/CMSIS/NN/Examples/ARM/arm_nn_examples/gru/RTE/Compiler/EventRecorderConf.h \
      ${CMSIS_PATH}/CMSIS/NN/Examples/ARM/arm_nn_examples/gru/RTE/_ARMCM3/RTE_Components.h \
      ${CMSIS_PATH}/CMSIS/NN/Examples/ARM/arm_nn_examples/cifar10/RTE/_ARMCM0/RTE_Components.h \
      ${CMSIS_PATH}/CMSIS/NN/Examples/ARM/arm_nn_examples/cifar10/RTE/_ARMCM7_SP/RTE_Components.h \
      ${CMSIS_PATH}/CMSIS/NN/Examples/ARM/arm_nn_examples/cifar10/RTE/_ARMCM4_FP/RTE_Components.h \
      ${CMSIS_PATH}/CMSIS/NN/Examples/ARM/arm_nn_examples/cifar10/RTE/Compiler/EventRecorderConf.h \
      ${CMSIS_PATH}/CMSIS/NN/Examples/ARM/arm_nn_examples/cifar10/RTE/_ARMCM3/RTE_Components.h \
      ${CMSIS_PATH}/CMSIS/NN/Examples/ARM/arm_nn_examples/cifar10/arm_nnexamples_cifar10_weights.h \
      ${CMSIS_PATH}/CMSIS/NN/Examples/ARM/arm_nn_examples/cifar10/arm_nnexamples_cifar10_inputs.h \
      ${CMSIS_PATH}/CMSIS/NN/Examples/ARM/arm_nn_examples/cifar10/arm_nnexamples_cifar10_parameter.h \
      ${CMSIS_PATH}/CMSIS/NN/Examples/IAR/iar_nn_examples/NN-example-gru/arm_nnexamples_gru_test_data.h \
      ${CMSIS_PATH}/CMSIS/NN/Examples/IAR/iar_nn_examples/NN-example-cifar10/arm_nnexamples_cifar10_weights.h \
      ${CMSIS_PATH}/CMSIS/NN/Examples/IAR/iar_nn_examples/NN-example-cifar10/arm_nnexamples_cifar10_inputs.h \
      ${CMSIS_PATH}/CMSIS/NN/Examples/IAR/iar_nn_examples/NN-example-cifar10/arm_nnexamples_cifar10_parameter.h \
      ${CMSIS_PATH}/CMSIS/NN/NN_Lib_Tests/nn_test/RTE/_ARMCM0/RTE_Components.h \
      ${CMSIS_PATH}/CMSIS/NN/NN_Lib_Tests/nn_test/RTE/_ARMCM7_SP/RTE_Components.h \
      ${CMSIS_PATH}/CMSIS/NN/NN_Lib_Tests/nn_test/RTE/_ARMCM4_FP/RTE_Components.h \
      ${CMSIS_PATH}/CMSIS/NN/NN_Lib_Tests/nn_test/RTE/_ARMCM3/RTE_Components.h \
      ${CMSIS_PATH}/CMSIS/NN/NN_Lib_Tests/nn_test/arm_nnexamples_nn_test.h \
      ${CMSIS_PATH}/CMSIS/NN/NN_Lib_Tests/nn_test/Ref_Implementations/ref_functions.h \
      ${CMSIS_PATH}/CMSIS/NN/NN_Lib_Tests/nn_test/Ref_Implementations/fully_connected_testing_weights.h \
      ${CMSIS_PATH}/CMSIS/DSP/PythonWrapper/cmsisdsp_pkg/src/cmsismodule.h \
      ${CMSIS_PATH}/CMSIS/DSP/Include/arm_common_tables.h \
      ${CMSIS_PATH}/CMSIS/DSP/Include/arm_const_structs.h \
      ${CMSIS_PATH}/CMSIS/DSP/Include/arm_math.h \
      ${CMSIS_PATH}/CMSIS/DSP/DSP_Lib_TestSuite/DspLibTest_FVP_A5/RTE/RTE_Components.h \
      ${CMSIS_PATH}/CMSIS/DSP/DSP_Lib_TestSuite/DspLibTest_FVP_A5/RTE/Device/ARMCA5/mem_ARMCA5.h \
      ${CMSIS_PATH}/CMSIS/DSP/DSP_Lib_TestSuite/DspLibTest_FVP_A5/RTE/Device/ARMCA5/system_ARMCA5.h \
      ${CMSIS_PATH}/CMSIS/DSP/DSP_Lib_TestSuite/DspLibTest_FVP_A5/RTE/CMSIS/RTX_Config.h \
      ${CMSIS_PATH}/CMSIS/DSP/DSP_Lib_TestSuite/Common/inc/all_tests.h \
      ${CMSIS_PATH}/CMSIS/DSP/DSP_Lib_TestSuite/Common/inc/templates/template.h \
      ${CMSIS_PATH}/CMSIS/DSP/DSP_Lib_TestSuite/Common/inc/templates/test_templates.h \
      ${CMSIS_PATH}/CMSIS/DSP/DSP_Lib_TestSuite/Common/inc/type_abbrev.h \
      ${CMSIS_PATH}/CMSIS/DSP/DSP_Lib_TestSuite/Common/inc/statistics_tests/statistics_templates.h \
      ${CMSIS_PATH}/CMSIS/DSP/DSP_Lib_TestSuite/Common/inc/statistics_tests/statistics_test_group.h \
      ${CMSIS_PATH}/CMSIS/DSP/DSP_Lib_TestSuite/Common/inc/statistics_tests/statistics_tests.h \
      ${CMSIS_PATH}/CMSIS/DSP/DSP_Lib_TestSuite/Common/inc/statistics_tests/statistics_test_data.h \
      ${CMSIS_PATH}/CMSIS/DSP/DSP_Lib_TestSuite/Common/inc/filtering_tests/filtering_templates.h \
      ${CMSIS_PATH}/CMSIS/DSP/DSP_Lib_TestSuite/Common/inc/filtering_tests/filtering_test_group.h \
      ${CMSIS_PATH}/CMSIS/DSP/DSP_Lib_TestSuite/Common/inc/filtering_tests/filtering_tests.h \
      ${CMSIS_PATH}/CMSIS/DSP/DSP_Lib_TestSuite/Common/inc/filtering_tests/filtering_test_data.h \
      ${CMSIS_PATH}/CMSIS/DSP/DSP_Lib_TestSuite/Common/inc/math_helper.h \
      ${CMSIS_PATH}/CMSIS/DSP/DSP_Lib_TestSuite/Common/inc/fast_math_tests/fast_math_test_data.h \
      ${CMSIS_PATH}/CMSIS/DSP/DSP_Lib_TestSuite/Common/inc/fast_math_tests/fast_math_test_group.h \
      ${CMSIS_PATH}/CMSIS/DSP/DSP_Lib_TestSuite/Common/inc/fast_math_tests/fast_math_templates.h \
      ${CMSIS_PATH}/CMSIS/DSP/DSP_Lib_TestSuite/Common/inc/transform_tests/transform_templates.h \
      ${CMSIS_PATH}/CMSIS/DSP/DSP_Lib_TestSuite/Common/inc/transform_tests/transform_test_data.h \
      ${CMSIS_PATH}/CMSIS/DSP/DSP_Lib_TestSuite/Common/inc/transform_tests/transform_test_group.h \
      ${CMSIS_PATH}/CMSIS/DSP/DSP_Lib_TestSuite/Common/inc/transform_tests/transform_tests.h \
      ${CMSIS_PATH}/CMSIS/DSP/DSP_Lib_TestSuite/Common/inc/intrinsics_tests/intrinsics_test_data.h \
      ${CMSIS_PATH}/CMSIS/DSP/DSP_Lib_TestSuite/Common/inc/intrinsics_tests/intrinsics_templates.h \
      ${CMSIS_PATH}/CMSIS/DSP/DSP_Lib_TestSuite/Common/inc/intrinsics_tests/intrinsics_test_group.h \
      ${CMSIS_PATH}/CMSIS/DSP/DSP_Lib_TestSuite/Common/inc/basic_math_tests/basic_math_test_group.h \
      ${CMSIS_PATH}/CMSIS/DSP/DSP_Lib_TestSuite/Common/inc/basic_math_tests/basic_math_templates.h \
      ${CMSIS_PATH}/CMSIS/DSP/DSP_Lib_TestSuite/Common/inc/basic_math_tests/basic_math_test_data.h \
      ${CMSIS_PATH}/CMSIS/DSP/DSP_Lib_TestSuite/Common/inc/basic_math_tests/basic_math_tests.h \
      ${CMSIS_PATH}/CMSIS/DSP/DSP_Lib_TestSuite/Common/inc/complex_math_tests/complex_math_test_group.h \
      ${CMSIS_PATH}/CMSIS/DSP/DSP_Lib_TestSuite/Common/inc/complex_math_tests/complex_math_tests.h \
      ${CMSIS_PATH}/CMSIS/DSP/DSP_Lib_TestSuite/Common/inc/complex_math_tests/complex_math_test_data.h \
      ${CMSIS_PATH}/CMSIS/DSP/DSP_Lib_TestSuite/Common/inc/complex_math_tests/complex_math_templates.h \
      ${CMSIS_PATH}/CMSIS/DSP/DSP_Lib_TestSuite/Common/inc/matrix_tests/matrix_templates.h \
      ${CMSIS_PATH}/CMSIS/DSP/DSP_Lib_TestSuite/Common/inc/matrix_tests/matrix_tests.h \
      ${CMSIS_PATH}/CMSIS/DSP/DSP_Lib_TestSuite/Common/inc/matrix_tests/matrix_test_group.h \
      ${CMSIS_PATH}/CMSIS/DSP/DSP_Lib_TestSuite/Common/inc/matrix_tests/matrix_test_data.h \
      ${CMSIS_PATH}/CMSIS/DSP/DSP_Lib_TestSuite/Common/inc/controller_tests/controller_tests.h \
      ${CMSIS_PATH}/CMSIS/DSP/DSP_Lib_TestSuite/Common/inc/controller_tests/controller_test_group.h \
      ${CMSIS_PATH}/CMSIS/DSP/DSP_Lib_TestSuite/Common/inc/controller_tests/controller_test_data.h \
      ${CMSIS_PATH}/CMSIS/DSP/DSP_Lib_TestSuite/Common/inc/controller_tests/controller_templates.h \
      ${CMSIS_PATH}/CMSIS/DSP/DSP_Lib_TestSuite/Common/inc/support_tests/support_test_data.h \
      ${CMSIS_PATH}/CMSIS/DSP/DSP_Lib_TestSuite/Common/inc/support_tests/support_tests.h \
      ${CMSIS_PATH}/CMSIS/DSP/DSP_Lib_TestSuite/Common/inc/support_tests/support_templates.h \
      ${CMSIS_PATH}/CMSIS/DSP/DSP_Lib_TestSuite/Common/inc/support_tests/support_test_group.h \
      ${CMSIS_PATH}/CMSIS/DSP/DSP_Lib_TestSuite/Common/JTest/inc/jtest_fw.h \
      ${CMSIS_PATH}/CMSIS/DSP/DSP_Lib_TestSuite/Common/JTest/inc/util/util.h \
      ${CMSIS_PATH}/CMSIS/DSP/DSP_Lib_TestSuite/Common/JTest/inc/jtest_test_define.h \
      ${CMSIS_PATH}/CMSIS/DSP/DSP_Lib_TestSuite/Common/JTest/inc/jtest_define.h \
      ${CMSIS_PATH}/CMSIS/DSP/DSP_Lib_TestSuite/Common/JTest/inc/jtest_test_call.h \
      ${CMSIS_PATH}/CMSIS/DSP/DSP_Lib_TestSuite/Common/JTest/inc/jtest_test_ret.h \
      ${CMSIS_PATH}/CMSIS/DSP/DSP_Lib_TestSuite/Common/JTest/inc/jtest_group_call.h \
      ${CMSIS_PATH}/CMSIS/DSP/DSP_Lib_TestSuite/Common/JTest/inc/jtest_group_define.h \
      ${CMSIS_PATH}/CMSIS/DSP/DSP_Lib_TestSuite/Common/JTest/inc/jtest_cycle.h \
      ${CMSIS_PATH}/CMSIS/DSP/DSP_Lib_TestSuite/Common/JTest/inc/jtest_util.h \
      ${CMSIS_PATH}/CMSIS/DSP/DSP_Lib_TestSuite/Common/JTest/inc/jtest_systick.h \
      ${CMSIS_PATH}/CMSIS/DSP/DSP_Lib_TestSuite/Common/JTest/inc/jtest.h \
      ${CMSIS_PATH}/CMSIS/DSP/DSP_Lib_TestSuite/Common/JTest/inc/arr_desc/arr_desc.h \
      ${CMSIS_PATH}/CMSIS/DSP/DSP_Lib_TestSuite/Common/JTest/inc/jtest_test.h \
      ${CMSIS_PATH}/CMSIS/DSP/DSP_Lib_TestSuite/Common/JTest/inc/jtest_group.h \
      ${CMSIS_PATH}/CMSIS/DSP/DSP_Lib_TestSuite/Common/JTest/inc/jtest_pf.h \
      ${CMSIS_PATH}/CMSIS/DSP/DSP_Lib_TestSuite/Common/JTest/inc/opt_arg/pp_narg.h \
      ${CMSIS_PATH}/CMSIS/DSP/DSP_Lib_TestSuite/Common/JTest/inc/opt_arg/splice.h \
      ${CMSIS_PATH}/CMSIS/DSP/DSP_Lib_TestSuite/Common/JTest/inc/opt_arg/opt_arg.h \
      ${CMSIS_PATH}/CMSIS/DSP/DSP_Lib_TestSuite/RefLibs/inc/ref.h \
      ${CMSIS_PATH}/CMSIS/DSP/Platforms/FVP/ARMCA5/Include/ARMCA5.h \
      ${CMSIS_PATH}/CMSIS/DSP/Platforms/FVP/ARMCA5/LinkScripts/AC6/mem_ARMCA5.h \
      ${CMSIS_PATH}/CMSIS/DSP/Platforms/FVP/ARMCA5/LinkScripts/AC6/system_ARMCA5.h \
      ${CMSIS_PATH}/CMSIS/DSP/Platforms/FVP/ARMCA5/LinkScripts/GCC/mem_ARMCA5.h \
      ${CMSIS_PATH}/CMSIS/DSP/Platforms/FVP/ARMCA5/LinkScripts/GCC/system_ARMCA5.h \
      ${CMSIS_PATH}/CMSIS/DSP/Platforms/FVP/ARMCM4/Include/ARMCM4_FP.h \
      ${CMSIS_PATH}/CMSIS/DSP/Platforms/FVP/ARMCM4/Include/system_ARMCM4.h \
      ${CMSIS_PATH}/CMSIS/DSP/Platforms/FVP/ARMCM4/Include/ARMCM4.h \
      ${CMSIS_PATH}/CMSIS/DSP/Platforms/FVP/ARMCM4/LinkScripts/AC6/mem_ARMCM4.h \
      ${CMSIS_PATH}/CMSIS/DSP/Platforms/FVP/ARMCM33/Include/ARMCM33_DSP_FP.h \
      ${CMSIS_PATH}/CMSIS/DSP/Platforms/FVP/ARMCM33/Include/ARMCM33_TZ.h \
      ${CMSIS_PATH}/CMSIS/DSP/Platforms/FVP/ARMCM33/Include/system_ARMCM33.h \
      ${CMSIS_PATH}/CMSIS/DSP/Platforms/FVP/ARMCM33/Include/ARMCM33_DSP_FP_TZ.h \
      ${CMSIS_PATH}/CMSIS/DSP/Platforms/FVP/ARMCM33/Include/ARMCM33.h \
      ${CMSIS_PATH}/CMSIS/DSP/Platforms/FVP/ARMCM33/LinkScripts/AC6/mem_ARMCM33.h \
      ${CMSIS_PATH}/CMSIS/DSP/Platforms/FVP/ARMCM7/Include/ARMCM7.h \
      ${CMSIS_PATH}/CMSIS/DSP/Platforms/FVP/ARMCM7/Include/system_ARMCM7.h \
      ${CMSIS_PATH}/CMSIS/DSP/Platforms/FVP/ARMCM7/Include/ARMCM7_DP.h \
      ${CMSIS_PATH}/CMSIS/DSP/Platforms/FVP/ARMCM7/Include/ARMCM7_SP.h \
      ${CMSIS_PATH}/CMSIS/DSP/Platforms/FVP/ARMCM7/LinkScripts/AC6/mem_ARMCM7.h \
      ${CMSIS_PATH}/CMSIS/DSP/Platforms/FVP/ARMCM7/LinkScripts/GCC/mem_ARMCM7.h \
      ${CMSIS_PATH}/CMSIS/DSP/Platforms/FVP/ARMCM0/Include/ARMCM0.h \
      ${CMSIS_PATH}/CMSIS/DSP/Platforms/FVP/ARMCM0/Include/system_ARMCM0.h \
      ${CMSIS_PATH}/CMSIS/DSP/Platforms/FVP/ARMCM0/LinkScripts/AC6/mem_ARMCM0.h \
      ${CMSIS_PATH}/CMSIS/DSP/Examples/ARM/arm_matrix_example/math_helper.h \
      ${CMSIS_PATH}/CMSIS/DSP/Examples/ARM/arm_fir_example/math_helper.h \
      ${CMSIS_PATH}/CMSIS/DSP/Examples/ARM/arm_convolution_example/math_helper.h \
      ${CMSIS_PATH}/CMSIS/DSP/Examples/ARM/arm_graphic_equalizer_example/math_helper.h \
      ${CMSIS_PATH}/CMSIS/DSP/Examples/ARM/arm_linear_interp_example/math_helper.h \
      ${CMSIS_PATH}/CMSIS/DSP/Examples/ARM/arm_signal_converge_example/math_helper.h \
      ${CMSIS_PATH}/CMSIS/DSP/Source/DistanceFunctions/arm_boolean_distance_template.h \
      ${CMSIS_PATH}/CMSIS/DSP/Testing/RTE_Components.h \
      ${CMSIS_PATH}/CMSIS/DSP/Testing/Include/Benchmarks/FIRF32.h \
      ${CMSIS_PATH}/CMSIS/DSP/Testing/Include/Benchmarks/BasicMathsBenchmarksF32.h \
      ${CMSIS_PATH}/CMSIS/DSP/Testing/Include/Benchmarks/PoolingBench.h \
      ${CMSIS_PATH}/CMSIS/DSP/Testing/Include/Benchmarks/ComplexMathsBenchmarksQ15.h \
      ${CMSIS_PATH}/CMSIS/DSP/Testing/Include/Benchmarks/BasicMathsBenchmarksQ15.h \
      ${CMSIS_PATH}/CMSIS/DSP/Testing/Include/Benchmarks/ComplexMathsBenchmarksF32.h \
      ${CMSIS_PATH}/CMSIS/DSP/Testing/Include/Benchmarks/MISCQ7.h \
      ${CMSIS_PATH}/CMSIS/DSP/Testing/Include/Benchmarks/BasicMathsBenchmarksQ7.h \
      ${CMSIS_PATH}/CMSIS/DSP/Testing/Include/Benchmarks/BasicMathsBenchmarksQ31.h \
      ${CMSIS_PATH}/CMSIS/DSP/Testing/Include/Benchmarks/FIRQ31.h \
      ${CMSIS_PATH}/CMSIS/DSP/Testing/Include/Benchmarks/MISCF32.h \
      ${CMSIS_PATH}/CMSIS/DSP/Testing/Include/Benchmarks/FullyConnectedBench.h \
      ${CMSIS_PATH}/CMSIS/DSP/Testing/Include/Benchmarks/DECIMQ31.h \
      ${CMSIS_PATH}/CMSIS/DSP/Testing/Include/Benchmarks/MISCQ15.h \
      ${CMSIS_PATH}/CMSIS/DSP/Testing/Include/Benchmarks/DECIMQ15.h \
      ${CMSIS_PATH}/CMSIS/DSP/Testing/Include/Benchmarks/DECIMF32.h \
      ${CMSIS_PATH}/CMSIS/DSP/Testing/Include/Benchmarks/BIQUADF64.h \
      ${CMSIS_PATH}/CMSIS/DSP/Testing/Include/Benchmarks/BIQUADF32.h \
      ${CMSIS_PATH}/CMSIS/DSP/Testing/Include/Benchmarks/FIRQ15.h \
      ${CMSIS_PATH}/CMSIS/DSP/Testing/Include/Benchmarks/ComplexMathsBenchmarksQ31.h \
      ${CMSIS_PATH}/CMSIS/DSP/Testing/Include/Benchmarks/MISCQ31.h \
      ${CMSIS_PATH}/CMSIS/DSP/Testing/Include/Tests/StatsTestsF32.h \
      ${CMSIS_PATH}/CMSIS/DSP/Testing/Include/Tests/Softmax.h \
      ${CMSIS_PATH}/CMSIS/DSP/Testing/Include/Tests/Pooling.h \
      ${CMSIS_PATH}/CMSIS/DSP/Testing/Include/Tests/NNSupport.h \
      ${CMSIS_PATH}/CMSIS/DSP/Testing/Include/Tests/SVMF32.h \
      ${CMSIS_PATH}/CMSIS/DSP/Testing/Include/Tests/FullyConnected.h \
      ${CMSIS_PATH}/CMSIS/DSP/Testing/Include/Tests/DistanceTestsF32.h \
      ${CMSIS_PATH}/CMSIS/DSP/Testing/Include/Tests/DistanceTestsU32.h \
      ${CMSIS_PATH}/CMSIS/DSP/Testing/Include/Tests/SupportTestsF32.h \
      ${CMSIS_PATH}/CMSIS/DSP/Testing/Include/Tests/BayesF32.h \
      ${CMSIS_PATH}/CMSIS/DSP/Testing/Include/Tests/BasicTestsF32.h \
      ${CMSIS_PATH}/CMSIS/DSP/Testing/FrameworkInclude/Test.h \
      ${CMSIS_PATH}/CMSIS/DSP/Testing/FrameworkInclude/Semihosting.h \
      ${CMSIS_PATH}/CMSIS/DSP/Testing/FrameworkInclude/Pattern.h \
      ${CMSIS_PATH}/CMSIS/DSP/Testing/FrameworkInclude/Calibrate.h \
      ${CMSIS_PATH}/CMSIS/DSP/Testing/FrameworkInclude/ArrayMemory.h \
      ${CMSIS_PATH}/CMSIS/DSP/Testing/FrameworkInclude/IORunner.h \
      ${CMSIS_PATH}/CMSIS/DSP/Testing/FrameworkInclude/Error.h \
      ${CMSIS_PATH}/CMSIS/DSP/Testing/FrameworkInclude/Generators.h \
      ${CMSIS_PATH}/CMSIS/DSP/Testing/FrameworkInclude/Timing.h \
      ${CMSIS_PATH}/CMSIS/DSP/Testing/FrameworkInclude/FPGA.h \
      ${CMSIS_PATH}/CMSIS/DSP/ComputeLibrary/Include/NEMath.h

    INCLUDES += -I$(CMSIS_PATH)/CMSIS/Core/Include \
                -I$(CMSIS_PATH)/CMSIS/NN/Include \
                -I$(CMSIS_PATH)/CMSIS/DSP/Include
endif
