# Settings for targets that use the CMSIS library.
ifneq ($(filter CMSIS,$(ALL_TAGS)),)
  INCLUDES += \
    -isystem$(MAKEFILE_DIR)/downloads/cmsis/CMSIS/Core/Include/ \
    -isystem$(MAKEFILE_DIR)/downloads/cmsis/CMSIS/DSP/Include/ \
    -I$(MAKEFILE_DIR)/downloads/CMSIS_ext/

  GENERATED_PROJECT_INCLUDES += \
    -isystemthird_party/cmsis/CMSIS/Core/Include/ \
    -isystemthird_party/cmsis/CMSIS/DSP/Include/ \
    -Ithird_party/CMSIS_ext/

  CMSIS_PREPROCESSOR_SRCS := \
    tensorflow/lite/micro/examples/micro_speech/CMSIS/hanning.cc \
    tensorflow/lite/micro/examples/micro_speech/CMSIS/sin_1k.cc \

  CMSIS_PREPROCESSOR_HDRS := \
    tensorflow/lite/micro/examples/micro_speech/CMSIS/hanning.h \
    tensorflow/lite/micro/examples/micro_speech/CMSIS/sin_1k.h \
    third_party/CMSIS_ext/README.md \
    third_party/CMSIS_ext/arm_cmplx_mag_squared_q10p6.h

  PREPROCESSOR_TEST_SRCS += $(CMSIS_PREPROCESSOR_SRCS)
  PREPROCESSOR_TEST_HDRS += $(CMSIS_PREPROCESSOR_HDRS)

  FEATURE_PROVIDER_TEST_SRCS += $(CMSIS_PREPROCESSOR_SRCS)
  FEATURE_PROVIDER_TEST_HDRS += $(CMSIS_PREPROCESSOR_HDRS)

  SIMPLE_FEATURES_GENERATOR_TEST_SRCS += $(CMSIS_PREPROCESSOR_SRCS)
  SIMPLE_FEATURES_GENERATOR_TEST_HDRS += $(CMSIS_PREPROCESSOR_HDRS)

  MICRO_SPEECH_SRCS += $(CMSIS_PREPROCESSOR_SRCS)
  MICRO_SPEECH_HDRS += $(CMSIS_PREPROCESSOR_HDRS)

  THIRD_PARTY_CC_SRCS += \
    $(MAKEFILE_DIR)/downloads/CMSIS_ext/arm_cmplx_mag_squared_q10p6.c \
    $(MAKEFILE_DIR)/downloads/cmsis/CMSIS/DSP/Source/BasicMathFunctions/arm_mult_q15.c \
    $(MAKEFILE_DIR)/downloads/cmsis/CMSIS/DSP/Source/TransformFunctions/arm_bitreversal.c \
    $(MAKEFILE_DIR)/downloads/cmsis/CMSIS/DSP/Source/TransformFunctions/arm_rfft_init_q15.c \
    $(MAKEFILE_DIR)/downloads/cmsis/CMSIS/DSP/Source/TransformFunctions/arm_rfft_q15.c \
    $(MAKEFILE_DIR)/downloads/cmsis/CMSIS/DSP/Source/TransformFunctions/arm_cfft_q15.c \
    $(MAKEFILE_DIR)/downloads/cmsis/CMSIS/DSP/Source/TransformFunctions/arm_cfft_radix4_q15.c \
    $(MAKEFILE_DIR)/downloads/cmsis/CMSIS/DSP/Source/TransformFunctions/arm_bitreversal2.S \
    $(MAKEFILE_DIR)/downloads/cmsis/CMSIS/DSP/Source/CommonTables/arm_const_structs.c \
    $(MAKEFILE_DIR)/downloads/cmsis/CMSIS/DSP/Source/CommonTables/arm_common_tables.c \
    $(MAKEFILE_DIR)/downloads/cmsis/CMSIS/DSP/Source/StatisticsFunctions/arm_mean_q15.c \
    $(MAKEFILE_DIR)/downloads/cmsis/CMSIS/DSP/Source/StatisticsFunctions/arm_max_q7.c

  THIRD_PARTY_CC_HDRS += \
    third_party/cmsis/LICENSE.txt \
    third_party/cmsis/CMSIS/Core/Include/cmsis_compiler.h \
    third_party/cmsis/CMSIS/Core/Include/cmsis_gcc.h \
    third_party/cmsis/CMSIS/Core/Include/cmsis_version.h \
    third_party/cmsis/CMSIS/Core/Include/core_cm3.h \
    third_party/cmsis/CMSIS/DSP/Include/arm_common_tables.h \
    third_party/cmsis/CMSIS/DSP/Include/arm_const_structs.h \
    third_party/cmsis/CMSIS/DSP/Include/arm_math.h

endif
