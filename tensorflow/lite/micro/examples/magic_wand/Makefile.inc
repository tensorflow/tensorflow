ifeq ($(TARGET), sparkfun_edge)
  INCLUDES += \
    -I$(APOLLO3_SDK)/$(SF_BSPS_DEST)/common/third_party/lis2dh12/

  THIRD_PARTY_CC_SRCS += \
    $(APOLLO3_SDK)/$(SF_BSPS_DEST)/common/third_party/lis2dh12/lis2dh12_platform_apollo3.c \
    $(APOLLO3_SDK)/boards_sfe/common/third_party/lis2dh12/lis2dh12_reg.c

  THIRD_PARTY_CC_HDRS += \
    $(APOLLO3_SDK)/boards_sfe/common/third_party/lis2dh12/lis2dh12_platform_apollo3.h \
    $(APOLLO3_SDK)/boards_sfe/common/third_party/lis2dh12/lis2dh12_reg.h
endif

ACCELEROMETER_HANDLER_TEST_SRCS := \
tensorflow/lite/micro/examples/magic_wand/accelerometer_handler.cc \
tensorflow/lite/micro/examples/magic_wand/accelerometer_handler_test.cc

ACCELEROMETER_HANDLER_TEST_HDRS := \
tensorflow/lite/micro/examples/magic_wand/accelerometer_handler.h

OUTPUT_HANDLER_TEST_SRCS := \
tensorflow/lite/micro/examples/magic_wand/output_handler.cc \
tensorflow/lite/micro/examples/magic_wand/output_handler_test.cc

OUTPUT_HANDLER_TEST_HDRS := \
tensorflow/lite/micro/examples/magic_wand/output_handler.h

GESTURE_PREDICTOR_TEST_SRCS := \
tensorflow/lite/micro/examples/magic_wand/constants.cc \
tensorflow/lite/micro/examples/magic_wand/gesture_predictor.cc \
tensorflow/lite/micro/examples/magic_wand/gesture_predictor_test.cc

GESTURE_PREDICTOR_TEST_HDRS := \
tensorflow/lite/micro/examples/magic_wand/constants.h \
tensorflow/lite/micro/examples/magic_wand/gesture_predictor.h \

magic_wand_TEST_SRCS := \
tensorflow/lite/micro/examples/magic_wand/magic_wand_test.cc \
tensorflow/lite/micro/examples/magic_wand/magic_wand_model_data.cc \
tensorflow/lite/micro/examples/magic_wand/slope_micro_features_data.cc \
tensorflow/lite/micro/examples/magic_wand/ring_micro_features_data.cc

magic_wand_TEST_HDRS := \
tensorflow/lite/micro/examples/magic_wand/magic_wand_model_data.h \
tensorflow/lite/micro/examples/magic_wand/slope_micro_features_data.h \
tensorflow/lite/micro/examples/magic_wand/ring_micro_features_data.h

magic_wand_SRCS := \
tensorflow/lite/micro/examples/magic_wand/main.cc \
tensorflow/lite/micro/examples/magic_wand/main_functions.cc \
tensorflow/lite/micro/examples/magic_wand/constants.cc \
tensorflow/lite/micro/examples/magic_wand/magic_wand_model_data.cc \
tensorflow/lite/micro/examples/magic_wand/accelerometer_handler.cc \
tensorflow/lite/micro/examples/magic_wand/gesture_predictor.cc \
tensorflow/lite/micro/examples/magic_wand/output_handler.cc

magic_wand_HDRS := \
tensorflow/lite/micro/examples/magic_wand/main_functions.h \
tensorflow/lite/micro/examples/magic_wand/constants.h \
tensorflow/lite/micro/examples/magic_wand/magic_wand_model_data.h \
tensorflow/lite/micro/examples/magic_wand/accelerometer_handler.h \
tensorflow/lite/micro/examples/magic_wand/gesture_predictor.h \
tensorflow/lite/micro/examples/magic_wand/output_handler.h

#Find any platform - specific rules for this example.
include $(wildcard tensorflow/lite/micro/examples/magic_wand/*/Makefile.inc)

# Tests the accelerometer handler
$(eval $(call microlite_test,gesture_accelerometer_handler_test,\
$(ACCELEROMETER_HANDLER_TEST_SRCS),$(ACCELEROMETER_HANDLER_TEST_HDRS)))

# Tests the output handler
$(eval $(call microlite_test,gesture_output_handler_test,\
$(OUTPUT_HANDLER_TEST_SRCS),$(OUTPUT_HANDLER_TEST_HDRS)))

# Tests the gesture predictor
$(eval $(call microlite_test,gesture_predictor_test,\
$(GESTURE_PREDICTOR_TEST_SRCS),$(GESTURE_PREDICTOR_TEST_HDRS)))

# Tests loading and running the gesture recognition model
$(eval $(call microlite_test,magic_wand_test,\
$(magic_wand_TEST_SRCS),$(magic_wand_TEST_HDRS)))

# Builds a standalone binary
$(eval $(call microlite_test,magic_wand,\
$(magic_wand_SRCS),$(magic_wand_HDRS)))
