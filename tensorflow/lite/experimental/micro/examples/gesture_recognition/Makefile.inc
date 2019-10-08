ifeq ($(TARGET), sparkfun_edge)
  INCLUDES += \
    -I$(APOLLO3_SDK)/boards/SparkFun_TensorFlow_Apollo3_BSP/examples/example1_edge_test/src/tf_accelerometer/ \
    -I$(APOLLO3_SDK)/boards/SparkFun_TensorFlow_Apollo3_BSP/examples/example1_edge_test/src/tf_adc/

  THIRD_PARTY_CC_SRCS += \
    $(APOLLO3_SDK)/boards/SparkFun_TensorFlow_Apollo3_BSP/examples/example1_edge_test/src/tf_accelerometer/tf_accelerometer.c \
    $(APOLLO3_SDK)/boards/SparkFun_TensorFlow_Apollo3_BSP/examples/example1_edge_test/src/tf_accelerometer/lis2dh12_reg.c \
    $(APOLLO3_SDK)/boards/SparkFun_TensorFlow_Apollo3_BSP/examples/example1_edge_test/src/tf_adc/tf_adc.c

  THIRD_PARTY_CC_HDRS += \
    $(APOLLO3_SDK)/boards/SparkFun_TensorFlow_Apollo3_BSP/examples/example1_edge_test/src/tf_accelerometer/tf_accelerometer.h \
    $(APOLLO3_SDK)/boards/SparkFun_TensorFlow_Apollo3_BSP/examples/example1_edge_test/src/tf_accelerometer/lis2dh12_reg.h \
    $(APOLLO3_SDK)/boards/SparkFun_TensorFlow_Apollo3_BSP/examples/example1_edge_test/src/tf_adc/tf_adc.h
endif

ACCELEROMETER_HANDLER_TEST_SRCS := \
tensorflow/lite/experimental/micro/examples/gesture_recognition/accelerometer_handler.cc \
tensorflow/lite/experimental/micro/examples/gesture_recognition/accelerometer_handler_test.cc

ACCELEROMETER_HANDLER_TEST_HDRS := \
tensorflow/lite/experimental/micro/examples/gesture_recognition/accelerometer_handler.h

OUTPUT_HANDLER_TEST_SRCS := \
tensorflow/lite/experimental/micro/examples/gesture_recognition/output_handler.cc \
tensorflow/lite/experimental/micro/examples/gesture_recognition/output_handler_test.cc

OUTPUT_HANDLER_TEST_HDRS := \
tensorflow/lite/experimental/micro/examples/gesture_recognition/output_handler.h

GESTURE_PREDICTOR_TEST_SRCS := \
tensorflow/lite/experimental/micro/examples/gesture_recognition/constants.cc \
tensorflow/lite/experimental/micro/examples/gesture_recognition/gesture_predictor.cc \
tensorflow/lite/experimental/micro/examples/gesture_recognition/gesture_predictor_test.cc

GESTURE_PREDICTOR_TEST_HDRS := \
tensorflow/lite/experimental/micro/examples/gesture_recognition/constants.h \
tensorflow/lite/experimental/micro/examples/gesture_recognition/gesture_predictor.h \

GESTURE_RECOGNITION_TEST_SRCS := \
tensorflow/lite/experimental/micro/examples/gesture_recognition/gesture_recognition_test.cc \
tensorflow/lite/experimental/micro/examples/gesture_recognition/gesture_recognition_model_data.cc \
tensorflow/lite/experimental/micro/examples/gesture_recognition/angle_micro_features_data.cc \
tensorflow/lite/experimental/micro/examples/gesture_recognition/circle_micro_features_data.cc

GESTURE_RECOGNITION_TEST_HDRS := \
tensorflow/lite/experimental/micro/examples/gesture_recognition/gesture_recognition_model_data.h \
tensorflow/lite/experimental/micro/examples/gesture_recognition/angle_micro_features_data.h \
tensorflow/lite/experimental/micro/examples/gesture_recognition/circle_micro_features_data.h

GESTURE_RECOGNITION_SRCS := \
tensorflow/lite/experimental/micro/examples/gesture_recognition/main.cc \
tensorflow/lite/experimental/micro/examples/gesture_recognition/main_functions.cc \
tensorflow/lite/experimental/micro/examples/gesture_recognition/constants.cc \
tensorflow/lite/experimental/micro/examples/gesture_recognition/gesture_recognition_model_data.cc \
tensorflow/lite/experimental/micro/examples/gesture_recognition/accelerometer_handler.cc \
tensorflow/lite/experimental/micro/examples/gesture_recognition/gesture_predictor.cc \
tensorflow/lite/experimental/micro/examples/gesture_recognition/output_handler.cc

GESTURE_RECOGNITION_HDRS := \
tensorflow/lite/experimental/micro/examples/gesture_recognition/main_functions.h \
tensorflow/lite/experimental/micro/examples/gesture_recognition/constants.h \
tensorflow/lite/experimental/micro/examples/gesture_recognition/gesture_recognition_model_data.h \
tensorflow/lite/experimental/micro/examples/gesture_recognition/accelerometer_handler.h \
tensorflow/lite/experimental/micro/examples/gesture_recognition/gesture_predictor.h \
tensorflow/lite/experimental/micro/examples/gesture_recognition/output_handler.h

# Find any platform-specific rules for this example.
include $(wildcard tensorflow/lite/experimental/micro/examples/gesture_recognition/*/Makefile.inc)

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
$(eval $(call microlite_test,gesture_recognition_test,\
$(GESTURE_RECOGNITION_TEST_SRCS),$(GESTURE_RECOGNITION_TEST_HDRS)))

# Builds a standalone binary
$(eval $(call microlite_test,gesture_recognition,\
$(GESTURE_RECOGNITION_SRCS),$(GESTURE_RECOGNITION_HDRS)))
