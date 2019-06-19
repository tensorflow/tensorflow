MICRO_RECOGNITION_HDRS := \
tensorflow/lite/experimental/micro/examples/micro_recognition/image_recognition_model.h \
tensorflow/lite/experimental/micro/examples/micro_recognition/image_provider.h \
tensorflow/lite/experimental/micro/examples/micro_recognition/disco/image_util.h \
tensorflow/lite/experimental/micro/examples/micro_recognition/disco/display_util.h \
tensorflow/lite/experimental/micro/examples/micro_recognition/util.h

MICRO_RECOGNITION_SRCS := \
tensorflow/lite/experimental/micro/examples/micro_recognition/image_recognition_model.cc \
tensorflow/lite/experimental/micro/examples/micro_recognition/main.cc \
tensorflow/lite/experimental/micro/examples/micro_recognition/disco/image_provider.cc \
tensorflow/lite/experimental/micro/examples/micro_recognition/disco/image_util.cc \
tensorflow/lite/experimental/micro/examples/micro_recognition/disco/display_util.cc

include $(wildcard tensorflow/lite/experimental/micro/examples/micro_recognition/*/Makefile.inc)


ifneq ($(filter disco_f746ng,$(ALL_TAGS)),)
  MBED_PROJECT_FILES += \
    BSP_DISCO_F746NG.lib \
    LCD_DISCO_F746NG.lib
endif

$(eval $(call microlite_test,micro_recognition,\
$(MICRO_RECOGNITION_SRCS),$(MICRO_RECOGNITION_HDRS)))
