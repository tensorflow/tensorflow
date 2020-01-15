ifeq ($(TARGET),$(filter $(TARGET),apollo3evb sparkfun_edge_int8))
  person_detection_SRCS += \
  tensorflow/lite/micro/examples/person_detection/himax_driver/HM01B0.c \
  tensorflow/lite/micro/examples/person_detection/himax_driver/HM01B0_debug.c \
  tensorflow/lite/micro/examples/person_detection/himax_driver/HM01B0_optimized.c

  person_detection_HDRS += \
  tensorflow/lite/micro/examples/person_detection/himax_driver/HM01B0.h \
  tensorflow/lite/micro/examples/person_detection/himax_driver/HM01B0_debug.h \
  tensorflow/lite/micro/examples/person_detection/himax_driver/HM01B0_optimized.h \
  tensorflow/lite/micro/examples/person_detection/himax_driver/HM01B0_RAW8_QVGA_8bits_lsb_5fps.h \
  tensorflow/lite/micro/examples/person_detection/himax_driver/HM01B0_Walking1s_01.h
endif
