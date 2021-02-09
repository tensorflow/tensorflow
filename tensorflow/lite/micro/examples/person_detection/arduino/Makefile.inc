ifeq ($(TARGET),$(filter $(TARGET),arduino))

person_detection_SRCS += \
	tensorflow/lite/micro/examples/person_detection/sparkfun_edge/image_provider.cc \
	tensorflow/lite/micro/examples/person_detection/sparkfun_edge/detection_responder.cc \
	tensorflow/lite/micro/examples/person_detection/himax_driver/HM01B0_debug.c \
	tensorflow/lite/micro/examples/person_detection/himax_driver/HM01B0_optimized.c \
	tensorflow/lite/micro/examples/person_detection/himax_driver/HM01B0.c

person_detection_HDRS += \
	tensorflow/lite/micro/examples/person_detection/himax_driver/HM01B0_debug.h \
	tensorflow/lite/micro/examples/person_detection/himax_driver/HM01B0_optimized.h \
	tensorflow/lite/micro/examples/person_detection/himax_driver/HM01B0_RAW8_QVGA_8bits_lsb_5fps.h \
	tensorflow/lite/micro/examples/person_detection/himax_driver/HM01B0_Walking1s_01.h \
	tensorflow/lite/micro/examples/person_detection/himax_driver/HM01B0.h \
	tensorflow/lite/micro/examples/person_detection/arduino/HM01B0_platform.h

endif
