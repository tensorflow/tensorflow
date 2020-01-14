ifeq ($(TARGET),$(filter $(TARGET),arduino))

person_detection_SRCS += \
	tensorflow/lite/micro/examples/person_detection/sparkfun_edge/image_provider.cc \
	tensorflow/lite/micro/examples/person_detection/sparkfun_edge/detection_responder.cc

endif
