ifeq ($(TARGET),$(filter $(TARGET),arduino))

MICRO_SPEECH_SRCS += \
	tensorflow/lite/micro/examples/person_detection/sparkfun_edge/image_provider.cc \
	tensorflow/lite/micro/examples/person_detection/sparkfun_edge/detection_responder.cc

endif
