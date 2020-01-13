ifeq ($(TARGET),$(filter $(TARGET),arduino))

MICRO_SPEECH_SRCS += \
	tensorflow/lite/micro/examples/magic_wand/sparkfun_edge/accelerometer_handler.cc \
	tensorflow/lite/micro/examples/magic_wand/sparkfun_edge/output_handler.cc

endif
