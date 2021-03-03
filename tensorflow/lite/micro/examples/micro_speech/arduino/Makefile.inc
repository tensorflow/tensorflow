ifeq ($(TARGET),$(filter $(TARGET),arduino))

MICRO_SPEECH_SRCS += \
	tensorflow/lite/micro/examples/micro_speech/sparkfun_edge/audio_provider.cc \
	tensorflow/lite/micro/examples/micro_speech/sparkfun_edge/command_responder.cc

endif
