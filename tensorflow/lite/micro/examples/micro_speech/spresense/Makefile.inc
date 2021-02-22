# Settings for Spresense platform for Hello World example
# This should be read when the EXTERNALS_TENSORFLOW_EXAMPLE_MICROSPEECH option is selected
# in Spresense configuration.

ifeq ($(TARGET), spresense)
ifeq ($(CONFIG_EXTERNALS_TENSORFLOW_EXAMPLE_MICROSPEECH),y)

SPRESENSE_MICRO_SPEECH_EXCLUDED_SRCS = \
    tensorflow/lite/micro/examples/micro_speech/main.cc \
    tensorflow/lite/micro/examples/micro_speech/audio_provider.cc \
    tensorflow/lite/micro/examples/micro_speech/command_responder.cc

SPRESENSE_MICRO_SPEECH_SRCS = \
    tensorflow/lite/micro/examples/micro_speech/spresense/src/spresense_audio_provider.cc \
    tensorflow/lite/micro/examples/micro_speech/spresense/src/spresense_command_responder.cc \
    $(filter-out $(SPRESENSE_MICRO_SPEECH_EXCLUDED_SRCS),$(MICRO_SPEECH_SRCS))

# In spresence case, those file should be included into libtensorflow-microlite.
THIRD_PARTY_CC_SRCS += $(SPRESENSE_MICRO_SPEECH_SRCS)

endif
endif
