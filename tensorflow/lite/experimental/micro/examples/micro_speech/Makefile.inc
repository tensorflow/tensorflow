
MICRO_SPEECH_TEST_SRCS := \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_speech_test.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/tiny_conv_model_data.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/no_features_data.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/yes_features_data.cc

MICRO_SPEECH_TEST_HDRS := \
tensorflow/lite/experimental/micro/examples/micro_speech/tiny_conv_model_data.h \
tensorflow/lite/experimental/micro/examples/micro_speech/no_features_data.h \
tensorflow/lite/experimental/micro/examples/micro_speech/yes_features_data.h \

PREPROCESSOR_TEST_SRCS := \
tensorflow/lite/experimental/micro/examples/micro_speech/preprocessor.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/preprocessor_test.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/no_30ms_sample_data.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/yes_30ms_sample_data.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/no_power_spectrum_data.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/yes_power_spectrum_data.cc

PREPROCESSOR_TEST_HDRS := \
tensorflow/lite/experimental/micro/examples/micro_speech/preprocessor.h \
tensorflow/lite/experimental/micro/examples/micro_speech/model_settings.h \
tensorflow/lite/experimental/micro/examples/micro_speech/no_30ms_sample_data.h \
tensorflow/lite/experimental/micro/examples/micro_speech/yes_30ms_sample_data.h \
tensorflow/lite/experimental/micro/examples/micro_speech/no_power_spectrum_data.h \
tensorflow/lite/experimental/micro/examples/micro_speech/yes_power_spectrum_data.h

AUDIO_PROVIDER_TEST_SRCS := \
tensorflow/lite/experimental/micro/examples/micro_speech/audio_provider_test.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/model_settings.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/audio_provider.cc

AUDIO_PROVIDER_TEST_HDRS := \
tensorflow/lite/experimental/micro/examples/micro_speech/model_settings.h \
tensorflow/lite/experimental/micro/examples/micro_speech/audio_provider.h \

FEATURE_PROVIDER_TEST_SRCS := \
tensorflow/lite/experimental/micro/examples/micro_speech/feature_provider_test.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/model_settings.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/audio_provider.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/preprocessor.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/feature_provider.cc

FEATURE_PROVIDER_TEST_HDRS := \
tensorflow/lite/experimental/micro/examples/micro_speech/model_settings.h \
tensorflow/lite/experimental/micro/examples/micro_speech/audio_provider.h \
tensorflow/lite/experimental/micro/examples/micro_speech/preprocessor.h \
tensorflow/lite/experimental/micro/examples/micro_speech/feature_provider.h

RECOGNIZE_COMMANDS_TEST_SRCS := \
tensorflow/lite/experimental/micro/examples/micro_speech/recognize_commands_test.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/model_settings.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/recognize_commands.cc

RECOGNIZE_COMMANDS_TEST_HDRS := \
tensorflow/lite/experimental/micro/examples/micro_speech/model_settings.h \
tensorflow/lite/experimental/micro/examples/micro_speech/recognize_commands.h

MICRO_SPEECH_SRCS := \
tensorflow/lite/experimental/micro/examples/micro_speech/main.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/model_settings.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/audio_provider.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/feature_provider.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/preprocessor.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/no_features_data.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/yes_features_data.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/tiny_conv_model_data.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/recognize_commands.cc

MICRO_SPEECH_HDRS := \
tensorflow/lite/experimental/micro/examples/micro_speech/model_settings.h \
tensorflow/lite/experimental/micro/examples/micro_speech/audio_provider.h \
tensorflow/lite/experimental/micro/examples/micro_speech/feature_provider.h \
tensorflow/lite/experimental/micro/examples/micro_speech/preprocessor.h \
tensorflow/lite/experimental/micro/examples/micro_speech/no_features_data.h \
tensorflow/lite/experimental/micro/examples/micro_speech/yes_features_data.h \
tensorflow/lite/experimental/micro/examples/micro_speech/tiny_conv_model_data.h \
tensorflow/lite/experimental/micro/examples/micro_speech/recognize_commands.h

# Find any platform-specific rules for this example.
include $(wildcard tensorflow/lite/experimental/micro/examples/micro_speech/*/Makefile.inc)

# Tests loading and running a speech model.
$(eval $(call microlite_test,micro_speech_test,\
$(MICRO_SPEECH_TEST_SRCS),$(MICRO_SPEECH_TEST_HDRS)))

# Test the code for feature generation.
$(eval $(call microlite_test,preprocessor_test,\
$(PREPROCESSOR_TEST_SRCS), $(PREPROCESSOR_TEST_HDRS)))

# Tests the audio provider module.
$(eval $(call microlite_test,audio_provider_test,\
$(AUDIO_PROVIDER_TEST_SRCS),$(AUDIO_PROVIDER_TEST_HDRS)))

# Tests the feature provider module.
$(eval $(call microlite_test,feature_provider_test,\
$(FEATURE_PROVIDER_TEST_SRCS),$(FEATURE_PROVIDER_TEST_HDRS)))

# Tests the feature provider module.
$(eval $(call microlite_test,recognize_commands_test,\
$(RECOGNIZE_COMMANDS_TEST_SRCS),$(RECOGNIZE_COMMANDS_TEST_HDRS)))

# Builds a standalone speech command recognizer binary.
$(eval $(call microlite_test,micro_speech,\
$(MICRO_SPEECH_SRCS),$(MICRO_SPEECH_HDRS)))
