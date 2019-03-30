
INCLUDES += \
 -I$(MAKEFILE_DIR)/downloads/kissfft

GENERATED_PROJECT_INCLUDES += \
-I./third_party/kissfft

PROJECT_INCLUDES += \
third_party/kissfft

KISSFFT_LIB_SRCS := \
$(MAKEFILE_DIR)/downloads/kissfft/kiss_fft.c \
$(MAKEFILE_DIR)/downloads/kissfft/tools/kiss_fftr.c

KISSFFT_LIB_HDRS := \
$(MAKEFILE_DIR)/downloads/kissfft/COPYING \
$(MAKEFILE_DIR)/downloads/kissfft/kiss_fft.h \
$(MAKEFILE_DIR)/downloads/kissfft/_kiss_fft_guts.h \
$(MAKEFILE_DIR)/downloads/kissfft/tools/kiss_fftr.h

THIRD_PARTY_CC_HDRS += \
third_party/kissfft/COPYING \
third_party/kissfft/kiss_fft.h \
third_party/kissfft/_kiss_fft_guts.h \
third_party/kissfft/tools/kiss_fftr.h

MICRO_SPEECH_TEST_SRCS := \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_speech_test.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/tiny_conv_micro_features_model_data.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/no_micro_features_data.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/yes_micro_features_data.cc

MICRO_SPEECH_TEST_HDRS := \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/tiny_conv_micro_features_model_data.h \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/no_micro_features_data.h \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/yes_micro_features_data.h \

SIMPLE_FEATURES_GENERATOR_TEST_SRCS := \
tensorflow/lite/experimental/micro/examples/micro_speech/simple_features/simple_features_generator.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/simple_features/simple_features_generator_test.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/no_30ms_sample_data.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/yes_30ms_sample_data.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/simple_features/no_power_spectrum_data.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/simple_features/yes_power_spectrum_data.cc

SIMPLE_FEATURES_GENERATOR_TEST_HDRS := \
tensorflow/lite/experimental/micro/examples/micro_speech/simple_features/simple_model_settings.h \
tensorflow/lite/experimental/micro/examples/micro_speech/simple_features/simple_features_generator.h \
tensorflow/lite/experimental/micro/examples/micro_speech/no_30ms_sample_data.h \
tensorflow/lite/experimental/micro/examples/micro_speech/yes_30ms_sample_data.h \
tensorflow/lite/experimental/micro/examples/micro_speech/simple_features/no_power_spectrum_data.h \
tensorflow/lite/experimental/micro/examples/micro_speech/simple_features/yes_power_spectrum_data.h

MICRO_FEATURES_LIB_SRCS := \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/fft.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/fft_util.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/filterbank.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/filterbank_util.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/frontend.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/frontend_util.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/log_lut.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/log_scale.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/log_scale_util.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/noise_reduction.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/noise_reduction_util.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/pcan_gain_control.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/pcan_gain_control_util.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/window.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/window_util.cc \
$(KISSFFT_LIB_SRCS)

MICRO_FEATURES_LIB_HDRS := \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/bits.h \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/fft.h \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/fft_util.h \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/filterbank.h \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/filterbank_util.h \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/frontend.h \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/frontend_util.h \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/log_lut.h \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/log_scale.h \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/log_scale_util.h \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/noise_reduction.h \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/noise_reduction_util.h \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/no_micro_features_data.h \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/pcan_gain_control.h \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/pcan_gain_control_util.h \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/static_alloc.h \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/window.h \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/window_util.h \
$(KISSFFT_LIB_HDRS)

MICRO_FEATURES_FFT_TEST_SRCS := \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/fft_test.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/fft.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/fft_util.cc \
$(KISSFFT_LIB_SRCS)

MICRO_FEATURES_FFT_TEST_HDRS := \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/fft.h \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/fft_util.h \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/static_alloc.h \
$(KISSFFT_LIB_HDRS)

MICRO_FEATURES_FILTERBANK_TEST_SRCS := \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/filterbank_test.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/filterbank.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/filterbank_util.cc

MICRO_FEATURES_FILTERBANK_TEST_HDRS := \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/filterbank.h \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/filterbank_util.h \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/static_alloc.h

MICRO_FEATURES_FRONTEND_TEST_SRCS := \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/frontend_test.cc \
$(MICRO_FEATURES_LIB_SRCS)

MICRO_FEATURES_FRONTEND_TEST_HDRS := \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/frontend.h \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/frontend_util.h \
$(MICRO_FEATURES_LIB_HDRS)

MICRO_FEATURES_LOG_SCALE_TEST_SRCS := \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/log_scale_test.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/log_lut.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/log_scale.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/log_scale_util.cc

MICRO_FEATURES_LOG_SCALE_TEST_HDRS := \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/log_lut.h \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/log_scale.h \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/log_scale_util.h \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/bits.h \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/static_alloc.h

MICRO_FEATURES_NOISE_REDUCTION_TEST_SRCS := \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/noise_reduction_test.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/noise_reduction.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/noise_reduction_util.cc

MICRO_FEATURES_NOISE_REDUCTION_TEST_HDRS := \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/noise_reduction.h \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/noise_reduction_util.h \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/static_alloc.h

MICRO_FEATURES_PCAN_GAIN_CONTROL_TEST_SRCS := \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/pcan_gain_control_test.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/pcan_gain_control.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/pcan_gain_control_util.cc

MICRO_FEATURES_PCAN_GAIN_CONTROL_TEST_HDRS := \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/pcan_gain_control.h \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/pcan_gain_control_util.h \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/static_alloc.h

MICRO_FEATURES_WINDOW_TEST_SRCS := \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/window_test.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/window.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/window_util.cc

MICRO_FEATURES_WINDOW_TEST_HDRS := \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/window.h \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/window_util.h \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/static_alloc.h

MICRO_FEATURES_GENERATOR_SRCS := \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/micro_features_generator.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/micro_model_settings.cc \
$(MICRO_FEATURES_LIB_SRCS)

MICRO_FEATURES_GENERATOR_HDRS := \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/micro_features_generator.h \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/micro_model_settings.h \
$(MICRO_FEATURES_LIB_HDRS)

MICRO_FEATURES_GENERATOR_TEST_SRCS := \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/micro_features_generator_test.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/no_feature_data_slice.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/yes_feature_data_slice.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/no_30ms_sample_data.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/yes_30ms_sample_data.cc \
$(MICRO_FEATURES_GENERATOR_SRCS)

MICRO_FEATURES_GENERATOR_TEST_HDRS := \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/no_feature_data_slice.h \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/yes_feature_data_slice.h \
tensorflow/lite/experimental/micro/examples/micro_speech/no_30ms_sample_data.h \
tensorflow/lite/experimental/micro/examples/micro_speech/yes_30ms_sample_data.h \
$(MICRO_FEATURES_GENERATOR_HDRS)

AUDIO_PROVIDER_TEST_SRCS := \
tensorflow/lite/experimental/micro/examples/micro_speech/audio_provider_test.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/micro_model_settings.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/audio_provider.cc

AUDIO_PROVIDER_TEST_HDRS := \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/micro_model_settings.h \
tensorflow/lite/experimental/micro/examples/micro_speech/audio_provider.h \

AUDIO_PROVIDER_MOCK_TEST_SRCS := \
tensorflow/lite/experimental/micro/examples/micro_speech/audio_provider_mock_test.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/micro_model_settings.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/no_1000ms_sample_data.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/yes_1000ms_sample_data.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/audio_provider_mock.cc

AUDIO_PROVIDER_MOCK_TEST_HDRS := \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/micro_model_settings.h \
tensorflow/lite/experimental/micro/examples/micro_speech/no_1000ms_sample_data.h \
tensorflow/lite/experimental/micro/examples/micro_speech/yes_1000ms_sample_data.h \
tensorflow/lite/experimental/micro/examples/micro_speech/audio_provider.h \

FEATURE_PROVIDER_TEST_SRCS := \
tensorflow/lite/experimental/micro/examples/micro_speech/feature_provider_test.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/audio_provider.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/feature_provider.cc \
$(MICRO_FEATURES_GENERATOR_SRCS)

FEATURE_PROVIDER_TEST_HDRS := \
tensorflow/lite/experimental/micro/examples/micro_speech/audio_provider.h \
tensorflow/lite/experimental/micro/examples/micro_speech/feature_provider.h \
$(MICRO_FEATURES_GENERATOR_HDRS)

FEATURE_PROVIDER_MOCK_TEST_SRCS := \
tensorflow/lite/experimental/micro/examples/micro_speech/feature_provider_test.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/audio_provider_mock.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/no_1000ms_sample_data.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/yes_1000ms_sample_data.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/feature_provider.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/no_micro_features_data.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/yes_micro_features_data.cc \
$(MICRO_FEATURES_GENERATOR_SRCS)

FEATURE_PROVIDER_MOCK_TEST_HDRS := \
tensorflow/lite/experimental/micro/examples/micro_speech/audio_provider.h \
tensorflow/lite/experimental/micro/examples/micro_speech/no_1000ms_sample_data.h \
tensorflow/lite/experimental/micro/examples/micro_speech/yes_1000ms_sample_data.h \
tensorflow/lite/experimental/micro/examples/micro_speech/feature_provider.h \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/no_micro_features_data.h \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/yes_micro_features_data.h \
$(MICRO_FEATURES_GENERATOR_HDRS)

RECOGNIZE_COMMANDS_TEST_SRCS := \
tensorflow/lite/experimental/micro/examples/micro_speech/recognize_commands_test.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/micro_model_settings.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/recognize_commands.cc

RECOGNIZE_COMMANDS_TEST_HDRS := \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/micro_model_settings.h \
tensorflow/lite/experimental/micro/examples/micro_speech/recognize_commands.h

COMMAND_RESPONDER_TEST_SRCS := \
tensorflow/lite/experimental/micro/examples/micro_speech/command_responder_test.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/command_responder.cc

COMMAND_RESPONDER_TEST_HDRS := \
tensorflow/lite/experimental/micro/examples/micro_speech/command_responder.h

MICRO_SPEECH_SRCS := \
tensorflow/lite/experimental/micro/examples/micro_speech/main.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/audio_provider.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/feature_provider.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/no_micro_features_data.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/yes_micro_features_data.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/tiny_conv_micro_features_model_data.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/recognize_commands.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/command_responder.cc \
$(MICRO_FEATURES_GENERATOR_SRCS)

MICRO_SPEECH_HDRS := \
tensorflow/lite/experimental/micro/examples/micro_speech/audio_provider.h \
tensorflow/lite/experimental/micro/examples/micro_speech/feature_provider.h \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/no_micro_features_data.h \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/yes_micro_features_data.h \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/tiny_conv_micro_features_model_data.h \
tensorflow/lite/experimental/micro/examples/micro_speech/recognize_commands.h \
tensorflow/lite/experimental/micro/examples/micro_speech/command_responder.h \
$(MICRO_FEATURES_GENERATOR_HDRS)

MICRO_SPEECH_MOCK_SRCS := \
tensorflow/lite/experimental/micro/examples/micro_speech/main.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/audio_provider_mock.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/no_1000ms_sample_data.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/yes_1000ms_sample_data.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/feature_provider.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/no_micro_features_data.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/yes_micro_features_data.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/tiny_conv_micro_features_model_data.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/recognize_commands.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/command_responder.cc \
$(MICRO_FEATURES_GENERATOR_SRCS)

MICRO_SPEECH_MOCK_HDRS := \
tensorflow/lite/experimental/micro/examples/micro_speech/audio_provider.h \
tensorflow/lite/experimental/micro/examples/micro_speech/no_1000ms_sample_data.h \
tensorflow/lite/experimental/micro/examples/micro_speech/yes_1000ms_sample_data.h \
tensorflow/lite/experimental/micro/examples/micro_speech/feature_provider.h \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/micro_features_generator.h \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/no_micro_features_data.h \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/yes_micro_features_data.h \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/tiny_conv_micro_features_model_data.h \
tensorflow/lite/experimental/micro/examples/micro_speech/recognize_commands.h \
tensorflow/lite/experimental/micro/examples/micro_speech/command_responder.h \
$(MICRO_FEATURES_GENERATOR_HDRS)

# Find any platform-specific rules for this example.
include $(wildcard tensorflow/lite/experimental/micro/examples/micro_speech/*/Makefile.inc)

$(eval $(call microlite_test,micro_features_fft_test,\
$(MICRO_FEATURES_FFT_TEST_SRCS),$(MICRO_FEATURES_FFT_TEST_HDRS)))

$(eval $(call microlite_test,micro_features_filterbank_test,\
$(MICRO_FEATURES_FILTERBANK_TEST_SRCS),$(MICRO_FEATURES_FILTERBANK_TEST_HDRS)))

$(eval $(call microlite_test,micro_features_frontend_test,\
$(MICRO_FEATURES_FRONTEND_TEST_SRCS),$(MICRO_FEATURES_FRONTEND_TEST_HDRS)))

$(eval $(call microlite_test,micro_features_log_scale_test,\
$(MICRO_FEATURES_LOG_SCALE_TEST_SRCS),$(MICRO_FEATURES_LOG_SCALE_TEST_HDRS)))

$(eval $(call microlite_test,micro_features_noise_reduction_test,\
$(MICRO_FEATURES_NOISE_REDUCTION_TEST_SRCS),$(MICRO_FEATURES_NOISE_REDUCTION_TEST_HDRS)))

$(eval $(call microlite_test,micro_features_pcan_gain_control_test,\
$(MICRO_FEATURES_PCAN_GAIN_CONTROL_TEST_SRCS),$(MICRO_FEATURES_PCAN_GAIN_CONTROL_TEST_HDRS)))

$(eval $(call microlite_test,micro_features_window_test,\
$(MICRO_FEATURES_WINDOW_TEST_SRCS),$(MICRO_FEATURES_WINDOW_TEST_HDRS)))

# Test the code for feature generation.
$(eval $(call microlite_test,micro_features_generator_test,\
$(MICRO_FEATURES_GENERATOR_TEST_SRCS), $(MICRO_FEATURES_GENERATOR_TEST_HDRS)))

# Tests loading and running a speech model.
$(eval $(call microlite_test,micro_speech_test,\
$(MICRO_SPEECH_TEST_SRCS),$(MICRO_SPEECH_TEST_HDRS)))

# Test the code for feature generation.
$(eval $(call microlite_test,simple_features_generator_test,\
$(SIMPLE_FEATURES_GENERATOR_TEST_SRCS), $(SIMPLE_FEATURES_GENERATOR_TEST_HDRS)))

# Tests the audio provider module.
$(eval $(call microlite_test,audio_provider_test,\
$(AUDIO_PROVIDER_TEST_SRCS),$(AUDIO_PROVIDER_TEST_HDRS)))

# Tests the audio provider mock module.
$(eval $(call microlite_test,audio_provider_mock_test,\
$(AUDIO_PROVIDER_MOCK_TEST_SRCS),$(AUDIO_PROVIDER_MOCK_TEST_HDRS)))

# Tests the feature provider module.
$(eval $(call microlite_test,feature_provider_test,\
$(FEATURE_PROVIDER_TEST_SRCS),$(FEATURE_PROVIDER_TEST_HDRS)))

# Tests the feature provider module using the mock audio provider.
$(eval $(call microlite_test,feature_provider_mock_test,\
$(FEATURE_PROVIDER_MOCK_TEST_SRCS),$(FEATURE_PROVIDER_MOCK_TEST_HDRS)))

# Tests the command recognizer module.
$(eval $(call microlite_test,recognize_commands_test,\
$(RECOGNIZE_COMMANDS_TEST_SRCS),$(RECOGNIZE_COMMANDS_TEST_HDRS)))

# Tests responding to a command.
$(eval $(call microlite_test,command_responder_test,\
$(COMMAND_RESPONDER_TEST_SRCS),$(COMMAND_RESPONDER_TEST_HDRS)))

# Builds a standalone speech command recognizer binary.
$(eval $(call microlite_test,micro_speech,\
$(MICRO_SPEECH_SRCS),$(MICRO_SPEECH_HDRS)))

# Builds a standalone speech command recognizer binary using fake audio input.
$(eval $(call microlite_test,micro_speech_mock,\
$(MICRO_SPEECH_MOCK_SRCS),$(MICRO_SPEECH_MOCK_HDRS)))
