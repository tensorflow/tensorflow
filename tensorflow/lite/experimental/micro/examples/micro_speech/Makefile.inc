
# Tests loading and running a speech model.
MICRO_SPEECH_TEST_SRCS := \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_speech_test.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/tiny_conv_model_data.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/no_features_data.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/yes_features_data.cc
ALL_SRCS += $(MICRO_SPEECH_TEST_SRCS)
MICRO_SPEECH_TEST_OBJS := $(addprefix $(OBJDIR), \
$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(MICRO_SPEECH_TEST_SRCS))))
MICRO_SPEECH_TEST_BINARY := $(BINDIR)micro_speech_test
ALL_BINARIES += $(MICRO_SPEECH_TEST_BINARY)
$(MICRO_SPEECH_TEST_BINARY): $(MICRO_SPEECH_TEST_OBJS) $(MICROLITE_LIB_PATH)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
	-o $(MICRO_SPEECH_TEST_BINARY) $(MICRO_SPEECH_TEST_OBJS) \
	$(LIBFLAGS) $(MICROLITE_LIB_PATH) $(LDFLAGS) $(MICROLITE_LIBS)
micro_speech_test: $(MICRO_SPEECH_TEST_BINARY)
micro_speech_test_bin: $(MICRO_SPEECH_TEST_BINARY).bin
test_micro_speech: $(MICRO_SPEECH_TEST_BINARY)
	$(TEST_SCRIPT) $(MICRO_SPEECH_TEST_BINARY) '~~~ALL TESTS PASSED~~~'

# Source files that are used by multiple preprocessor tests.
PREPROCESSOR_TEST_SHARED_SRCS := \
tensorflow/lite/experimental/micro/examples/micro_speech/preprocessor_test.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/no_30ms_sample_data.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/yes_30ms_sample_data.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/no_power_spectrum_data.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/yes_power_spectrum_data.cc

# Test the float reference code for feature generation.
PREPROCESSOR_REFERENCE_TEST_SRCS = \
$(PREPROCESSOR_TEST_SHARED_SRCS) \
tensorflow/lite/experimental/micro/examples/micro_speech/preprocessor.cc
ALL_SRCS += $(PREPROCESSOR_REFERENCE_TEST_SRCS)
PREPROCESSOR_REFERENCE_TEST_OBJS := $(addprefix $(OBJDIR), \
$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(PREPROCESSOR_REFERENCE_TEST_SRCS))))
PREPROCESSOR_REFERENCE_TEST_BINARY := $(BINDIR)preprocessor_reference_test
ALL_BINARIES += $(PREPROCESSOR_REFERENCE_TEST_BINARY)
$(PREPROCESSOR_REFERENCE_TEST_BINARY): $(PREPROCESSOR_REFERENCE_TEST_OBJS) $(MICROLITE_LIB_PATH)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
	-o $(PREPROCESSOR_REFERENCE_TEST_BINARY) $(PREPROCESSOR_REFERENCE_TEST_OBJS) \
	$(LIBFLAGS) $(MICROLITE_LIB_PATH) $(LDFLAGS) $(MICROLITE_LIBS)
preprocessor_reference_test: $(PREPROCESSOR_REFERENCE_TEST_BINARY)
preprocessor_reference_test_bin: $(PREPROCESSOR_REFERENCE_TEST_BINARY).bin
test_preprocessor_reference: $(PREPROCESSOR_REFERENCE_TEST_BINARY)
	$(TEST_SCRIPT) $(PREPROCESSOR_REFERENCE_TEST_BINARY) '~~~ALL TESTS PASSED~~~'

# Test the fixed point reference code for feature generation.
PREPROCESSOR_FIXED_TEST_SRCS = \
$(PREPROCESSOR_TEST_SHARED_SRCS) \
tensorflow/lite/experimental/micro/examples/micro_speech/fixed_point/preprocessor.cc
ALL_SRCS += $(PREPROCESSOR_FIXED_TEST_SRCS)
PREPROCESSOR_FIXED_TEST_OBJS := $(addprefix $(OBJDIR), \
$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(PREPROCESSOR_FIXED_TEST_SRCS))))
PREPROCESSOR_FIXED_TEST_BINARY := $(BINDIR)preprocessor_fixed_test
ALL_BINARIES += $(PREPROCESSOR_FIXED_TEST_BINARY)
$(PREPROCESSOR_FIXED_TEST_BINARY): $(PREPROCESSOR_FIXED_TEST_OBJS) $(MICROLITE_LIB_PATH)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
	-o $(PREPROCESSOR_FIXED_TEST_BINARY) $(PREPROCESSOR_FIXED_TEST_OBJS) \
	$(LIBFLAGS) $(MICROLITE_LIB_PATH) $(LDFLAGS) $(MICROLITE_LIBS)
preprocessor_fixed_test: $(PREPROCESSOR_FIXED_TEST_BINARY)
preprocessor_fixed_test_bin: $(PREPROCESSOR_FIXED_TEST_BINARY).bin
test_preprocessor_fixed: $(PREPROCESSOR_FIXED_TEST_BINARY)
	$(TEST_SCRIPT) $(PREPROCESSOR_FIXED_TEST_BINARY) '~~~ALL TESTS PASSED~~~'

# Tests the audio provider module.
AUDIO_PROVIDER_TEST_SRCS := \
tensorflow/lite/experimental/micro/examples/micro_speech/audio_provider_test.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/model_settings.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/audio_provider.cc
ALL_SRCS += $(AUDIO_PROVIDER_TEST_SRCS)
AUDIO_PROVIDER_TEST_OBJS := $(addprefix $(OBJDIR), \
$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(AUDIO_PROVIDER_TEST_SRCS))))
AUDIO_PROVIDER_TEST_BINARY := $(BINDIR)audio_provider_test
ALL_BINARIES += $(AUDIO_PROVIDER_TEST_BINARY)
$(AUDIO_PROVIDER_TEST_BINARY): $(AUDIO_PROVIDER_TEST_OBJS) $(MICROLITE_LIB_PATH)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
	-o $(AUDIO_PROVIDER_TEST_BINARY) $(AUDIO_PROVIDER_TEST_OBJS) \
	$(LIBFLAGS) $(MICROLITE_LIB_PATH) $(LDFLAGS) $(MICROLITE_LIBS)
audio_provider_test: $(AUDIO_PROVIDER_TEST_BINARY)
audio_provider_test_bin: $(AUDIO_PROVIDER_TEST_BINARY).bin
test_audio_provider: $(AUDIO_PROVIDER_TEST_BINARY)
	$(TEST_SCRIPT) $(AUDIO_PROVIDER_TEST_BINARY) '~~~ALL TESTS PASSED~~~'

# Tests the feature provider module.
FEATURE_PROVIDER_TEST_SRCS := \
tensorflow/lite/experimental/micro/examples/micro_speech/feature_provider_test.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/model_settings.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/audio_provider.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/timer.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/preprocessor.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/feature_provider.cc
ALL_SRCS += $(FEATURE_PROVIDER_TEST_SRCS)
FEATURE_PROVIDER_TEST_OBJS := $(addprefix $(OBJDIR), \
$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(FEATURE_PROVIDER_TEST_SRCS))))
FEATURE_PROVIDER_TEST_BINARY := $(BINDIR)feature_provider_test
ALL_BINARIES += $(FEATURE_PROVIDER_TEST_BINARY)
$(FEATURE_PROVIDER_TEST_BINARY): $(FEATURE_PROVIDER_TEST_OBJS) $(MICROLITE_LIB_PATH)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
	-o $(FEATURE_PROVIDER_TEST_BINARY) $(FEATURE_PROVIDER_TEST_OBJS) \
	$(LIBFLAGS) $(MICROLITE_LIB_PATH) $(LDFLAGS) $(MICROLITE_LIBS)
feature_provider_test: $(FEATURE_PROVIDER_TEST_BINARY)
feature_provider_test_bin: $(FEATURE_PROVIDER_TEST_BINARY).bin
test_feature_provider: $(FEATURE_PROVIDER_TEST_BINARY)
	$(TEST_SCRIPT) $(FEATURE_PROVIDER_TEST_BINARY) '~~~ALL TESTS PASSED~~~'

# Tests the timer module.
TIMER_TEST_SRCS := \
tensorflow/lite/experimental/micro/examples/micro_speech/timer_test.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/timer.cc
ALL_SRCS += $(TIMER_TEST_SRCS)
TIMER_TEST_OBJS := $(addprefix $(OBJDIR), \
$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(TIMER_TEST_SRCS))))
TIMER_TEST_BINARY := $(BINDIR)timer_test
ALL_BINARIES += $(TIMER_TEST_BINARY)
$(TIMER_TEST_BINARY): $(TIMER_TEST_OBJS) $(MICROLITE_LIB_PATH)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
	-o $(TIMER_TEST_BINARY) $(TIMER_TEST_OBJS) \
	$(LIBFLAGS) $(MICROLITE_LIB_PATH) $(LDFLAGS) $(MICROLITE_LIBS)
timer_test: $(TIMER_TEST_BINARY)
timer_test_bin: $(TIMER_TEST_BINARY).bin
test_timer: $(TIMER_TEST_BINARY)
	$(TEST_SCRIPT) $(TIMER_TEST_BINARY) '~~~ALL TESTS PASSED~~~'

# Builds a standalone speech command recognizer binary.
MICRO_SPEECH_SRCS := \
tensorflow/lite/experimental/micro/examples/micro_speech/main.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/model_settings.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/audio_provider.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/feature_provider.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/timer.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/preprocessor.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/no_features_data.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/yes_features_data.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/tiny_conv_model_data.cc
ALL_SRCS += $(MICRO_SPEECH_SRCS)
MICRO_SPEECH_OBJS := $(addprefix $(OBJDIR), \
$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(MICRO_SPEECH_SRCS))))
MICRO_SPEECH_BINARY := $(BINDIR)micro_speech
ALL_BINARIES += $(MICRO_SPEECH_BINARY)
$(MICRO_SPEECH_BINARY): $(MICRO_SPEECH_OBJS) $(MICROLITE_LIB_PATH)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
	-o $(MICRO_SPEECH_BINARY) $(MICRO_SPEECH_OBJS) \
	$(LIBFLAGS) $(MICROLITE_LIB_PATH) $(LDFLAGS) $(MICROLITE_LIBS)
micro_speech: $(MICRO_SPEECH_BINARY)
micro_speech_bin: $(MICRO_SPEECH_BINARY).bin

# Find any platform-specific rules for this example.
include $(wildcard tensorflow/lite/experimental/micro/examples/micro_speech/*/Makefile.inc)
