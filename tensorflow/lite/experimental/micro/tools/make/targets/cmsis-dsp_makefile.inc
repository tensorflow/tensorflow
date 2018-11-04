ifeq ($(VENDORLIB), cmsis-dsp)
PREPROCESSOR_TEST_SRCS := \
tensorflow/contrib/lite/experimental/micro/examples/micro_speech/CMSIS/preprocessor.cc \
tensorflow/contrib/lite/experimental/micro/examples/micro_speech/CMSIS/preprocessor_test.cc \
tensorflow/contrib/lite/experimental/micro/examples/micro_speech/CMSIS/no_30ms_sample_data.cc \
tensorflow/contrib/lite/experimental/micro/examples/micro_speech/CMSIS/yes_30ms_sample_data.cc \
tensorflow/contrib/lite/experimental/micro/examples/micro_speech/CMSIS/no_power_spectrum_data.cc \
tensorflow/contrib/lite/experimental/micro/examples/micro_speech/CMSIS/yes_power_spectrum_data.cc
endif
