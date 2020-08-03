$(eval $(call add_third_party_download,$(PERSON_MODEL_URL),$(PERSON_MODEL_MD5),person_model_grayscale,))

KEYWORD_BENCHMARK_SRCS := \
tensorflow/lite/micro/benchmarks/keyword_benchmark.cc \
tensorflow/lite/micro/benchmarks/keyword_scrambled_model_data.cc

KEYWORD_BENCHMARK_HDRS := \
tensorflow/lite/micro/benchmarks/keyword_scrambled_model_data.h

PERSON_DETECTION_BENCHMARK_SRCS := \
tensorflow/lite/micro/benchmarks/person_detection_benchmark.cc \
$(MAKEFILE_DIR)/downloads/person_model_grayscale/no_person_image_data.cc \
$(MAKEFILE_DIR)/downloads/person_model_grayscale/person_detect_model_data.cc \
$(MAKEFILE_DIR)/downloads/person_model_grayscale/person_image_data.cc

PERSON_DETECTION_BENCHMARK_HDRS := \
tensorflow/lite/micro/examples/person_detection/person_detect_model_data.h

# Builds a standalone binary.
$(eval $(call microlite_test,keyword_benchmark,\
$(KEYWORD_BENCHMARK_SRCS),$(KEYWORD_BENCHMARK_HDRS)))

$(eval $(call microlite_test,person_detection_benchmark,\
$(PERSON_DETECTION_BENCHMARK_SRCS),$(PERSON_DETECTION_BENCHMARK_HDRS)))


#
# EEMBC Harness
#

EEMBC_TH_SRCS := \
tensorflow/lite/micro/benchmarks/eembc/monitor/ee_main.cc \
tensorflow/lite/micro/$(TARGET)/eembc/monitor/th_api/th_lib.cc \
tensorflow/lite/micro/$(TARGET)/eembc/monitor/th_api/th_libc.cc \
tensorflow/lite/micro/benchmarks/eembc/profile/ee_profile.cc \
tensorflow/lite/micro/benchmarks/eembc/profile/ee_buffer.cc \
tensorflow/lite/micro/benchmarks/eembc/profile/ee_util.cc \

#
# person_detection for EEMBC, override GetImage and RespondToDetection
#
# The default model is person_image_data.cc.
#

PERSON_DETECTION_EEMBC_TH_SRCS := \
$(EEMBC_TH_SRCS) \
tensorflow/lite/micro/examples/person_detection/main_functions.cc \
$(MAKEFILE_DIR)/downloads/person_model_grayscale/person_image_data.cc \
tensorflow/lite/micro/benchmarks/eembc/profile/ee_person_detection.cc \
$(person_detection_MODEL_SRCS)
 
PERSON_DETECTION_EEMBC_TH_HDRS := \
tensorflow/lite/micro/examples/person_detection/main_functions.h \
tensorflow/lite/micro/examples/person_detection/no_person_image_data.h \
tensorflow/lite/micro/examples/person_detection/person_image_data.h \
$(person_detection_MODEL_HDRS)

$(eval $(call microlite_test,person_detection_eembc_th,\
$(PERSON_DETECTION_EEMBC_TH_SRCS),$(PERSON_DETECTION_EEMBC_TH_HDRS)))


#
# micro_speech for EEMBC, override GetAudioSamples and RespondToCommand
#

MICRO_SPEECH_EEMBC_TH_SRCS := \
$(EEMBC_TH_SRCS) \
tensorflow/lite/micro/examples/micro_speech/yes_1000ms_sample_data.cc \
tensorflow/lite/micro/examples/micro_speech/micro_features/model.cc \
tensorflow/lite/micro/examples/micro_speech/micro_features/micro_model_settings.cc \
tensorflow/lite/micro/examples/micro_speech/micro_features/micro_features_generator.cc \
tensorflow/lite/micro/examples/micro_speech/feature_provider.cc \
tensorflow/lite/micro/examples/micro_speech/recognize_commands.cc \
tensorflow/lite/micro/examples/micro_speech/main_functions.cc \
tensorflow/lite/micro/benchmarks/eembc/profile/ee_micro_speech.cc \
$(MICRO_FEATURES_LIB_SRCS) \
$(micro_speech_MODEL_SRCS)
 
MICRO_SPEECH_EEMBC_TH_HDRS := \
tensorflow/lite/micro/examples/micro_speech/yes_1000ms_sample_data.h \
tensorflow/lite/micro/examples/micro_speech/micro_features/model.h \
tensorflow/lite/micro/examples/micro_speech/micro_features/micro_model_settings.h \
tensorflow/lite/micro/examples/micro_speech/micro_features/micro_features_generator.h \
tensorflow/lite/micro/examples/micro_speech/feature_provider.h \
tensorflow/lite/micro/examples/micro_speech/recognize_commands.h \
tensorflow/lite/micro/examples/micro_speech/main_functions.h \
$(MICRO_FEATURES_LIB_HDRS) \
$(micro_speech_MODEL_HDRS)

$(eval $(call microlite_test,micro_speech_eembc_th,\
$(MICRO_SPEECH_EEMBC_TH_SRCS),$(MICRO_SPEECH_EEMBC_TH_HDRS)))

