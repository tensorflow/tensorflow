KEYWORD_BENCHMARK_SRCS := \
tensorflow/lite/micro/benchmarks/keyword_benchmark.cc \
tensorflow/lite/micro/benchmarks/keyword_scrambled_model_data.cc

KEYWORD_BENCHMARK_HDRS := \
tensorflow/lite/micro/benchmarks/keyword_scrambled_model_data.h \
tensorflow/lite/micro/benchmarks/micro_benchmark.h

PERSON_DETECTION_BENCHMARK_SRCS := \
tensorflow/lite/micro/benchmarks/person_detection_benchmark.cc \
$(MAKEFILE_DIR)/downloads/person_model_int8/no_person_image_data.cc \
$(MAKEFILE_DIR)/downloads/person_model_int8/person_detect_model_data.cc \
$(MAKEFILE_DIR)/downloads/person_model_int8/person_image_data.cc

PERSON_DETECTION_BENCHMARK_HDRS := \
tensorflow/lite/micro/examples/person_detection/person_detect_model_data.h

# Builds a standalone binary.
$(eval $(call microlite_test,keyword_benchmark,\
$(KEYWORD_BENCHMARK_SRCS),$(KEYWORD_BENCHMARK_HDRS)))

$(eval $(call microlite_test,person_detection_benchmark,\
$(PERSON_DETECTION_BENCHMARK_SRCS),$(PERSON_DETECTION_BENCHMARK_HDRS)))
