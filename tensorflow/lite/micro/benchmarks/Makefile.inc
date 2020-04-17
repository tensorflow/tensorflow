KEYWORD_BENCHMARK_SRCS := \
tensorflow/lite/micro/benchmarks/keyword_benchmark.cc \
tensorflow/lite/micro/benchmarks/keyword_scrambled_model_data.cc

KEYWORD_BENCHMARK_HDRS := \
tensorflow/lite/micro/benchmarks/keyword_scrambled_model_data.h

# Builds a standalone binary.
$(eval $(call microlite_test,keyword_benchmark,\
$(KEYWORD_BENCHMARK_SRCS),$(KEYWORD_BENCHMARK_HDRS)))
