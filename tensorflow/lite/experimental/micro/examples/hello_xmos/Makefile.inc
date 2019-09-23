HELLO_XMOS_TEST_SRCS := \
tensorflow/lite/experimental/micro/examples/hello_xmos/hello_xmos_test.cc \
tensorflow/lite/experimental/micro/examples/hello_xmos/sine_model_data.cc

HELLO_XMOS_TEST_HDRS := \
tensorflow/lite/experimental/micro/examples/hello_xmos/sine_model_data.h

OUTPUT_HANDLER_TEST_SRCS := \
tensorflow/lite/experimental/micro/examples/hello_xmos/output_handler_test.cc \
tensorflow/lite/experimental/micro/examples/hello_xmos/output_handler.cc

OUTPUT_HANDLER_TEST_HDRS := \
tensorflow/lite/experimental/micro/examples/hello_xmos/output_handler.h \
tensorflow/lite/experimental/micro/examples/hello_xmos/constants.h

HELLO_XMOS_SRCS := \
tensorflow/lite/experimental/micro/examples/hello_xmos/main.cc \
tensorflow/lite/experimental/micro/examples/hello_xmos/sine_model_data.cc \
tensorflow/lite/experimental/micro/examples/hello_xmos/output_handler.cc \
tensorflow/lite/experimental/micro/examples/hello_xmos/constants.cc

HELLO_XMOS_HDRS := \
tensorflow/lite/experimental/micro/examples/hello_xmos/sine_model_data.h \
tensorflow/lite/experimental/micro/examples/hello_xmos/output_handler.h \
tensorflow/lite/experimental/micro/examples/hello_xmos/constants.h

#Find any platform-specific rules for this example.
include $(wildcard tensorflow/lite/experimental/micro/examples/hello_xmos/*/Makefile.inc)

# Tests loading and running the sine model.
$(eval $(call microlite_test,hello_xmos_test,\
$(HELLO_XMOS_TEST_SRCS),$(HELLO_XMOS_TEST_HDRS)))

# Tests producing an output.
$(eval $(call microlite_test,output_handler_test,\
$(OUTPUT_HANDLER_TEST_SRCS),$(OUTPUT_HANDLER_TEST_HDRS)))

# Builds a standalone binary.
$(eval $(call microlite_test,hello_xmos,\
$(HELLO_XMOS_SRCS),$(HELLO_XMOS_HDRS)))
