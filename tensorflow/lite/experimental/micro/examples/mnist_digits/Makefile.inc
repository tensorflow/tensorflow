MNIST_DIGITS_TEST_SRCS := \
tensorflow/lite/experimental/micro/examples/mnist_digits/mnist_digits_test.cc \
tensorflow/lite/experimental/micro/examples/mnist_digits/mnist_model_data.cc

MNIST_DIGITS_TEST_HDRS := \
tensorflow/lite/experimental/micro/examples/mnist_digits/mnist_model_data.h

OUTPUT_HANDLER_TEST_SRCS := \
tensorflow/lite/experimental/micro/examples/mnist_digits/output_handler_test.cc \
tensorflow/lite/experimental/micro/examples/mnist_digits/output_handler.cc

OUTPUT_HANDLER_TEST_HDRS := \
tensorflow/lite/experimental/micro/examples/mnist_digits/output_handler.h \
tensorflow/lite/experimental/micro/examples/mnist_digits/constants.h

MNIST_DIGITS_SRCS := \
tensorflow/lite/experimental/micro/examples/mnist_digits/main.cc \
tensorflow/lite/experimental/micro/examples/mnist_digits/mnist_model_data.cc \
tensorflow/lite/experimental/micro/examples/mnist_digits/output_handler.cc \
tensorflow/lite/experimental/micro/examples/mnist_digits/constants.cc

MNIST_DIGITS_HDRS := \
tensorflow/lite/experimental/micro/examples/mnist_digits/mnist_model_data.h \
tensorflow/lite/experimental/micro/examples/mnist_digits/output_handler.h \
tensorflow/lite/experimental/micro/examples/mnist_digits/constants.h

#Find any platform-specific rules for this example.
include $(wildcard tensorflow/lite/experimental/micro/examples/mnist_digits/*/Makefile.inc)

# Tests loading and running the mnist model.
$(eval $(call microlite_test,mnist_digits_test,\
$(MNIST_DIGITS_TEST_SRCS),$(MNIST_DIGITS_TEST_HDRS)))

# Tests producing an output.
$(eval $(call microlite_test,output_handler_test,\
$(OUTPUT_HANDLER_TEST_SRCS),$(OUTPUT_HANDLER_TEST_HDRS)))

# Builds a standalone binary.
$(eval $(call microlite_test,mnist_digits,\
$(MNIST_DIGITS_SRCS),$(MNIST_DIGITS_HDRS)))
