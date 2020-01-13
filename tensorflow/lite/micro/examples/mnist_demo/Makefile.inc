
$(warning mnist demo Makefile has been included)

MNIST_DEMO_SRCS := \
tensorflow/lite/experimental/micro/examples/mnist_demo/main.cc \
tensorflow/lite/experimental/micro/examples/mnist_demo/model/mnist_model.cc

MNIST_DEMO_HDRS := \
tensorflow/lite/experimental/micro/examples/mnist_demo/model/mnist_model.h \
tensorflow/lite/experimental/micro/examples/mnist_demo/model/mnist_test_data.h

MNIST_DEMO_TEST_SRCS := \
tensorflow/lite/experimental/micro/examples/mnist_demo/mnist_demo_test.cc \
tensorflow/lite/experimental/micro/examples/mnist_demo/model/mnist_model.cc

MNIST_DEMO_TEST_HDRS := \
tensorflow/lite/experimental/micro/examples/mnist_demo/model/mnist_model.h \
tensorflow/lite/experimental/micro/examples/mnist_demo/model/mnist_test_data.h

# Find any platform-specific rules for this example.
include $(wildcard tensorflow/lite/experimental/micro/examples/mnist_demo/*/Makefile.inc)

# Tests the character classifier model with a seven.
$(eval $(call microlite_test,mnist_demo_test,\
$(MNIST_DEMO_TEST_SRCS),$(MNIST_DEMO_TEST_HDRS)))

# Builds an MNIST character classifier demo binary.
$(eval $(call microlite_test,mnist_demo,\
$(MNIST_DEMO_SRCS),$(MNIST_DEMO_HDRS)))



# Builds a standalone speech command recognizer binary.
$(eval $(call microlite_test,micro_speech,\
$(MICRO_SPEECH_SRCS),$(MICRO_SPEECH_HDRS)))
