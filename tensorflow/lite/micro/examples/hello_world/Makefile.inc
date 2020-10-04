HELLO_WORLD_TEST_SRCS := \
tensorflow/lite/micro/examples/hello_world/hello_world_test.cc \
tensorflow/lite/micro/examples/hello_world/model.cc

HELLO_WORLD_TEST_HDRS := \
tensorflow/lite/micro/examples/hello_world/model.h

OUTPUT_HANDLER_TEST_SRCS := \
tensorflow/lite/micro/examples/hello_world/output_handler_test.cc \
tensorflow/lite/micro/examples/hello_world/output_handler.cc

OUTPUT_HANDLER_TEST_HDRS := \
tensorflow/lite/micro/examples/hello_world/output_handler.h \
tensorflow/lite/micro/examples/hello_world/constants.h

HELLO_WORLD_SRCS := \
tensorflow/lite/micro/examples/hello_world/main.cc \
tensorflow/lite/micro/examples/hello_world/main_functions.cc \
tensorflow/lite/micro/examples/hello_world/model.cc \
tensorflow/lite/micro/examples/hello_world/output_handler.cc \
tensorflow/lite/micro/examples/hello_world/constants.cc

HELLO_WORLD_HDRS := \
tensorflow/lite/micro/examples/hello_world/model.h \
tensorflow/lite/micro/examples/hello_world/output_handler.h \
tensorflow/lite/micro/examples/hello_world/constants.h \
tensorflow/lite/micro/examples/hello_world/main_functions.h

#Find any platform - specific rules for this example.
include $(wildcard tensorflow/lite/micro/examples/hello_world/*/Makefile.inc)

# Tests loading and running the sine model.
$(eval $(call microlite_test,hello_world_test,\
$(HELLO_WORLD_TEST_SRCS),$(HELLO_WORLD_TEST_HDRS)))

# Tests producing an output.
$(eval $(call microlite_test,output_handler_test,\
$(OUTPUT_HANDLER_TEST_SRCS),$(OUTPUT_HANDLER_TEST_HDRS)))

# Builds a standalone binary.
$(eval $(call microlite_test,hello_world,\
$(HELLO_WORLD_SRCS),$(HELLO_WORLD_HDRS)))
