EXAMPLE_NAME:=hello_world

HELLO_WORLD_TEST_SRCS := \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/$(EXAMPLE_NAME)_test.cc \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/model.cc

HELLO_WORLD_TEST_HDRS := \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/model.h

OUTPUT_HANDLER_TEST_SRCS := \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/output_handler_test.cc \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/output_handler.cc

OUTPUT_HANDLER_TEST_HDRS := \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/output_handler.h \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/constants.h

HELLO_WORLD_SRCS := \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/main.cc \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/main_functions.cc \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/model.cc \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/output_handler.cc \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/constants.cc

HELLO_WORLD_HDRS := \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/model.h \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/output_handler.h \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/constants.h \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/main_functions.h

#Find any platform - specific rules for this example.
include $(wildcard tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/*/Makefile.inc)

# Tests loading and running the sine model.
$(eval $(call microlite_test,$(EXAMPLE_NAME)_test,\
$(HELLO_WORLD_TEST_SRCS),$(HELLO_WORLD_TEST_HDRS)))

# Tests producing an output.
$(eval $(call microlite_test,output_handler_test,\
$(OUTPUT_HANDLER_TEST_SRCS),$(OUTPUT_HANDLER_TEST_HDRS)))

# Builds a standalone binary.
$(eval $(call microlite_test,$(EXAMPLE_NAME),\
$(HELLO_WORLD_SRCS),$(HELLO_WORLD_HDRS)))

list_$(EXAMPLE_NAME)_example_sources:
	@echo $(HELLO_WORLD_SRCS)

list_$(EXAMPLE_NAME)_example_headers:
	@echo $(HELLO_WORLD_HDRS)
