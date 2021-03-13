# Settings for Spresense platform for Hello World example
# This should be read when the EXTERNALS_TENSORFLOW_EXAMPLE_HELLOWORLD option is selected
# in Spresense configuration.

ifeq ($(TARGET), spresense)
ifeq ($(CONFIG_EXTERNALS_TENSORFLOW_EXAMPLE_HELLOWORLD),y)

SPRESENSE_HELLO_WORLD_EXCLUDED_SRCS = \
    tensorflow/lite/micro/examples/hello_world/main.cc

SPRESENSE_HELLO_WORLD_SRCS = \
    $(filter-out $(SPRESENSE_HELLO_WORLD_EXCLUDED_SRCS),$(HELLO_WORLD_SRCS))

# In spresence case, those file should be included into libtensorflow-microlite.
THIRD_PARTY_CC_SRCS += $(SPRESENSE_HELLO_WORLD_SRCS)

endif
endif
