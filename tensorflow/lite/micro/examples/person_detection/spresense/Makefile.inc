# Settings for Spresense platform for Person detection  example
# This should be read when the EXTERNALS_TENSORFLOW_EXAMPLE_PERSONDETECTION option is selected
# in Spresense configuration.

ifeq ($(TARGET), spresense)
ifeq ($(CONFIG_EXTERNALS_TENSORFLOW_EXAMPLE_PERSONDETECTION),y)

SPRESENSE_PERSON_DETECTION_EXCLUDED_SRCS = \
    tensorflow/lite/micro/examples/person_detection/main.cc \
    tensorflow/lite/micro/examples/person_detection/image_provider.cc

SPRESENSE_PERSON_DETECTION_SRCS = \
    tensorflow/lite/micro/examples/person_detection/spresense/src/spresense_image_provider.cc \
    $(filter-out $(SPRESENSE_PERSON_DETECTION_EXCLUDED_SRCS),$(person_detection_SRCS))

# In spresence case, those file should be included into libtensorflow-microlite.
THIRD_PARTY_CC_SRCS += $(SPRESENSE_PERSON_DETECTION_SRCS)

endif
endif
