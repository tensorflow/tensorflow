person_detection_MODEL_SRCS := \
tensorflow/lite/micro/examples/person_detection/model_settings.cc \
$(MAKEFILE_DIR)/downloads/person_model_int8/person_detect_model_data.cc

person_detection_MODEL_HDRS := \
tensorflow/lite/micro/examples/person_detection/model_settings.h \
tensorflow/lite/micro/examples/person_detection/person_detect_model_data.h

person_detection_TEST_SRCS := \
tensorflow/lite/micro/examples/person_detection/person_detection_test.cc \
$(MAKEFILE_DIR)/downloads/person_model_int8/no_person_image_data.cc \
$(MAKEFILE_DIR)/downloads/person_model_int8/person_image_data.cc \
$(person_detection_MODEL_SRCS)

person_detection_TEST_HDRS := \
tensorflow/lite/micro/examples/person_detection/no_person_image_data.h \
tensorflow/lite/micro/examples/person_detection/person_image_data.h \
$(person_detection_MODEL_HDRS)

IMAGE_PROVIDER_TEST_SRCS := \
tensorflow/lite/micro/examples/person_detection/image_provider.cc \
tensorflow/lite/micro/examples/person_detection/image_provider_test.cc \
tensorflow/lite/micro/examples/person_detection/model_settings.cc

IMAGE_PROVIDER_TEST_HDRS := \
tensorflow/lite/micro/examples/person_detection/image_provider.h \
tensorflow/lite/micro/examples/person_detection/model_settings.h

DETECTION_RESPONDER_TEST_SRCS := \
tensorflow/lite/micro/examples/person_detection/detection_responder.cc \
tensorflow/lite/micro/examples/person_detection/detection_responder_test.cc

DETECTION_RESPONDER_TEST_HDRS := \
tensorflow/lite/micro/examples/person_detection/detection_responder.h

person_detection_SRCS := \
tensorflow/lite/micro/examples/person_detection/detection_responder.cc \
tensorflow/lite/micro/examples/person_detection/image_provider.cc \
tensorflow/lite/micro/examples/person_detection/main.cc \
tensorflow/lite/micro/examples/person_detection/main_functions.cc \
$(person_detection_MODEL_SRCS)

person_detection_HDRS := \
tensorflow/lite/micro/examples/person_detection/detection_responder.h \
tensorflow/lite/micro/examples/person_detection/image_provider.h \
tensorflow/lite/micro/examples/person_detection/main_functions.h \
$(person_detection_MODEL_HDRS)

#Find any platform - specific rules for this example.
include $(wildcard tensorflow/lite/micro/examples/person_detection/*/Makefile.inc)

# Tests loading and running a vision model.
$(eval $(call microlite_test,person_detection_test_int8,\
$(person_detection_TEST_SRCS),$(person_detection_TEST_HDRS)))

# Three conflicting issues here:
# 1. The image_provider_test fails on Sparkfun Edge we do not have a way to
#    filter out individual tests within and example.
# 2. We do not want to completely remove person_detection from the sparkfun_edge
#    build.
# 3. We do want to keep as many targets as possible be part of the sparkfun_edge
#    CI build to avoid getting into similar situations where some parts of the
#    code are supported on a platform while other parts are not.
#
# The current nasty workaround is to explicitly exclude the offending test for
# the sparkfun_edge target. Note that we are not exluding it for
# TARGET=apollo3evb becuase that is not part of our CI builds (and the two are
# basically equivalent).
ifneq ($(TARGET),sparkfun_edge)
# Tests the image provider module.
$(eval $(call microlite_test,image_provider_test_int8,\
$(IMAGE_PROVIDER_TEST_SRCS),$(IMAGE_PROVIDER_TEST_HDRS)))
endif

# Tests the detection responder module.
$(eval $(call microlite_test,detection_responder_test_int8,\
$(DETECTION_RESPONDER_TEST_SRCS),$(DETECTION_RESPONDER_TEST_HDRS)))

# Builds a standalone object recognition binary.
$(eval $(call microlite_test,person_detection_int8,\
$(person_detection_SRCS),$(person_detection_HDRS)))
