$(eval $(call add_third_party_download,$(PERSON_MODEL_INT8_URL),$(PERSON_MODEL_INT8_MD5),person_model_int8,))

person_detection_MODEL_SRCS := \
tensorflow/lite/micro/examples/person_detection_experimental/model_settings.cc \
$(MAKEFILE_DIR)/downloads/person_model_int8/person_detect_model_data.cc

person_detection_MODEL_HDRS := \
tensorflow/lite/micro/examples/person_detection_experimental/model_settings.h \
tensorflow/lite/micro/examples/person_detection_experimental/person_detect_model_data.h

person_detection_TEST_SRCS := \
tensorflow/lite/micro/examples/person_detection_experimental/person_detection_test.cc \
$(MAKEFILE_DIR)/downloads/person_model_int8/no_person_image_data.cc \
$(MAKEFILE_DIR)/downloads/person_model_int8/person_image_data.cc \
$(person_detection_MODEL_SRCS)

person_detection_TEST_HDRS := \
tensorflow/lite/micro/examples/person_detection_experimental/no_person_image_data.h \
tensorflow/lite/micro/examples/person_detection_experimental/person_image_data.h \
$(person_detection_MODEL_HDRS)

IMAGE_PROVIDER_TEST_SRCS := \
tensorflow/lite/micro/examples/person_detection_experimental/image_provider.cc \
tensorflow/lite/micro/examples/person_detection_experimental/image_provider_test.cc \
tensorflow/lite/micro/examples/person_detection_experimental/model_settings.cc

IMAGE_PROVIDER_TEST_HDRS := \
tensorflow/lite/micro/examples/person_detection_experimental/image_provider.h \
tensorflow/lite/micro/examples/person_detection_experimental/model_settings.h

DETECTION_RESPONDER_TEST_SRCS := \
tensorflow/lite/micro/examples/person_detection_experimental/detection_responder.cc \
tensorflow/lite/micro/examples/person_detection_experimental/detection_responder_test.cc

DETECTION_RESPONDER_TEST_HDRS := \
tensorflow/lite/micro/examples/person_detection_experimental/detection_responder.h

person_detection_SRCS := \
tensorflow/lite/micro/examples/person_detection_experimental/detection_responder.cc \
tensorflow/lite/micro/examples/person_detection_experimental/image_provider.cc \
tensorflow/lite/micro/examples/person_detection_experimental/main.cc \
tensorflow/lite/micro/examples/person_detection_experimental/main_functions.cc \
$(person_detection_MODEL_SRCS)

person_detection_HDRS := \
tensorflow/lite/micro/examples/person_detection_experimental/detection_responder.h \
tensorflow/lite/micro/examples/person_detection_experimental/image_provider.h \
tensorflow/lite/micro/examples/person_detection_experimental/main_functions.h \
$(person_detection_MODEL_HDRS)

#Find any platform - specific rules for this example.
include $(wildcard tensorflow/lite/micro/examples/person_detection_experimental/*/Makefile.inc)

# Tests loading and running a vision model.
$(eval $(call microlite_test,person_detection_test_int8,\
$(person_detection_TEST_SRCS),$(person_detection_TEST_HDRS)))

# Tests the image provider module.
$(eval $(call microlite_test,image_provider_test_int8,\
$(IMAGE_PROVIDER_TEST_SRCS),$(IMAGE_PROVIDER_TEST_HDRS)))

# Tests the detection responder module.
$(eval $(call microlite_test,detection_responder_test_int8,\
$(DETECTION_RESPONDER_TEST_SRCS),$(DETECTION_RESPONDER_TEST_HDRS)))

# Builds a standalone object recognition binary.
$(eval $(call microlite_test,person_detection_int8,\
$(person_detection_SRCS),$(person_detection_HDRS)))
