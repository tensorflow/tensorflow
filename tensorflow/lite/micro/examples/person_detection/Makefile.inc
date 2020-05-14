$(eval $(call add_third_party_download,$(PERSON_MODEL_URL),$(PERSON_MODEL_MD5),person_model_grayscale,))
$(eval $(call add_third_party_download,$(RUY_URL),$(RUY_MD5),ruy,))

person_detection_MODEL_SRCS := \
tensorflow/lite/micro/examples/person_detection/model_settings.cc \
$(MAKEFILE_DIR)/downloads/person_model_grayscale/person_detect_model_data.cc

person_detection_MODEL_HDRS := \
tensorflow/lite/micro/examples/person_detection/model_settings.h \
tensorflow/lite/micro/examples/person_detection/person_detect_model_data.h

person_detection_TEST_SRCS := \
tensorflow/lite/micro/examples/person_detection/person_detection_test.cc \
$(MAKEFILE_DIR)/downloads/person_model_grayscale/no_person_image_data.cc \
$(MAKEFILE_DIR)/downloads/person_model_grayscale/person_image_data.cc \
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
$(eval $(call microlite_test,person_detection_test,\
$(person_detection_TEST_SRCS),$(person_detection_TEST_HDRS)))

# Tests the image provider module.
$(eval $(call microlite_test,image_provider_test,\
$(IMAGE_PROVIDER_TEST_SRCS),$(IMAGE_PROVIDER_TEST_HDRS)))

# Tests the detection responder module.
$(eval $(call microlite_test,detection_responder_test,\
$(DETECTION_RESPONDER_TEST_SRCS),$(DETECTION_RESPONDER_TEST_HDRS)))


# Builds a standalone object recognition binary.
$(eval $(call microlite_test,person_detection,\
$(person_detection_SRCS),$(person_detection_HDRS)))
