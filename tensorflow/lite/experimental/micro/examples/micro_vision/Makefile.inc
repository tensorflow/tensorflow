$(eval $(call add_third_party_download,$(PERSON_MODEL_URL),$(PERSON_MODEL_MD5),person_model_grayscale,))

MICRO_VISION_MODEL_SRCS := \
tensorflow/lite/experimental/micro/examples/micro_vision/model_settings.cc \
$(MAKEFILE_DIR)/downloads/person_model_grayscale/person_detect_model_data.cc

MICRO_VISION_MODEL_HDRS := \
tensorflow/lite/experimental/micro/examples/micro_vision/model_settings.h \
tensorflow/lite/experimental/micro/examples/micro_vision/person_detect_model_data.h

MICRO_VISION_TEST_SRCS := \
tensorflow/lite/experimental/micro/examples/micro_vision/micro_vision_test.cc \
$(MAKEFILE_DIR)/downloads/person_model_grayscale/no_person_image_data.cc \
$(MAKEFILE_DIR)/downloads/person_model_grayscale/person_image_data.cc \
$(MICRO_VISION_MODEL_SRCS)

MICRO_VISION_TEST_HDRS := \
tensorflow/lite/experimental/micro/examples/micro_vision/no_person_image_data.h \
tensorflow/lite/experimental/micro/examples/micro_vision/person_image_data.h \
$(MICRO_VISION_MODEL_HDRS)

IMAGE_PROVIDER_TEST_SRCS := \
tensorflow/lite/experimental/micro/examples/micro_vision/image_provider.cc \
tensorflow/lite/experimental/micro/examples/micro_vision/image_provider_test.cc \
tensorflow/lite/experimental/micro/examples/micro_vision/model_settings.cc

IMAGE_PROVIDER_TEST_HDRS := \
tensorflow/lite/experimental/micro/examples/micro_vision/image_provider.h \
tensorflow/lite/experimental/micro/examples/micro_vision/model_settings.h

DETECTION_RESPONDER_TEST_SRCS := \
tensorflow/lite/experimental/micro/examples/micro_vision/detection_responder.cc \
tensorflow/lite/experimental/micro/examples/micro_vision/detection_responder_test.cc

DETECTION_RESPONDER_TEST_HDRS := \
tensorflow/lite/experimental/micro/examples/micro_vision/detection_responder.h

MICRO_VISION_SRCS := \
tensorflow/lite/experimental/micro/examples/micro_vision/detection_responder.cc \
tensorflow/lite/experimental/micro/examples/micro_vision/image_provider.cc \
tensorflow/lite/experimental/micro/examples/micro_vision/main.cc \
$(MICRO_VISION_MODEL_SRCS)

MICRO_VISION_HDRS := \
tensorflow/lite/experimental/micro/examples/micro_vision/detection_responder.h \
tensorflow/lite/experimental/micro/examples/micro_vision/image_provider.h \
$(MICRO_VISION_MODEL_HDRS)

# Find any platform-specific rules for this example.
include $(wildcard tensorflow/lite/experimental/micro/examples/micro_vision/*/Makefile.inc)

# Tests loading and running a vision model.
$(eval $(call microlite_test,micro_vision_test,\
$(MICRO_VISION_TEST_SRCS),$(MICRO_VISION_TEST_HDRS)))

# Tests the image provider module.
$(eval $(call microlite_test,image_provider_test,\
$(IMAGE_PROVIDER_TEST_SRCS),$(IMAGE_PROVIDER_TEST_HDRS)))

# Tests the detection responder module.
$(eval $(call microlite_test,detection_responder_test,\
$(DETECTION_RESPONDER_TEST_SRCS),$(DETECTION_RESPONDER_TEST_HDRS)))

# Builds a standalone object recognition binary.
$(eval $(call microlite_test,micro_vision,\
$(MICRO_VISION_SRCS),$(MICRO_VISION_HDRS)))
