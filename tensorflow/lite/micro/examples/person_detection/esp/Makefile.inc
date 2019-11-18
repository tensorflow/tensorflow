# Adding some esp specific files in the main CMakeLists.txt
ESP_PERSON_DETECTION_SRCS := \
tensorflow/lite/micro/examples/person_detection/esp/app_camera_esp.c

ESP_PERSON_DETECTION_HDRS := \
tensorflow/lite/micro/examples/person_detection/esp/app_camera_esp.h

CCFLAGS += -Wno-nonnull
CXXFLAGS += -Wno-return-type -Wno-strict-aliasing
person_detection_SRCS += $(ESP_PERSON_DETECTION_SRCS)
person_detection_HDRS += $(ESP_PERSON_DETECTION_HDRS)
MAIN_SRCS += $(ESP_PERSON_DETECTION_SRCS)
GRAYSCALE_SRCS := tensorflow/lite/micro/tools/make/downloads/person_model_grayscale/person_detect_model_data.cc
person_detection_SRCS := $(filter-out $(GRAYSCALE_SRCS), $(person_detection_SRCS))
MICROLITE_CC_SRCS += $(GRAYSCALE_SRCS)
