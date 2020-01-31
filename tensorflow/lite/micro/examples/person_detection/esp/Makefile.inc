# Adding some esp specific files in the main CMakeLists.txt
ESP_PERSON_DETECTION_SRCS := \
tensorflow/lite/micro/examples/person_detection/esp/app_camera_esp.c

ESP_PERSON_DETECTION_HDRS := \
tensorflow/lite/micro/examples/person_detection/esp/app_camera_esp.h

person_detection_SRCS += $(ESP_PERSON_DETECTION_SRCS)
person_detection_HDRS += $(ESP_PERSON_DETECTION_HDRS)
MAIN_SRCS += $(ESP_PERSON_DETECTION_SRCS)

# ESP specific flags and files
CCFLAGS += -Wno-nonnull
CXXFLAGS += -Wno-return-type -Wno-strict-aliasing
ESP_PROJECT_FILES += \
  sdkconfig.defaults \
  main/Kconfig.projbuild

# Remap downloaded model files as if they were in tensorflow/lite/micro/examples/..
MODEL_DOWNLOADS_DIR := tensorflow/lite/micro/tools/make/downloads/person_model_grayscale
MODEL_EXAMPLES_DIR := tensorflow/lite/micro/examples/person_detection/person_model_grayscale
person_detection_SRCS := $(patsubst $(MODEL_DOWNLOADS_DIR)/%,$(MODEL_EXAMPLES_DIR)/%,$(person_detection_SRCS))

# Custom rule to transform downloaded model files
$(PRJDIR)person_detection/esp-idf/main/person_model_grayscale/%: $(MODEL_DOWNLOADS_DIR)/%
	@mkdir -p $(dir $@)
	@python tensorflow/lite/micro/tools/make/transform_source.py \
        --platform=esp \
        --is_example_source \
        --source_path="$(patsubst $(MODEL_DOWNLOADS_DIR)/%,$(MODEL_EXAMPLES_DIR)/%,$<)" \
        < $< > $@
