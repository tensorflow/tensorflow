$(eval $(call add_third_party_download,$(IMAGE_RECOGNITION_MODEL_URL),$(IMAGE_RECOGNITION_MODEL_MD5),image_recognition_model,))
$(eval $(call add_third_party_download,$(CIFAR10_DATASET_URL),$(CIFAR10_DATASET_MD5),cifar10,patch_cifar10_dataset))

IMAGE_RECOGNITION_HDRS := \
tensorflow/lite/micro/examples/image_recognition_experimental/image_recognition_model.h \
tensorflow/lite/micro/examples/image_recognition_experimental/image_provider.h \
tensorflow/lite/micro/examples/image_recognition_experimental/stm32f746_discovery/image_util.h \
tensorflow/lite/micro/examples/image_recognition_experimental/stm32f746_discovery/display_util.h \
tensorflow/lite/micro/examples/image_recognition_experimental/util.h

IMAGE_RECOGNITION_SRCS := \
$(MAKEFILE_DIR)/downloads/image_recognition_model/image_recognition_model.cc \
tensorflow/lite/micro/examples/image_recognition_experimental/main.cc \
tensorflow/lite/micro/examples/image_recognition_experimental/stm32f746_discovery/image_provider.cc \
tensorflow/lite/micro/examples/image_recognition_experimental/stm32f746_discovery/image_util.cc \
tensorflow/lite/micro/examples/image_recognition_experimental/stm32f746_discovery/display_util.cc

IMAGE_RECOGNITION_TEST_SRCS := \
tensorflow/lite/micro/examples/image_recognition_experimental/image_recognition_test.cc \
$(MAKEFILE_DIR)/downloads/image_recognition_model/image_recognition_model.cc

IMAGE_RECOGNITION_TEST_HDRS := \
tensorflow/lite/micro/examples/image_recognition_experimental/image_recognition_model.h \
tensorflow/lite/micro/examples/image_recognition_experimental/util.h

include $(wildcard tensorflow/lite/micro/examples/image_recognition_experimental/*/Makefile.inc)

ifneq ($(filter disco_f746ng,$(ALL_TAGS)),)
  MBED_PROJECT_FILES += \
    BSP_DISCO_F746NG.lib \
    LCD_DISCO_F746NG.lib
endif

$(eval $(call microlite_test,image_recognition,\
$(IMAGE_RECOGNITION_SRCS),$(IMAGE_RECOGNITION_HDRS)))

$(eval $(call microlite_test,image_recognition_test,\
$(IMAGE_RECOGNITION_TEST_SRCS),$(IMAGE_RECOGNITION_TEST_HDRS)))
