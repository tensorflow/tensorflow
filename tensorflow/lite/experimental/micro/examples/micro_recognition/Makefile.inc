MICRO_RECOGNITION_HDRS := \
tensorflow/lite/experimental/micro/examples/micro_recognition/image_recognition_model.h \
tensorflow/lite/experimental/micro/examples/micro_recognition/image_provider.h \
tensorflow/lite/experimental/micro/examples/micro_recognition/disco/image_util.h \
tensorflow/lite/experimental/micro/examples/micro_recognition/disco/display_util.h \
tensorflow/lite/experimental/micro/examples/micro_recognition/util.h

MICRO_RECOGNITION_SRCS := \
tensorflow/lite/experimental/micro/examples/micro_recognition/image_recognition_model.cc \
tensorflow/lite/experimental/micro/examples/micro_recognition/main.cc \
tensorflow/lite/experimental/micro/examples/micro_recognition/disco/image_provider.cc \
tensorflow/lite/experimental/micro/examples/micro_recognition/disco/image_util.cc \
tensorflow/lite/experimental/micro/examples/micro_recognition/disco/display_util.cc

MICRO_RECOGNITION_TEST_SRCS := \
tensorflow/lite/experimental/micro/examples/micro_recognition/micro_recognition_test.cc \
tensorflow/lite/experimental/micro/examples/micro_recognition/image_recognition_model.cc

MICRO_RECOGNITION_TEST_HDRS := \
tensorflow/lite/experimental/micro/examples/micro_recognition/image_recognition_model.h \
tensorflow/lite/experimental/micro/examples/micro_recognition/util.h

CIFAR10_DATASET_URL="https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
CIFAR10_DATASET_MD5="c32a1d4ab5d03f1284b67883e8d87530"
$(eval $(call add_third_party_download,$(CIFAR10_DATASET_URL),$(CIFAR10_DATASET_MD5),cifar10,patch_cifar10_dataset))

include $(wildcard tensorflow/lite/experimental/micro/examples/micro_recognition/*/Makefile.inc)


ifneq ($(filter disco_f746ng,$(ALL_TAGS)),)
  MBED_PROJECT_FILES += \
    BSP_DISCO_F746NG.lib \
    LCD_DISCO_F746NG.lib
endif

$(eval $(call microlite_test,micro_recognition_test,\
$(MICRO_RECOGNITION_TEST_SRCS),$(MICRO_RECOGNITION_TEST_HDRS)))

$(eval $(call microlite_test,micro_recognition,\
$(MICRO_RECOGNITION_SRCS),$(MICRO_RECOGNITION_HDRS)))
