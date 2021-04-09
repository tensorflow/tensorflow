# Settings for Espressif ESP32

TARGET_ARCH := xtensa-esp32
CCFLAGS := $(filter-out -std=c11,$(CCFLAGS))
CFLAGS += -std=c11
