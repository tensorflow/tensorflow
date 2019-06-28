# Settings for x86 on Linux
ifeq ($(TARGET), leon)
  PLATFORM_FLAGS = -O3 -mcpu=leon3
  CXXFLAGS += -std=c++11 $(PLATFORM_FLAGS)
  CCFLAGS += $(PLATFORM_FLAGS)
  TARGET_ARCH := leon
  TARGET_TOOLCHAIN_PREFIX := sparc-gaisler-elf-
  TEST_SCRIPT := tensorflow/lite/experimental/micro/testing/test_leon_binary.sh
endif
