# Settings for XMOS XS3 based processors
ifeq ($(TARGET), xcore)

#  LIB_LOGGING=tensorflow/lite/experimental/micro/tools/make/downloads/lib_logging
#  xccgcc -target=XU316-1024-QF60-C20 -Os -mcmodel=large -report -g -DDEBUG_PRINT_ENABLE -c -I $(LIB_LOGGING)/api/ $(LIB_LOGGING)/src/debug_printf.c
  PLATFORM_FLAGS = -target=XU316-1024-QF60-C20 -Os -report -g -DDEBUG_PRINT_ENABLE -mcmodel=large 
  CXXFLAGS += -std=c++11 $(PLATFORM_FLAGS)
  CCFLAGS += $(PLATFORM_FLAGS)
  TARGET_ARCH := xcore
  TARGET_TOOLCHAIN_PREFIX := tensorflow/lite/experimental/micro/tools/make/downloads/xtimecomposer/bin/xcc
  TEST_SCRIPT := tensorflow/lite/experimental/micro/testing/test_xcore_binary.sh
  GCC_XCORE := $(MAKEFILE_DIR)/downloads/xtimecomposer/
  
endif
