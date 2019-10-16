# Settings for XMOS XS3 based processors
ifeq ($(TARGET), xcore)
  PLATFORM_FLAGS = -target=XU316-1024-QF60-C20 -mcmodel=large -O0 -DDEBUG_PRINT_ENABLE -DXCORE -Wno-unknown-attributes -Wno-xcore-fptrgroup -report
  CXX_TOOL := xcc
  CC_TOOL := xcc
  AR_TOOL := xmosar   
  CXXFLAGS += $(PLATFORM_FLAGS)
  CCFLAGS += $(PLATFORM_FLAGS)
  TARGET_ARCH := xcore
  TARGET_TOOLCHAIN_PREFIX := tensorflow/lite/experimental/micro/tools/make/downloads/xtimecomposer/bin/
  TEST_SCRIPT := tensorflow/lite/experimental/micro/testing/test_xcore_binary.sh
  GCC_XCORE := $(MAKEFILE_DIR)/downloads/xtimecomposer/
endif
