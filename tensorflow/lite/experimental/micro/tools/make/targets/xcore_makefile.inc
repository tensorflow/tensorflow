# Settings for XMOS XS3 based processors
#-Wno-unknown-attributes
ifeq ($(TARGET), xcore)
  PLATFORM_FLAGS = -target=XU316-1024-QF60-C20 -mcmodel=large -Os -DXCORE -Wno-xcore-fptrgroup -Wno-unknown-pragmas -report
  CXX_TOOL := xcc
  CC_TOOL := xcc
  AR_TOOL := xmosar   
  override CXXFLAGS := -std=c++11 -g -DTF_LITE_STATIC_MEMORY -DNDEBUG
  override CXXFLAGS += $(PLATFORM_FLAGS) 
  override CCFLAGS := -g -DTF_LITE_STATIC_MEMORY -DNDEBUG
  override CCFLAGS += $(PLATFORM_FLAGS)
  TARGET_ARCH := xcore
  TARGET_TOOLCHAIN_PREFIX := tensorflow/lite/experimental/micro/tools/make/downloads/xtimecomposer/bin/
  TEST_SCRIPT := tensorflow/lite/experimental/micro/testing/test_xcore_binary.sh
  GCC_XCORE := $(MAKEFILE_DIR)/downloads/xtimecomposer/bin/
endif
