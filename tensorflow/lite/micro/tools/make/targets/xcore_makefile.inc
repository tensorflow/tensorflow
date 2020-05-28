# Settings for XMOS XS3 based processors (xcore.ai, ...)

#IMPORTANT: to set up environment variables correctly run the following from the top tensorflow directory:
# $ make -f tensorflow/lite/micro/tools/make/Makefile TARGET="xcore" clean clean_downloads test
# $ pushd tensorflow/lite/micro/tools/make/downloads/xtimecomposer/xTIMEcomposer/15.0.0/ && source SetEnv && popd
# $ make -f tensorflow/lite/micro/tools/make/Makefile TARGET="xcore" test

ifeq ($(TARGET), xcore)
  XTIME_URL := "https://www.xmos.com/download/Tools-15---Linux-64%2815.0.0_rc1%29.tgz?key=132D-9DC9-E913-0229-ECE6-D5AB-F511-2B19"
  XTIME_MD5 := "8f6543c8ac4af7583edf75e62df322a2"
  $(eval $(call add_third_party_download,$(XTIME_URL),$(XTIME_MD5),xtimecomposer))
  PLATFORM_FLAGS = -target=XU316-1024-FB265-C32 -mcmodel=large -Os -DXCORE -Wno-xcore-fptrgroup -report
  CXX_TOOL := xcc
  CC_TOOL := xcc
  AR_TOOL := xmosar   
  override CXXFLAGS := -std=c++11 -g -DTF_LITE_STATIC_MEMORY -DNDEBUG
  override CXXFLAGS += $(PLATFORM_FLAGS) 
  override CCFLAGS := -g -DTF_LITE_STATIC_MEMORY -DNDEBUG
  override CCFLAGS += $(PLATFORM_FLAGS)
  TARGET_ARCH := xcore
  #TARGET_TOOLCHAIN_PREFIX := tensorflow/lite/micro/tools/make/downloads/xtimecomposer/bin/
  TEST_SCRIPT := tensorflow/lite/micro/testing/test_xcore_binary.sh
  #GCC_XCORE := $(MAKEFILE_DIR)/downloads/xtimecomposer/bin/
endif
