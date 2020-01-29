# Settings for XMOS XS3 based processors

#IMPORTANT: to set up environment variables correctly run the following from the top tensorflow directory:
# $ make -f tensorflow/lite/micro/tools/make/Makefile TARGET="xcore" test
# $ pushd ./tensorflow/lite/micro/tools/make/downloads/xtimecomposer/ && source SetEnv && popd
# $ make -f tensorflow/lite/micro/tools/make/Makefile TARGET="xcore" test
# OR
# simply run the docker container before calling make, etc.:
# $ docker run -it -v$(pwd):/home/builder:z --rm andrewxcav/xcore_test:latest make -f tensorflow/lite/micro/tools/make/Makefile TARGET="xcore" <your make target>

#TODO changing this file doesn't mark build as dirty
ifeq ($(TARGET), xcore)
  XTIME_URL := "http://totoro.xmos.local:8080/83987c819e1ffa0deab46bf2580c409f/xtimecomposer.zip"
  XTIME_MD5 := "83987c819e1ffa0deab46bf2580c409f"
  $(eval $(call add_third_party_download,$(XTIME_URL),$(XTIME_MD5),xtimecomposer))
  PLATFORM_FLAGS = -target=XU316-1024-QF60-C20 -mcmodel=large -Os -DXCORE -Wno-xcore-fptrgroup -Wno-unknown-pragmas -Wno-unknown-attributes -report
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
