# Settings for XMOS XS3 based processors

#IMPORTANT: 
# either install xmos tools 15.x on your system
# OR (assumes docker installed and docker daemon running)
# simply run the docker container before calling make, etc.:
# $ docker run -it -v$(pwd):/home/builder:z --rm xcoreai/build-tools:latest make -f tensorflow/lite/micro/tools/make/Makefile TARGET="xcore" <your make target>

#TODO changing this file doesn't mark build as dirty
ifeq ($(TARGET), xcore)
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
