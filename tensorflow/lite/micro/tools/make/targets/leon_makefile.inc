# Settings for SparcV8 based LEON processors from Gaisler Aeroflex
ifeq ($(TARGET), leon)
  PLATFORM_FLAGS = -O3 -mcpu=leon3
  CXXFLAGS += -std=c++11 $(PLATFORM_FLAGS)
  CCFLAGS += $(PLATFORM_FLAGS)
  TARGET_ARCH := leon
  TARGET_TOOLCHAIN_PREFIX := tensorflow/lite/micro/tools/make/downloads/leon_bcc2/bin/sparc-gaisler-elf-
  TEST_SCRIPT := tensorflow/lite/micro/testing/test_leon_binary.sh
  GCC_LEON := $(MAKEFILE_DIR)/downloads/leon_bcc2/

  $(eval $(call add_third_party_download,$(LEON_BCC2_URL),$(LEON_BCC2_MD5),leon_bcc2,))
  $(eval $(call add_third_party_download,$(TSIM_URL),$(TSIM_MD5),tsim,))

endif
