# Settings for Xtensa XPG toolchain.
# REQUIRED:
#   - RI2019.2 Toolkit (for xt-clang/xt-clang++).
#   - XTENSA_CORE: The name of the core to use, will cause a compiler exception
#                  without providing a core.
ifeq ($(TARGET), xtensa-xpg)
  TARGET_ARCH := xtensa-xpg

  PLATFORM_ARGS = \
    -DTF_LITE_STATIC_MEMORY \
    -DNDEBUG \
    -DTF_LITE_MCU_DEBUG_LOG \
    --xtensa-core=$(XTENSA_CORE) \
    -mcoproc \
    -O3 \
    -DXTENSA -DMAX_RFFT_PWR=9 -DMIN_RFFT_PWR=MAX_RFFT_PWR -fdata-sections \
    -fmessage-length=0

  TARGET_TOOLCHAIN_PREFIX := xt-
  CXX_TOOL := clang++
  CC_TOOL := clang

  CXXFLAGS = $(PLATFORM_ARGS) -std=c++11
  CCFLAGS = $(PLATFORM_ARGS) -std=c11

  TEST_SCRIPT := tensorflow/lite/micro/testing/test_xtensa_xpg_binary.sh

  # These are microcontroller-specific rules for converting the ELF output
  # of the linker into a binary image that can be loaded directly.
  OBJCOPY := $(TARGET_TOOLCHAIN_PREFIX)objcopy

  $(BINDIR)/%.bin: $(BINDIR)/%
	  @mkdir -p $(dir $@)
	  $(OBJCOPY) $< $@ -O binary
endif
