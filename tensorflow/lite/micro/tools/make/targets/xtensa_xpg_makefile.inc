# Settings for Xtensa XPG toolchain.
# REQUIRED:
#   - RI2019.2 Toolkit (for xt-clang/xt-clang++).
#   - XTENSA_CORE: The name of the core to use, will cause a compiler exception
#                  without providing a core.

ifeq ($(TARGET), xtensa-xpg)
  TARGET_ARCH := xtensa-xpg

  PLATFORM_ARGS = \
    -DTF_LITE_MCU_DEBUG_LOG \
    --xtensa-core=$(XTENSA_CORE) \
    -mcoproc \
    -DXTENSA -DMAX_RFFT_PWR=9 -DMIN_RFFT_PWR=MAX_RFFT_PWR \
    -fdata-sections \
    -ffunction-sections \
    -fno-exceptions \
    -fno-unwind-tables \
    -fno-use-cxa-atexit \
    -fmessage-length=0 \
    -fno-threadsafe-statics

  TARGET_TOOLCHAIN_PREFIX := xt-
  CXX_TOOL := clang++
  CC_TOOL := clang

  CXXFLAGS += $(PLATFORM_ARGS)
  CCFLAGS += $(PLATFORM_ARGS)

  LDFLAGS += -Wl,-gc-sections

  TEST_SCRIPT := tensorflow/lite/micro/testing/test_xtensa_xpg_binary.sh
endif
