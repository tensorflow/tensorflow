# Settings for Hexagon toolchain.
# REQUIRED:
#   - Hexagon SDK 3.5 Toolkit (for hexagon-clang++, hexagon-sim).
#   - HEXAGON_SDK_PREFIX environment variable must be set to location of
#     Hexagon_SDK/<version>/ on your machine.
#   - HEXAGON_CPU_VER: The CPU version to use, will cause a compiler exception
#                  without providing a version. Acceptable values: v55-v67
#
# Unlike other targets, there is not currently a way to automatically download
# the Hexagon SDK.  For this reason, users are required to manually download
# and configure the SDK.
ifeq ($(TARGET), hexagon)
  TARGET_ARCH := hexagon

  ifndef HEXAGON_SDK_PREFIX
    $(error HEXAGON_SDK_PREFIX is undefined)
  endif

  ifndef HEXAGON_CPU_VER
    $(error HEXAGON_CPU_VER is undefined)
  endif

  PLATFORM_ARGS = \
    -DTF_LITE_MCU_DEBUG_LOG \
    -DHEXAGON_ASM \
    -DMALLOC_IN_STDLIB \
    -DPTHREAD_STUBS \
    -DUSE_PREALLOCATED_BUFFER \
    -D_HAS_C9X \
    -MMD \
    -DHEXAGON \
    -Wall \
    -Wextra \
    -Wno-missing-field-initializers \
    -Wno-sign-compare \
    -Wno-unused-parameter \
    -Wno-write-strings \
    -Wvla \
    -fdata-sections \
    -ffunction-sections \
    -fmessage-length=0 \
    -fno-builtin \
    -fno-delete-null-pointer-checks \
    -fno-exceptions \
    -fno-register-global-dtors-with-atexit \
    -fno-rtti \
    -fno-short-enums \
    -fno-threadsafe-statics \
    -fno-unwind-tables \
    -fno-use-cxa-atexit \
    -fomit-frame-pointer \
    -fpermissive \
    -funsigned-char \
    -mcpu=$(HEXAGON_CPU_VER) \
    -m$(HEXAGON_CPU_VER)

  TARGET_TOOLCHAIN_PREFIX := hexagon-
  CXX_TOOL := clang++
  CC_TOOL := clang

  CXXFLAGS += $(PLATFORM_ARGS)
  CCFLAGS += $(PLATFORM_ARGS)
  LDFLAGS += \
    -Wl,--gc-sections -lhexagon \
    $(HEXAGON_SDK_PREFIX)/tools/HEXAGON_Tools/8.3.07/Tools/target/hexagon/lib/v66/libstdc++.a

  INCLUDES += \
    -I$(HEXAGON_SDK_PREFIX)/libs/common/qurt/computev66/include/posix \
    -I$(HEXAGON_SDK_PREFIX)/libs/common/qurt/computev66/include/qurt

  TEST_SCRIPT := tensorflow/lite/micro/testing/test_hexagon_binary.sh
endif
