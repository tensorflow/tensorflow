# Settings for Hexagon toolchain.
ifeq ($(TARGET), hexagon)
  TARGET_ARCH := hexagon

  PLATFORM_ARGS = \
    -DHEXAGON_ASM \
    -DMALLOC_IN_STDLIB \
    -DMICRO_NN_ENABLED=1 \
    -DMICRO_TFLITE_ENABLED=0 \
    -DNDEBUG \
    -DPTHREAD_STUBS \
    -DTF_LITE_STATIC_MEMORY \
    -DUSE_PREALLOCATED_BUFFER \
    -D_HAS_C9X \
    -MMD \
    -O3 -DNDEBUG -DHEXAGON \
    -Wall \
    -Wextra \
    -Wno-missing-field-initializers \
    -Wno-sign-compare \
    -Wno-unused-parameter \
    -Wno-write-strings \
    -Wvla \
    -fdata-sections -ffunction-sections \
    -fdata-sections \
    -ffunction-sections \
    -fmessage-length=0 \
    -fno-builtin \
    -fno-builtin \
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
    -mcpu=v66 \
    -mv66

  TARGET_TOOLCHAIN_PREFIX := hexagon-
  CXX_TOOL := clang++
  CC_TOOL := clang

  CXXFLAGS = $(PLATFORM_ARGS) -std=c++11
  CCFLAGS = $(PLATFORM_ARGS) -std=c11
  LDFLAGS += \
    -Wl,--gc-sections -lhexagon

  INCLUDES += \
    -I$(HEXAGON_SDK_PREFIX)/libs/common/qurt/computev66/include/posix \
    -I$(HEXAGON_SDK_PREFIX)/libs/common/qurt/computev66/include/qurt

  TEST_SCRIPT := tensorflow/lite/micro/testing/test_hexagon_binary.sh

  # These are microcontroller-specific rules for converting the ELF output
  # of the linker into a binary image that can be loaded directly.
  OBJCOPY := $(TARGET_TOOLCHAIN_PREFIX)objcopy

  $(BINDIR)/%.bin: $(BINDIR)/%
	  @mkdir -p $(dir $@)
	  $(OBJCOPY) $< $@ -O binary
endif
