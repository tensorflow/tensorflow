# Settings for Xtensa toolchain for the hifi kernels.
# REQUIRED:
#  Environment variables:
#   - XTENSA_BASE  must be set to location of
#     the Xtensa developer tools installation directory.
#  Command line arguments:
#   - XTENSA_TOOLS_VERSION: For example: RI-2019.2-linux
#   - XTENSA_CORE: The name of the Xtensa core to use
#      For example: hifi3

ifeq ($(TARGET), xtensa_hifi)
  TARGET_ARCH := hifi3_bd5

  ifndef XTENSA_BASE
    $(error XTENSA_BASE is undefined)
  endif

  ifndef XTENSA_TOOLS_VERSION
    $(error XTENSA_TOOLS_VERSION is undefined)
  endif

  ifndef XTENSA_CORE
    $(error XTENSA_CORE is undefined)
  endif

  PLATFORM_ARGS = \
    -mno-mul16 \
    -mno-mul32 \
    -mno-div32 \
    -fsigned-char \
    -fno-exceptions \
    -mlongcalls \
    -INLINE:requested \
    -mcoproc \
    -fno-zero-initialized-in-bss \
    -mtext-section-literals \
    -fno-unsafe-math-optimizations \

  TF_LITE_MICRO_FLAGS = \
    -DTF_LITE_STATIC_MEMORY\

  export PATH := $(XTENSA_BASE)/tools/$(XTENSA_TOOLS_VERSION)/XtensaTools/bin:$(PATH)
  TARGET_TOOLCHAIN_PREFIX := xt-
  CXX_TOOL := clang++
  CC_TOOL := clang

  CXXFLAGS = -O0 $(PLATFORM_ARGS) -std=c++11 $(TF_LITE_MICRO_FLAGS)
  #TODO: Use -std=c11 ?
  CCFLAGS = -O3 $(PLATFORM_ARGS) $(TF_LITE_MICRO_FLAGS)

  TEST_SCRIPT := tensorflow/lite/micro/testing/test_xtensa_hifi_binary.sh

  # These are microcontroller-specific rules for converting the ELF output
  # of the linker into a binary image that can be loaded directly.
  OBJCOPY := $(TARGET_TOOLCHAIN_PREFIX)objcopy

  $(BINDIR)/%.bin: $(BINDIR)/%
	  echo "here"
	  @mkdir -p $(dir $@)
	  $(OBJCOPY) $< $@ -O binary
endif
