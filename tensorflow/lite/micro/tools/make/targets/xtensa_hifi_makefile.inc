# Settings for Xtensa toolchain.
# Derived from xtensa_xpg_makefile.inc
# The Xtensa environment variables should be configured externally (XTENSA_CORE, XTENSA_SYSTEM)

ifeq ($(TARGET), xtensa_hifi)
  TARGET_ARCH := hifi3_bd5

$(eval $(call add_third_party_download,$(XTENSA_HIFI4_URL),$(XTENSA_HIFI4_MD5),xa_nnlib,))

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
