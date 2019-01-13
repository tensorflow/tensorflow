# Settings for Raspberry Pi.
ifeq ($(TARGET),aarch64)
  # The aarch64 architecture covers all 64-bit ARM chips. This arch mandates
  # NEON, so FPU flags are not needed below.
  TARGET_ARCH := armv8-a
  TARGET_TOOLCHAIN_PREFIX := aarch64-linux-gnu-

  CXXFLAGS += \
    -march=armv8-a \
    -funsafe-math-optimizations \
    -ftree-vectorize \
    -fPIC

  CCFLAGS += \
    -march=armv8-a \
    -funsafe-math-optimizations \
    -ftree-vectorize \
    -fPIC

  LDFLAGS := \
    -Wl,--no-export-dynamic \
    -Wl,--exclude-libs,ALL \
    -Wl,--gc-sections \
    -Wl,--as-needed

       
  LIBS := \
    -lstdc++ \
    -lpthread \
    -lm \
    -ldl

endif
