# Settings for generic aarch64 boards such as Odroid C2 or Pine64.
ifeq ($(TARGET),aarch64)
  # The aarch64 architecture covers all 64-bit ARM chips. This arch mandates
  # NEON, so FPU flags are not needed below.
  TARGET_ARCH := aarch64
  TARGET_TOOLCHAIN_PREFIX := aarch64-linux-gnu-
  TARGET_OUT_DIR := linux_aarch64

  CXXFLAGS += \
    -march=armv8-a \
    -funsafe-math-optimizations \
    -ftree-vectorize \
    -fPIC

  CFLAGS += \
    -march=armv8-a \
    -funsafe-math-optimizations \
    -ftree-vectorize \
    -fPIC

  LDFLAGS := \
    -Wl,--no-export-dynamic \
    -Wl,--exclude-libs,ALL \
    -Wl,--gc-sections \
    -Wl,--as-needed \
    -lrt

  LIBS := \
    -lstdc++ \
    -lpthread \
    -lm \
    -ldl

endif
