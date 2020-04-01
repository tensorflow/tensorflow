# Settings for Raspberry Pi.
ifeq ($(TARGET),rpi)
  # Default to the architecture used on the Pi Two/Three (ArmV7), but override this
  # with TARGET_ARCH=armv6 to build for the Pi Zero or One.
  TARGET_ARCH := armv7l

  ifeq ($(TARGET_ARCH), armv7l)
    TARGET_TOOLCHAIN_PREFIX := arm-linux-gnueabihf-
    CXXFLAGS += \
      -march=armv7-a \
      -mfpu=neon-vfpv4 \
      -funsafe-math-optimizations \
      -ftree-vectorize \
      -fPIC

    CFLAGS += \
      -march=armv7-a \
      -mfpu=neon-vfpv4 \
      -funsafe-math-optimizations \
      -ftree-vectorize \
      -fPIC

    LDFLAGS := \
      -Wl,--no-export-dynamic \
      -Wl,--exclude-libs,ALL \
      -Wl,--gc-sections \
      -Wl,--as-needed

    BUILD_WITH_RUY := true
  endif

  # TODO(petewarden) In the future, we'll want to use OpenBLAS as a faster
  # alternative to Eigen on non-NEON ARM hardware like armv6.
  ifeq ($(TARGET_ARCH), armv6)
    TARGET_TOOLCHAIN_PREFIX := arm-linux-gnueabi-
    CXXFLAGS += \
      -march=armv6 \
      -mfpu=vfp \
      -funsafe-math-optimizations \
      -ftree-vectorize \
      -fPIC

    CFLAGS += \
      -march=armv6 \
      -mfpu=vfp \
      -funsafe-math-optimizations \
      -ftree-vectorize \
      -fPIC

    LDFLAGS := \
      -Wl,--no-export-dynamic \
      -Wl,--exclude-libs,ALL \
      -Wl,--gc-sections \
      -Wl,--as-needed
  endif

  LIBS := \
    -latomic \
    -lstdc++ \
    -lpthread \
    -lm \
    -ldl

endif
