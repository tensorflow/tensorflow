# Settings for Raspberry Pi.
ifeq ($(TARGET),rpi)
  # Default to the architecture used on the Pi Two/Three (ArmV7), but override this
  # with TARGET_ARCH=armv6 to build for the Pi Zero or One.
  TARGET_ARCH := armv7l
  TARGET_TOOLCHAIN_PREFIX := arm-linux-gnueabihf-

  ifeq ($(TARGET_ARCH), armv7)
    CXXFLAGS += \
			-march=armv7-a \
      -mfpu=neon-vfpv4 \
      -funsafe-math-optimizations \
      -ftree-vectorize \
      -fPIC

    CCFLAGS += \
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
  endif

  # TODO(petewarden) In the future, we'll want to use OpenBLAS as a faster
  # alternative to Eigen on non-NEON ARM hardware like armv6.
  ifeq ($(TARGET_ARCH), armv6)
    CXXFLAGS += \
      -march=armv6 \
      -mfpu=vfp \
      -funsafe-math-optimizations \
      -ftree-vectorize \
      -fPIC

    CCFLAGS += \
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
    -lstdc++ \
    -lpthread \
    -lm \
    -ldl

endif
