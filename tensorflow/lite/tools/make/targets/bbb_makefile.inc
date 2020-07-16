# Settings for BeagleBone Black.
ifeq ($(TARGET),bbb)
  TARGET_ARCH := armv7l
  TARGET_TOOLCHAIN_PREFIX := arm-linux-gnueabihf-

  ifeq ($(TARGET_ARCH), armv7l)
    CXXFLAGS += \
      -march=armv7-a \
      -mfpu=neon \
      -funsafe-math-optimizations \
      -ftree-vectorize \
      -fPIC

    CFLAGS += \
      -march=armv7-a \
      -mfpu=neon \
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
    -ldl \
    -lrt

endif
