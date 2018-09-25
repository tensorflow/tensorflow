# Settings for i.MX6 SOC.
ifeq ($(TARGET),mx6)
  TARGET_ARCH := armhf
  TARGET_TOOLCHAIN_PREFIX := arm-linux-gnueabihf-

  CXXFLAGS += \
    -march=armv7-a \
    -mcpu=cortex-a9 \
    -mtune=cortex-a9 \
    -mfpu=neon \
    -mvectorize-with-neon-quad \
    -mfloat-abi=hard \
    -funsafe-math-optimizations \
    -ftree-vectorize \
    -fPIC

  CCFLAGS += \
    -march=armv7-a \
    -mcpu=cortex-a9 \
    -mtune=cortex-a9 \
    -mfpu=neon \
    -mfloat-abi=hard \
    -mvectorize-with-neon-quad \
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
