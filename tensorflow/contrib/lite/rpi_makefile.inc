# Settings for Raspberry Pi.
ifeq ($(TARGET), RPI)
	ifeq ($(TARGET_ARCH), armv7)
		CXXFLAGS += \
			-march=armv7-a \
			-mfpu=neon-vfpv4 \
			-funsafe-math-optimizations \
			-ftree-vectorize

		CCFLAGS += \
			-march=armv7-a \
			-mfpu=neon-vfpv4 \
			-funsafe-math-optimizations \
			-ftree-vectorize

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

	OBJDIR := $(OBJDIR)rpi_$(TARGET_ARCH)/
	LIBDIR := $(LIBDIR)rpi_$(TARGET_ARCH)/
	BINDIR := $(BINDIR)rpi_$(TARGET_ARCH)/
	DEPDIR := $(DEPDIR)rpi_$(TARGET_ARCH)/
endif
