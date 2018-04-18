# Settings for Android.
ifeq ($(TARGET), ANDROID)
	HOST_ARCH = $(ARCH)

	#If ANDROID_TYPES is not set assume __ANDROID_TYPES_SLIM__
	ifeq ($(ANDROID_TYPES), )
		ANDROID_TYPES := -D__ANDROID_TYPES_SLIM__
	endif

	ANDROID_HOST_OS_ARCH :=
	ifeq ($(HOST_OS),LINUX)
		ANDROID_HOST_OS_ARCH=linux
	endif
	ifeq ($(HOST_OS),OSX)
		ANDROID_HOST_OS_ARCH=darwin
	endif
	ifeq ($(HOST_OS),WINDOWS)
    $(error "windows is not supported.")
	endif

	ifeq ($(HOST_ARCH),x86_32)
		ANDROID_HOST_OS_ARCH := $(ANDROID_HOST_OS_ARCH)-x86
	else
		ANDROID_HOST_OS_ARCH := $(ANDROID_HOST_OS_ARCH)-$(HOST_ARCH)
	endif

	ifndef ANDROID_ARCH
		ANDROID_ARCH := armeabi-v7a
	endif

	ifeq ($(ANDROID_ARCH),arm64-v8a)
		TOOLCHAIN := aarch64-linux-android-4.9
		SYSROOT_ARCH := arm64
		BIN_PREFIX := aarch64-linux-android
		MARCH_OPTION :=
	endif
	ifeq ($(ANDROID_ARCH),armeabi)
		TOOLCHAIN := arm-linux-androideabi-4.9
		SYSROOT_ARCH := arm
		BIN_PREFIX := arm-linux-androideabi
		MARCH_OPTION :=
	endif
	ifeq ($(ANDROID_ARCH),armeabi-v7a)
		TOOLCHAIN := arm-linux-androideabi-4.9
		SYSROOT_ARCH := arm
		BIN_PREFIX := arm-linux-androideabi
		MARCH_OPTION := -march=armv7-a -mfloat-abi=softfp -mfpu=neon
	endif
	ifeq ($(ANDROID_ARCH),mips)
		TOOLCHAIN := mipsel-linux-android-4.9
		SYSROOT_ARCH := mips
		BIN_PREFIX := mipsel-linux-android
		MARCH_OPTION :=
	endif
	ifeq ($(ANDROID_ARCH),mips64)
		TOOLCHAIN := mips64el-linux-android-4.9
		SYSROOT_ARCH := mips64
		BIN_PREFIX := mips64el-linux-android
		MARCH_OPTION :=
	endif
	ifeq ($(ANDROID_ARCH),x86)
		TOOLCHAIN := x86-4.9
		SYSROOT_ARCH := x86
		BIN_PREFIX := i686-linux-android
		MARCH_OPTION :=
	endif
	ifeq ($(ANDROID_ARCH),x86_64)
		TOOLCHAIN := x86_64-4.9
		SYSROOT_ARCH := x86_64
		BIN_PREFIX := x86_64-linux-android
		MARCH_OPTION :=
	endif

	ifndef NDK_ROOT
    $(error "NDK_ROOT is not defined.")
	endif

	CXX := $(CC_PREFIX) $(NDK_ROOT)/toolchains/$(TOOLCHAIN)/prebuilt/$(ANDROID_HOST_OS_ARCH)/bin/$(BIN_PREFIX)-g++

	CC := $(CC_PREFIX) $(NDK_ROOT)/toolchains/$(TOOLCHAIN)/prebuilt/$(ANDROID_HOST_OS_ARCH)/bin/$(BIN_PREFIX)-gcc

	CXXFLAGS +=\
	--sysroot $(NDK_ROOT)/platforms/android-21/arch-$(SYSROOT_ARCH) \
	-Wno-narrowing \
	-fomit-frame-pointer \
	$(MARCH_OPTION) \
	-fPIE \
	-fPIC \
	-D__USE_MAKE__

	INCLUDES += \
	-I. \
	-I$(NDK_ROOT)/sources/android/support/include \
	-I$(NDK_ROOT)/sources/android/cpufeatures \
	-I$(NDK_ROOT)/sources/cxx-stl/gnu-libstdc++/4.9/include \
	-I$(NDK_ROOT)/sources/cxx-stl/gnu-libstdc++/4.9/libs/$(ANDROID_ARCH)/include \
	-I$(NDK_ROOT)/platforms/android-21/arch-arm/usr/include \
	-I$(NDK_ROOT)/platforms/android-21/arch-arm/usr/include/machine

	LIBS := \
	-lgnustl_static \
	-llog \
	-lz \
	-lm \
	-ldl \
	-latomic \
	-lcpufeatures

	LD := $(NDK_ROOT)/toolchains/$(TOOLCHAIN)/prebuilt/$(ANDROID_HOST_OS_ARCH)/$(BIN_PREFIX)/bin/ld

	LDFLAGS := \
	$(MARCH_OPTION) \
	-L$(NDK_ROOT)/sources/cxx-stl/gnu-libstdc++/4.9/libs/$(ANDROID_ARCH) \
	-L$(NDK_ROOT)/sources/android/cpufeatures/obj/local/$(ANDROID_ARCH) \
	-fPIE \
	-pie \
	-v

	AR := $(NDK_ROOT)/toolchains/$(TOOLCHAIN)/prebuilt/$(ANDROID_HOST_OS_ARCH)/bin/$(BIN_PREFIX)-ar
	ARFLAGS := r
	LIBFLAGS += -Wl,--allow-multiple-definition -Wl,--whole-archive

	OBJDIR := $(OBJDIR)android_$(ANDROID_ARCH)/
	LIBDIR := $(LIBDIR)android_$(ANDROID_ARCH)/
	BINDIR := $(BINDIR)android_$(ANDROID_ARCH)/
	GENDIR := $(GENDIR)android_$(ANDROID_ARCH)/
endif
