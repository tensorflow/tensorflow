# Settings for iOS.
ifeq ($(TARGET), IOS)
        BUILD_FOR_IOS_SIMULATOR := false
	ifeq ($(IOS_ARCH), x86_64)
	     	BUILD_FOR_IOS_SIMULATOR := true
	endif
	ifeq ($(IOS_ARCH), i386)
	     	BUILD_FOR_IOS_SIMULATOR := true
	endif
	ifeq ($(BUILD_FOR_IOS_SIMULATOR), true)
		IPHONEOS_PLATFORM := $(shell xcrun --sdk iphonesimulator \
			--show-sdk-platform-path)
		IPHONEOS_SYSROOT := $(shell xcrun --sdk iphonesimulator \
			--show-sdk-path)
	else
		IPHONEOS_PLATFORM := $(shell xcrun --sdk iphoneos --show-sdk-platform-path)
		IPHONEOS_SYSROOT := $(shell xcrun --sdk iphoneos --show-sdk-path)
	endif
	IOS_SDK_VERSION := $(shell xcrun --sdk iphoneos --show-sdk-version)
	MIN_SDK_VERSION := 9.0
	# Override IOS_ARCH with armv7, armv7s, arm64, i386, or x86_64.
	IOS_ARCH := x86_64
	CXXFLAGS += -miphoneos-version-min=$(MIN_SDK_VERSION) \
		-DGEMMLOWP_ALLOW_SLOW_SCALAR_FALLBACK \
		-DTFLITE_USE_APPLE_ACCELERATE_FOR_CONV \
		-fembed-bitcode \
		-Wno-c++11-narrowing \
		-mno-thumb \
		-fno-exceptions \
		-isysroot \
		${IPHONEOS_SYSROOT} \
		-arch $(IOS_ARCH) \
		-O3
	ifeq ($(IOS_ARCH), x86_64)
		CXXFLAGS += -msse4.1
	endif
	CCFLAGS += -miphoneos-version-min=$(MIN_SDK_VERSION) \
		-fembed-bitcode \
		-mno-thumb \
		-isysroot \
		${IPHONEOS_SYSROOT} \
		-arch $(IOS_ARCH) \
		-O3
	LDFLAGS := -fembed-bitcode \
		-miphoneos-version-min=${MIN_SDK_VERSION} \
		-framework Accelerate \
		-arch $(IOS_ARCH)
	OBJDIR := $(OBJDIR)ios_$(IOS_ARCH)/
	LIBDIR := $(LIBDIR)ios_$(IOS_ARCH)/
	BINDIR := $(BINDIR)ios_$(IOS_ARCH)/
	DEPDIR := $(DEPDIR)ios_$(IOS_ARCH)/
endif
