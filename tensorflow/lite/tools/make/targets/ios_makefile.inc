# Settings for iOS.
ifeq ($(TARGET), ios)
  BUILD_FOR_IOS_SIMULATOR := false
	ifeq ($(TARGET_ARCH), x86_64)
	  BUILD_FOR_IOS_SIMULATOR := true
	endif
	ifeq ($(TARGET_ARCH), i386)
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
	# Override TARGET_ARCH with armv7, armv7s, arm64, i386, or x86_64.
	TARGET_ARCH := x86_64
	CXXFLAGS += -miphoneos-version-min=$(MIN_SDK_VERSION) \
		-DGEMMLOWP_ALLOW_SLOW_SCALAR_FALLBACK \
		-DTF_LITE_USE_CBLAS \
		-fembed-bitcode \
		-Wno-c++11-narrowing \
		-mno-thumb \
		-fno-exceptions \
		-isysroot \
		${IPHONEOS_SYSROOT} \
		-arch $(TARGET_ARCH) \
		-O3
	CCFLAGS += -miphoneos-version-min=$(MIN_SDK_VERSION) \
		-fembed-bitcode \
		-mno-thumb \
		-isysroot \
		${IPHONEOS_SYSROOT} \
		-arch $(TARGET_ARCH) \
		-O3
	LDFLAGS := -fembed-bitcode \
		-miphoneos-version-min=${MIN_SDK_VERSION} \
		-framework Accelerate \
		-arch $(TARGET_ARCH)
endif
