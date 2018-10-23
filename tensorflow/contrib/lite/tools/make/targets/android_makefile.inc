# Settings for Android.
# NDK r14, API 21
# Change `SYSROOT`, `NDK` and `API_VERSION` for your own env
# `NDK` dir name must be `ndk`

ifeq ($(TARGET), android)
    ifeq ($(TARGET_ARCH), armv7)
        TARGET_TOOLCHAIN_PREFIX := arm-linux-androideabi-

    	CXXFLAGS += --sysroot=${SYSROOT} \
            -D__ANDROID_API__=${API_VERSION} \
            -march=armv7-a \
            -std=c++11 \
            -mfpu=neon \
            -mfloat-abi=softfp \
            -fPIE \
            -pie \
            -fvisibility=default \
            -Wl,--no-export-dynamic \
            -Wl,--exclude-libs,ALL \
            -Wl,--as-needed \
            -s \
            -DNDEBUG \
    		-Os
    
    	CCFLAGS += --sysroot=${SYSROOT} \
            -march=armv7-a \
            -mfpu=neon \
            -D__ANDROID_API__=${API_VERSION} \
            -mfloat-abi=softfp \
            -pie \
            -s \
            -O3
    
        INCLUDES += -I${NDK}

    	LDFLAGS += -rdynamic -pie \
            -std=c++11 \
            -L${SYSROOT}/usr/lib \
            -llog \
            -D__ANDROID_API__=${API_VERSION} \
            -rdynamic -fPIE \
            -Wl,--gc-sections
    endif

    ifeq ($(TARGET_ARCH), armv8)
        TARGET_TOOLCHAIN_PREFIX := aarch64-linux-android-

    	CXXFLAGS += --sysroot=${SYSROOT} \
            -march=armv8-a \
            -D__ANDROID_API__=${API_VERSION} \
            -std=c++11 \
            -fPIE \
            -pie \
            -fvisibility=default \
            -Wl,--no-export-dynamic \
            -Wl,--exclude-libs,ALL \
            -Wl,--as-needed \
            -s \
    		-O3
    
    	CCFLAGS += --sysroot=${SYSROOT} \
            -march=armv8-a \
            -D__ANDROID_API__=${API_VERSION} \
            -pie \
            -s \
            -O3
    
        INCLUDES += -I${NDK}

    	LDFLAGS += -rdynamic -pie \
            -std=c++11 \
            -L${SYSROOT}/usr/lib \
            -llog \
            -rdynamic -fPIE \
            -Wl,--gc-sections \
            -D__ANDROID_API__=${API_VERSION}
    endif
endif
