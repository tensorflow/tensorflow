# Settings for Mac OS platforms.
ifeq ($(TARGET), osx)

  # Make sure we can find the embedded GCC compiler.
  export PATH := ${PATH}:tensorflow/lite/micro/tools/make/downloads/gcc_embedded/bin/

  PLATFORM_FLAGS = \
    -DTF_LITE_DISABLE_X86_NEON

  CXXFLAGS += $(PLATFORM_FLAGS)
  CCFLAGS += $(PLATFORM_FLAGS)

endif