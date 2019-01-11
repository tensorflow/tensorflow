#Settings for Mac OS platforms.
ifeq ($(TARGET), osx)

  PLATFORM_FLAGS = \
    -DTF_LITE_DISABLE_X86_NEON

  CXXFLAGS += $(PLATFORM_FLAGS)
  CCFLAGS += $(PLATFORM_FLAGS)

endif