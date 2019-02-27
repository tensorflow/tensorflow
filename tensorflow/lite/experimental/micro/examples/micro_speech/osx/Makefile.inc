# Settings for Mac OS platforms.
ifeq ($(TARGET), osx)
  LINKER_FLAGS := \
    -framework Foundation \
    -framework AudioToolbox

  MICROLITE_LIBS += $(LINKER_FLAGS)
endif
