#Settings for Mac OS platforms.
ifeq ($(TARGET), osx)
  MICROLITE_LIBS += \
    -framework Foundation \
    -framework AudioToolbox
endif
