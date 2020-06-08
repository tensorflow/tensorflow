# Settings for Mac OS platforms.
ifeq ($(TARGET), osx)
  LINKER_FLAGS := \
    -framework Foundation \
    -framework AudioToolbox

  MICROLITE_LIBS += $(LINKER_FLAGS)
  MICRO_SPEECH_HDRS += tensorflow/lite/micro/examples/micro_speech/simple_features/simple_model_settings.h
endif
