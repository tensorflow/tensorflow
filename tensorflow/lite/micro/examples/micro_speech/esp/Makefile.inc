# Adding some esp specific files in the main CMakeLists.txt
ESP_MICRO_SPEECH_SRCS := \
tensorflow/lite/micro/examples/micro_speech/esp/ringbuf.c

ESP_MICRO_SPEECH_HDRS := \
tensorflow/lite/micro/examples/micro_speech/esp/ringbuf.h

CCFLAGS += -Wno-nonnull
CXXFLAGS += -Wno-return-type -Wno-strict-aliasing -Wno-ignored-qualifiers
MICRO_SPEECH_SRCS += $(ESP_MICRO_SPEECH_SRCS)
MICRO_SPEECH_HDRS += $(ESP_MICRO_SPEECH_HDRS)
MAIN_SRCS += $(ESP_MICRO_SPEECH_SRCS)
# Adding the microfrontend lib in the CMakeLists.txt of tfmicro
PROJECT_INCLUDES += \
tensorflow/lite/experimental/microfrontend/lib

MICRO_FRONTEND_LIB_SRCS := \
tensorflow/lite/experimental/microfrontend/lib/fft.cc \
tensorflow/lite/experimental/microfrontend/lib/fft_util.cc \
tensorflow/lite/experimental/microfrontend/lib/filterbank.c \
tensorflow/lite/experimental/microfrontend/lib/filterbank_util.c \
tensorflow/lite/experimental/microfrontend/lib/frontend.c \
tensorflow/lite/experimental/microfrontend/lib/frontend_util.c \
tensorflow/lite/experimental/microfrontend/lib/log_lut.c \
tensorflow/lite/experimental/microfrontend/lib/log_scale.c \
tensorflow/lite/experimental/microfrontend/lib/log_scale_util.c \
tensorflow/lite/experimental/microfrontend/lib/noise_reduction.c \
tensorflow/lite/experimental/microfrontend/lib/noise_reduction_util.c \
tensorflow/lite/experimental/microfrontend/lib/pcan_gain_control.c \
tensorflow/lite/experimental/microfrontend/lib/pcan_gain_control_util.c \
tensorflow/lite/experimental/microfrontend/lib/window.c \
tensorflow/lite/experimental/microfrontend/lib/window_util.c

# Adding the micro frontend lib srcs into the CMakeLists.txt of tfmicro
MICROLITE_CC_SRCS += $(MICRO_FRONTEND_LIB_SRCS)
# Adding the kissfft srcs in the CMakeLists.txt of tfmicro
THIRD_PARTY_CC_SRCS += $(KISSFFT_LIB_SRCS)
# stopping microfrontend srcs from being included in the main srcs
MICRO_SPEECH_SRCS := $(filter-out $(MICRO_FRONTEND_LIB_SRCS), $(MICRO_SPEECH_SRCS))
MICRO_SPEECH_SRCS := $(filter-out $(KISSFFT_LIB_SRCS), $(MICRO_SPEECH_SRCS))

ESP_PROJECT_FILES += \
sdkconfig.defaults
