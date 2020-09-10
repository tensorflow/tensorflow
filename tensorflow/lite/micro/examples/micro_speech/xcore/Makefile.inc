ifeq ($(TARGET), xcore)

  override PLATFORM_FLAGS = -target=XCORE-AI-EXPLORER -mcmodel=large -Os -DXCORE -Wno-xcore-fptrgroup -report -fxscope -DDEBUG_PRINT_ENABLE=1  -DMIC_DUAL_ENABLED=1
  override CXXFLAGS := -std=c++11 -g -DTF_LITE_STATIC_MEMORY -DNDEBUG
  override CXXFLAGS += $(PLATFORM_FLAGS)
  override CCFLAGS := -g -DTF_LITE_STATIC_MEMORY -DNDEBUG
  override CCFLAGS += $(PLATFORM_FLAGS)

# Recipe to make xc objects
$(OBJDIR)%.xo: %.xc $(THIRD_PARTY_TARGETS)
	@mkdir -p $(dir $@)
	$(CC) $(CCFLAGS) $(INCLUDES) -c $< -o $@

MICRO_SPEECH_SRCS += \
	tensorflow/lite/micro/examples/micro_speech/xcore/main.xo                      \
	tensorflow/lite/micro/examples/micro_speech/xcore/fifo.c                       \
	tensorflow/lite/micro/examples/micro_speech/xcore/microspeech_xcore_support.c

MICRO_SPEECH_HDRS += \
	tensorflow/lite/micro/examples/micro_speech/xcore/main_support.h    \
	tensorflow/lite/micro/examples/micro_speech/xcore/fifo.h

# Append includes
THIRD_PARTY_CC_SRCS += \
    $(MAKEFILE_DIR)/downloads/lib_xassert/lib_xassert/src/xassert.xo                \
    $(MAKEFILE_DIR)/downloads/lib_mic_array/lib_mic_array/src/mic_array_dual.xo     \
    $(MAKEFILE_DIR)/downloads/lib_mic_array/lib_mic_array/src/fir/fir_coefs_dual.xo \
    $(MAKEFILE_DIR)/downloads/lib_mic_array/lib_mic_array/src/fir/fir_coefs.xo      \
    $(MAKEFILE_DIR)/downloads/lib_logging/lib_logging/src/debug_printf.c

INCLUDES += \
    -Itensorflow/lite/micro/examples/micro_speech/xcore/inc         \
    -I$(MAKEFILE_DIR)/downloads/lib_xassert/lib_xassert/api         \
    -I$(MAKEFILE_DIR)/downloads/lib_dsp/lib_dsp/api                 \
    -I$(MAKEFILE_DIR)/downloads/lib_mic_array/lib_mic_array/api     \
    -I$(MAKEFILE_DIR)/downloads/lib_mic_array/lib_mic_array/src/fir \
    -I$(MAKEFILE_DIR)/downloads/lib_logging/lib_logging/api

endif
