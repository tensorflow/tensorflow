# Settings for apollo3 evb platforms.
ifeq ($(TARGET), apollo3evb)

  PUSHBUTTON_MICRO_SPEECH_TEST_SRCS := \
    $(AP3_MICRO_DIR)/../preprocessor.cc \
    $(AP3_MICRO_DIR)/pushbutton_main.c \
    $(AP3_MICRO_DIR)/pushbutton_test.cc \
    $(AP3_MICRO_DIR)/../tiny_conv_model_data.cc \
    $(APOLLO3_SDK)/devices/am_devices_led.c
  ALL_SRCS += $(PUSHBUTTON_MICRO_SPEECH_TEST_SRCS)
  PUSHBUTTON_MICRO_SPEECH_TEST_OBJS := $(addprefix $(OBJDIR), \
    $(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(PUSHBUTTON_MICRO_SPEECH_TEST_SRCS))))
  PUSHBUTTON_MICRO_SPEECH_TEST_BINARY := $(BINDIR)pushbutton_micro_speech_test
  $(PUSHBUTTON_MICRO_SPEECH_TEST_BINARY): $(PUSHBUTTON_MICRO_SPEECH_TEST_OBJS) $(MICROLITE_LIB_PATH)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
	-o $(PUSHBUTTON_MICRO_SPEECH_TEST_BINARY) $(PUSHBUTTON_MICRO_SPEECH_TEST_OBJS) \
	$(LIBFLAGS) $(MICROLITE_LIB_PATH) $(LDFLAGS) $(MICROLITE_LIBS)
  pushbutton_micro_speech_test: $(PUSHBUTTON_MICRO_SPEECH_TEST_BINARY)
  pushbutton_micro_speech_test_bin: $(PUSHBUTTON_MICRO_SPEECH_TEST_BINARY).bin
  test_pushbutton_micro_speech: $(PUSHBUTTON_MICRO_SPEECH_TEST_BINARY)
	$(TEST_SCRIPT) $(PUSHBUTTON_MICRO_SPEECH_TEST_BINARY) '~~~ALL TESTS PASSED~~~'

  PUSHBUTTON_CMSIS_SPEECH_TEST_SRCS := \
    $(AP3_MICRO_DIR)/pushbutton_main.c \
    $(AP3_MICRO_DIR)/pushbutton_test.cc \
    $(AP3_MICRO_DIR)/../tiny_conv_model_data.cc \
    $(CMSIS_DIR)/preprocessor.cc \
    $(CMSIS_EXT_DIR)/arm_cmplx_mag_squared_q10p6.c \
    $(CMSIS_DIR)/hanning.c \
    $(APOLLO3_SDK)/devices/am_devices_led.c \
    $(CMSIS_SRCS)
  ALL_SRCS += $(PUSHBUTTON_CMSIS_SPEECH_TEST_SRCS)
  PUSHBUTTON_CMSIS_SPEECH_TEST_OBJS := $(addprefix $(OBJDIR), \
    $(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(PUSHBUTTON_CMSIS_SPEECH_TEST_SRCS))) \
    arm_bitreversal2.o)
  PUSHBUTTON_CMSIS_SPEECH_TEST_BINARY := $(BINDIR)pushbutton_cmsis_speech_test
  $(PUSHBUTTON_CMSIS_SPEECH_TEST_BINARY): $(PUSHBUTTON_CMSIS_SPEECH_TEST_OBJS) $(MICROLITE_LIB_PATH)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
	-o $(PUSHBUTTON_CMSIS_SPEECH_TEST_BINARY) $(PUSHBUTTON_CMSIS_SPEECH_TEST_OBJS) \
	$(LIBFLAGS) $(MICROLITE_LIB_PATH) $(LDFLAGS) $(MICROLITE_LIBS)
  pushbutton_cmsis_speech_test: $(PUSHBUTTON_CMSIS_SPEECH_TEST_BINARY)
  pushbutton_cmsis_speech_test_bin: $(PUSHBUTTON_CMSIS_SPEECH_TEST_BINARY).bin
  test_pushbutton_cmsis_speech: $(PUSHBUTTON_CMSIS_SPEECH_TEST_BINARY)
	$(TEST_SCRIPT) $(PUSHBUTTON_CMSIS_SPEECH_TEST_BINARY) '~~~ALL TESTS PASSED~~~'

  PREPROCESSOR_1K_SRCS := \
    tensorflow/lite/experimental/micro/examples/micro_speech/apollo3/preprocessor_1k.cc \
    tensorflow/lite/experimental/micro/examples/micro_speech/CMSIS/sin_1k.cc

  PREPROCESSOR_1K_MICRO_TEST_SRCS := \
    $(PREPROCESSOR_1K_SRCS) \
    $(AP3_MICRO_DIR)/../fixed_point/preprocessor.cc \
    $(AP3_EXT_MICRO_DIR)/system_apollo3.c \
    $(AP3_MICRO_DIR)/_main.c
  ALL_SRCS += $(PREPROCESSOR_1K_MICRO_TEST_SRCS)
  PREPROCESSOR_1K_MICRO_TEST_OBJS := $(addprefix $(OBJDIR), \
    $(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(PREPROCESSOR_1K_MICRO_TEST_SRCS))))
  PREPROCESSOR_1K_MICRO_TEST_BINARY := $(BINDIR)preprocessor_1k_micro_test
  $(PREPROCESSOR_1K_MICRO_TEST_BINARY): $(PREPROCESSOR_1K_MICRO_TEST_OBJS) $(MICROLITE_LIB_PATH)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
	-o $(PREPROCESSOR_1K_MICRO_TEST_BINARY) $(PREPROCESSOR_1K_MICRO_TEST_OBJS) \
	$(LIBFLAGS) $(MICROLITE_LIB_PATH) $(LDFLAGS) $(MICROLITE_LIBS)
  preprocessor_1k_micro_test: $(PREPROCESSOR_1K_MICRO_TEST_BINARY)
  preprocessor_1k_micro_test_bin: $(PREPROCESSOR_1K_MICRO_TEST_BINARY).bin
  test_preprocessor_1k_micro: $(PREPROCESSOR_1K_MICRO_TEST_BINARY)
	$(TEST_SCRIPT) $(PREPROCESSOR_1K_MICRO_TEST_BINARY) '~~~ALL TESTS PASSED~~~'

  PREPROCESSOR_1K_CMSIS_TEST_SRCS := \
    $(PREPROCESSOR_1K_SRCS) \
    $(CMSIS_DIR)/preprocessor.cc \
    $(CMSIS_EXT_DIR)/arm_cmplx_mag_squared_q10p6.c \
    $(CMSIS_DIR)/hanning.c \
    $(AP3_EXT_MICRO_DIR)/system_apollo3.c \
    $(AP3_MICRO_DIR)/_main.c \
    $(CMSIS_SRCS)
  ALL_SRCS += $(PREPROCESSOR_1K_CMSIS_TEST_SRCS)
  PREPROCESSOR_1K_CMSIS_TEST_BINARY := $(BINDIR)preprocessor_1k_cmsis_test
  PREPROCESSOR_1K_CMSIS_TEST_OBJS := $(addprefix $(OBJDIR), \
    $(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(PREPROCESSOR_1K_CMSIS_TEST_SRCS)))\
    arm_bitreversal2.o)
  $(PREPROCESSOR_1K_CMSIS_TEST_BINARY): $(PREPROCESSOR_1K_CMSIS_TEST_OBJS) $(MICROLITE_LIB_PATH)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
	-o $(PREPROCESSOR_1K_CMSIS_TEST_BINARY) $(PREPROCESSOR_1K_CMSIS_TEST_OBJS) \
	$(LIBFLAGS) $(MICROLITE_LIB_PATH) $(LDFLAGS) $(MICROLITE_LIBS)
  preprocessor_1k_cmsis_test: $(PREPROCESSOR_1K_CMSIS_TEST_BINARY)
  preprocessor_1k_cmsis_test_bin: $(PREPROCESSOR_1K_CMSIS_TEST_BINARY).bin
  test_preprocessor_1k_cmsis: $(PREPROCESSOR_1K_CMSIS_TEST_BINARY)
	$(TEST_SCRIPT) $(PREPROCESSOR_1K_CMSIS_TEST_BINARY) '~~~ALL TESTS PASSED~~~'

  PREPROCESSOR_TEST_SRCS += \
    $(AP3_MICRO_DIR)/_main.c 

  $(OBJDIR)arm_bitreversal2.o:
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $(CMSIS_SRC_DIR)/TransformFunctions/arm_bitreversal2.S -o $(OBJDIR)arm_bitreversal2.o

endif
