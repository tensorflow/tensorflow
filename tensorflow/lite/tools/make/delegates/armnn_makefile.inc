# Setting to enable ArmNN delegate
BUILD_WITH_ARMNN ?= false

ifeq ($(BUILD_WITH_ARMNN),true)
	# Add delegate files to core files
	TF_LITE_CC_SRCS += $(wildcard tensorflow/lite/delegates/armnn/*.cc)

	# Extend flags
	CXXFLAGS += --std=c++14

	# Add includes
	INCLUDES += \
		-I$(ARMNN_DIR) \
		-I$(ARMNN_DIR)/src \
		-I$(ARMNN_DIR)/src/armnn \
		-I$(ARMNN_DIR)/src/backends \
		-I$(ARMNN_DIR)/include \
		-I$(BOOST_DIR)

	# Add libs
	LIBS += -L$(ARMNN_BUILD_PATH)
	LIBS += -larmnn
endif
