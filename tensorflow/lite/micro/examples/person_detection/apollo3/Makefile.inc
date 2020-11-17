# all apollo3 hm01b0 implementations use these sources
MICROLITE_CC_SRCS+= \
	tensorflow/lite/micro/examples/person_detection/himax_driver/HM01B0.c \
	tensorflow/lite/micro/examples/person_detection/himax_driver/HM01B0_optimized.c

# target specific
ifeq ($(BOARD),apollo3evb)
$(error "apollo3evb requires an hm01b0 driver pin map (platform.h) - please implement")
endif
ifeq ($(BOARD),sparkfun_edge)
	INCLUDES+= -I$(APOLLO3_SDK)/$(SF_BSPS_DEST)/common/third_party/hm01b0
endif
