ifeq ($(BOARD),sparkfun_edge)
INCLUDES+= -I$(APOLLO3_SDK)/$(SF_BSPS_DEST)/common/third_party/lis2dh12
MICROLITE_CC_SRCS+= \
  $(APOLLO3_SDK)/$(SF_BSPS_DEST)/common/third_party/lis2dh12/lis2dh12_platform_apollo3.c \
  $(APOLLO3_SDK)/$(SF_BSPS_DEST)/common/third_party/lis2dh12/lis2dh12_reg.c
endif
