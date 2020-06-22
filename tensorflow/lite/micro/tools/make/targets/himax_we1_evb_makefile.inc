# Settings for himax WE_1 evb.
ifeq ($(TARGET), himax_we1_evb)
  
  CC_TOOL = ccac
  AR_TOOL = arac
  CXX_TOOL = ccac
  LD_TOOL := ccac
  TARGET_ARCH := arc
  #ARC_TOOLCHAIN := mwdt 

  BUILD_ARC_MLI := false
  ARC_MLI_PRE_COMPILED_TARGET := himax_arcem9d_r16
  
  include $(MAKEFILE_DIR)/targets/arc/arc_common.inc

  #download SDK & MLI
  HIMAX_WE1_SDK_NAME := himax_we1_sdk
  $(eval $(call add_third_party_download,$(HIMAX_WE1_SDK_URL),$(HIMAX_WE1_SDK_MD5),$(HIMAX_WE1_SDK_NAME),))

  #export path of toolchain
  #export PATH := $(MAKEFILE_DIR)/downloads/$(HIMAX_WE1_SDK_NAME)/image_gen_linux_v3/:$(PATH)
  
  TCF_FILE := $(PWD)/$(MAKEFILE_DIR)/downloads/$(HIMAX_WE1_SDK_NAME)/arcem9d_wei_r16.tcf
  LCF_FILE := $(PWD)/$(MAKEFILE_DIR)/downloads/$(HIMAX_WE1_SDK_NAME)/memory.lcf
  ARCLIB_FILE :=  $(PWD)/$(MAKEFILE_DIR)/downloads/$(HIMAX_WE1_SDK_NAME)/libembarc.a
  LIB_HEADER_FILE :=  $(PWD)/$(MAKEFILE_DIR)/downloads/$(HIMAX_WE1_SDK_NAME)/hx_drv_tflm.h
  

  DEFAULT_HEAPSZ := 8192
  DEFAULT_STACKSZ := 8192

  TCF_FILE_NAME = $(notdir $(TCF_FILE))
  ARC_TARGET_FILES_DIRS = $(dir $(TCF_FILE_NAME))
  MAKE_PROJECT_FILES += $(TCF_FILE_NAME)
    
  LCF_FILE_NAME = $(notdir $(LCF_FILE))
  ARC_TARGET_FILES_DIRS += $(dir $(LCF_FILE))
  MAKE_PROJECT_FILES += $(LCF_FILE_NAME)
  
  ARCLIB_FILE_NAME = $(notdir $(ARCLIB_FILE))
  ARC_TARGET_FILES_DIRS += $(dir $(ARCLIB_FILE))
  MAKE_PROJECT_FILES += $(ARCLIB_FILE_NAME)
  
  LIB_HEADER_FILE_NAME = $(notdir $(LIB_HEADER_FILE))
  ARC_TARGET_FILES_DIRS += $(dir $(LIB_HEADER_FILE))
  MAKE_PROJECT_FILES += $(LIB_HEADER_FILE_NAME)
  
  
    
  # Need a pointer to the TCF and lcf file

  PLATFORM_FLAGS = \
    -DNDEBUG \
    -g \
    -DCPU_ARC \
    -Hnosdata \
    -DTF_LITE_STATIC_MEMORY \
    -tcf=$(TCF_FILE_NAME) \
    -Hnocopyr \
    -Hpurge \
    -Hcl \
    -fslp-vectorize-aggressive \
    -ffunction-sections \
    -fdata-sections \
    -tcf_core_config \

  CXXFLAGS += -fno-rtti -DSCRATCH_MEM_Z_SIZE=0x10000 $(PLATFORM_FLAGS)
  CCFLAGS += $(PLATFORM_FLAGS)

  INCLUDES+= \
    -I $(MAKEFILE_DIR)/downloads/$(WEI_SDK_NAME) \
    -I $(MAKEFILE_DIR)/downloads/kissfft

  GENERATED_PROJECT_INCLUDES += \
    -I. \
    -I./third_party/kissfft

  LDFLAGS += \
    -Hheap=8192 \
    -tcf=$(TCF_FILE_NAME) \
    -Hnocopyr \
    -m \
    -Hldopt=-Coutput=$(TARGET).map \
    $(LCF_FILE_NAME) \
    -Hldopt=-Bgrouplib $(ARCLIB_FILE_NAME)

  CXXFLAGS := $(filter-out -std=c++11,$(CXXFLAGS))
  CCFLAGS := $(filter-out -std=c11,$(CCFLAGS))
  MICROLITE_LIBS := $(filter-out -lm,$(MICROLITE_LIBS))

endif
