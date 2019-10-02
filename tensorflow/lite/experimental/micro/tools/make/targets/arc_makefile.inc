# Settings for arc processors
ifeq ($(TARGET_ARCH), arc)
  CC = ccac
  CXX = ccac
  LD = ccac

ifneq ($(TCF_FILE), )
  TARGET = $(basename $(notdir $(TCF_FILE)))
else
  TARGET = em7d_voice_audio
  TCF_FILE = em7d_voice_audio 
endif

# The variable TCF_FILE_NAME stores the TCF file name (including .tcf extension), this variable is used later to add the option to the linker/compiler flags.
# This condition also handles the case when the user/makefile specifies the configuration bundled with MWDT (usually without .tcf extension) and that doesn't require copying.
ifneq (,$(findstring .tcf,$(TCF_FILE)))
  TCF_FILE_NAME = $(notdir $(TCF_FILE))
  THIRD_PARTY_CC_HDRS += $(TCF_FILE_NAME)
else
  TCF_FILE_NAME = $(TCF_FILE)
endif

  PLATFORM_FLAGS = -tcf=$(TCF_FILE_NAME) -Hnocopyr -O3 -Hpurge -Hcl -fslp-vectorize-aggressive -ffunction-sections -fdata-sections
  PLATFORM_LDFLAGS = -tcf=$(TCF_FILE_NAME) -Hnocopyr -m -Hldopt=-Coutput=memory.map

  CXXFLAGS += $(PLATFORM_FLAGS)
  CXXFLAGS:=$(filter-out -std=c++11,$(CXXFLAGS))
  CCFLAGS += $(PLATFORM_FLAGS)
  LDFLAGS += $(PLATFORM_LDFLAGS)

  MICROLITE_LIBS := $(filter-out -lm,$(MICROLITE_LIBS))

  USE_EMBARC_MLI ?= true

ifeq ($(USE_EMBARC_MLI), true)
  ALL_TAGS += arc

ifeq ($(PRE_COMPILED_MLI),true)
  $(eval $(call add_third_party_download,$(EMBARC_OSP_URL),$(EMBARC_OSP_MD5),embarc_osp,))

  MLI_INCLUDE_FOLDER = embarc_osp/library/embarc_mli/include
  MLI_LIB = third_party/embarc_osp/library/embarc_mli/lib/arcem9d/libmli_iotdk.a

  THIRD_PARTY_CC_HDRS += \
    third_party/embarc_osp/LICENSE
else
  MLI_LIB_DIR = embarc_mli_$(basename $(TCF_FILE_NAME))

  $(eval $(call add_third_party_download,$(EMBARC_MLI_URL),$(EMBARC_MLI_MD5),$(MLI_LIB_DIR),build_embarc_mli,$(TCF_FILE)))

  MLI_INCLUDE_FOLDER = $(MLI_LIB_DIR)/include
  MLI_LIB = third_party/$(MLI_LIB_DIR)/bin/libmli.a
  
  THIRD_PARTY_CC_HDRS += \
    third_party/$(MLI_LIB_DIR)/LICENSE
endif

  THIRD_PARTY_CC_HDRS += $(MLI_LIB)
  MICROLITE_LIBS += $(MLI_LIB)

  INCLUDES += \
    -I$(MAKEFILE_DIR)/downloads/$(MLI_INCLUDE_FOLDER) \
    -I$(MAKEFILE_DIR)/downloads/$(MLI_INCLUDE_FOLDER)/api

  GENERATED_PROJECT_INCLUDES += \
    -I. \
    -I./third_party/$(MLI_INCLUDE_FOLDER) \
    -I./third_party/$(MLI_INCLUDE_FOLDER)/api


  THIRD_PARTY_CC_HDRS += \
    third_party/$(MLI_INCLUDE_FOLDER)/mli_api.h \
    third_party/$(MLI_INCLUDE_FOLDER)/mli_config.h \
    third_party/$(MLI_INCLUDE_FOLDER)/mli_types.h \
    third_party/$(MLI_INCLUDE_FOLDER)/api/mli_helpers_api.h \
    third_party/$(MLI_INCLUDE_FOLDER)/api/mli_kernels_api.h \
    third_party/$(MLI_INCLUDE_FOLDER)/api/mli_krn_avepool_spec_api.h \
    third_party/$(MLI_INCLUDE_FOLDER)/api/mli_krn_conv2d_spec_api.h \
    third_party/$(MLI_INCLUDE_FOLDER)/api/mli_krn_depthwise_conv2d_spec_api.h \
    third_party/$(MLI_INCLUDE_FOLDER)/api/mli_krn_maxpool_spec_api.h \

endif # USE_EMBARC_MLI

endif
