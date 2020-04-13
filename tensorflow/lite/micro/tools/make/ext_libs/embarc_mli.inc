ifeq ($(TARGET_ARCH), arc)

# embarc_mli Library is used by default for ARC platform whenever it's possible.
# To use TFLM reference implementation it should be intentionally turned off 
# by passing 'no_embarc_mli' tag (make -f <tflm_main_makefile> TAGS=no_embarc_mli ...)
ifeq ($(filter no_embarc_mli,$(ALL_TAGS)),)


ALL_TAGS += embarc_mli

ifeq ($(PRE_COMPILED_MLI),true)
  # TODO: Replace with proper embarc_mli pre-builts.
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
  MICROLITE_LIBS += $(MAKEFILE_DIR)/downloads/$(MLI_LIB_DIR)/bin/libmli.a

  THIRD_PARTY_CC_HDRS += \
    third_party/$(MLI_LIB_DIR)/LICENSE
endif

  THIRD_PARTY_CC_HDRS += $(MLI_LIB)
  GENERATED_PROJECT_LIBS += $(MLI_LIB)

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
    third_party/$(MLI_INCLUDE_FOLDER)/api/mli_mov_api.h

    MICROLITE_CC_HDRS += tensorflow/lite/micro/kernels/embarc_mli/scratch_buffers.h
    MICROLITE_CC_SRCS += tensorflow/lite/micro/kernels/embarc_mli/scratch_buffers.cc
    MICROLITE_CC_HDRS += tensorflow/lite/micro/kernels/embarc_mli/scratch_buf_mgr.h
    MICROLITE_CC_SRCS += tensorflow/lite/micro/kernels/embarc_mli/scratch_buf_mgr.cc
    MICROLITE_CC_HDRS += tensorflow/lite/micro/kernels/embarc_mli/mli_slicers.h
    MICROLITE_CC_SRCS += tensorflow/lite/micro/kernels/embarc_mli/mli_slicers.cc
    MICROLITE_CC_HDRS += tensorflow/lite/micro/kernels/embarc_mli/mli_tf_utils.h

endif # no_embarc_mli
endif # TARGET_ARCH
