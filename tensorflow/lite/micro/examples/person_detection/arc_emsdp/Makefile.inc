ifeq ($(TARGET), arc_emsdp)

#Patch of arc make project to adjust it specifically
#for experimental person detection example.In particular:
# - Use Linker command file with better usage of fast memory
#- Stripout TFLM reference code by default.
#- Optional : replace mli switchers with specialized kernels
#for smaller code size

  person_detection_HDRS += \
  person_detection_int8_patch.txt
  
  person_detection_TEST_HDRS += \
  person_detection_int8_patch.txt
  
  ARC_MLI_BACKEND_PATH = /tensorflow/lite/micro/kernels/arc_mli

#Apply changes in generated project files.
#See related comment echoed(@echo <comment>) after each change
#to get understanding on it's purpose.
%/person_detection_int8_patch.txt: %/emsdp.lcf %/Makefile %$(ARC_MLI_BACKEND_PATH)/conv.cc %$(ARC_MLI_BACKEND_PATH)/depthwise_conv.cc %$(ARC_MLI_BACKEND_PATH)/pooling.cc
	@cp tensorflow/lite/micro/examples/person_detection/arc_emsdp/emsdp.lcf $< 
	@echo emsdp.lcf: Replace with example specific memory map  > $@

	@sed -E -i 's#MLI_ONLY *\?= *false#MLI_ONLY \?= true#' $(word 2, $^)
	@echo Makefile: No Reference fallback for MLI supported functions >> $@

ifneq ($(filter $(ARC_TAGS), reduce_codesize),)
#In case 'reduce_codesize' tag is present, we replace common MLI functions with
#specializations appropriate for this particular graph.But such changes of code
#with high probability may not be acceptable for other graphs and will need
#to be adjusted by the user

	@sed -E -i 's#mli_krn_conv2d_nhwc_sa8_sa8_sa32#mli_krn_conv2d_nhwc_sa8_sa8_sa32_k1x1_nopad#' $(word 3, $^)
	@sed -E -i 's#mli_krn_depthwise_conv2d_hwcn_sa8_sa8_sa32#mli_krn_depthwise_conv2d_hwcn_sa8_sa8_sa32_k3x3_krnpad#' $(word 4, $^)
	@sed -E -i 's#mli_krn_avepool_hwc_sa8#mli_krn_avepool_hwc_sa8_k3x3_nopad#' $(word 5, $^)
	@sed -E -i 's#mli_krn_maxpool_hwc_sa8\(in_ptr, \&cfg, out_ptr\);#return kTfLiteError;#' $(word 5, $^)
	@echo $(word 3, $^): Use specialization >> $@
	@echo $(word 4, $^): Use specialization >> $@
	@echo $(word 5, $^): Use specialization and remove max pooling >> $@
endif

endif
