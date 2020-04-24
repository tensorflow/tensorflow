ifeq ($(TARGET), arc_emsdp)

  person_detection_HDRS += \
  person_detection_int8_patch.txt
  
  person_detection_TEST_HDRS += \
  person_detection_int8_patch.txt
  

%/person_detection_int8_patch.txt: %/emsdp.lcf %/Makefile
	@cp tensorflow/lite/micro/examples/person_detection_experimental/arc_emsdp/emsdp.lcf $< 
	@echo emsdp.lcf > $@
	@sed -E -i 's#MLI_ONLY *\?= *false#MLI_ONLY \?= true#' $(word 2, $^)
	@echo Makefile > $@

endif
