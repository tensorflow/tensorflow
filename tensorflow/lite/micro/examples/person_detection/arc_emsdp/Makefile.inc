ifeq ($(TARGET), arc_emsdp)

  person_detection_HDRS += \
  person_detection_patch.txt
  
  person_detection_TEST_HDRS += \
  person_detection_patch.txt
  

%/person_detection_patch.txt: %/emsdp.lcf %/Makefile
	@cp tensorflow/lite/micro/examples/person_detection/arc_emsdp/emsdp.lcf $< 
	@echo emsdp.lcf > $@
	@sed -E -i 's#MLI_ONLY *\?= *false#MLI_ONLY \?= false\n\
	CXXFLAGS += -DSCRATCH_MEM_X_SIZE=0 -DSCRATCH_MEM_Y_SIZE=0 -DSCRATCH_MEM_Z_SIZE=0\
	CCFLAGS += -DSCRATCH_MEM_X_SIZE=0 -DSCRATCH_MEM_Y_SIZE=0 -DSCRATCH_MEM_Z_SIZE=0#'\
	  $(word 2, $^)
	@echo Makefile >> $@

endif
