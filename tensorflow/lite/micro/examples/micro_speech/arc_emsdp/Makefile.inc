ifeq ($(TARGET), arc_emsdp)

  MICRO_SPEECH_HDRS += \
  micro_speech_patch.txt
  
  MICRO_SPEECH_TEST_HDRS += \
  micro_speech_patch.txt
  
  MICRO_SPEECH_MOCK_HDRS += \
  micro_speech_patch.txt

%/micro_speech_patch.txt: %/emsdp.lcf %/Makefile
	@cp tensorflow/lite/micro/examples/person_detection/arc_emsdp/emsdp.lcf $< 
	@echo emsdp.lcf > $@
	@sed -E -i 's#-Hheap=[^ ]*#\-Hheap=16K \-Hstack=16K#g' $(word 2, $^)
	@sed -E -i 's#MLI_ONLY *\?= *false#MLI_ONLY \?= false\n\
	CXXFLAGS += -DSCRATCH_MEM_X_SIZE=0 -DSCRATCH_MEM_Y_SIZE=0 -DSCRATCH_MEM_Z_SIZE=0\
	CCFLAGS += -DSCRATCH_MEM_X_SIZE=0 -DSCRATCH_MEM_Y_SIZE=0 -DSCRATCH_MEM_Z_SIZE=0#'\
	  $(word 2, $^)
	@echo Makefile >> $@

endif
