ifeq ($(TARGET), arc_emsdp)

# Patch of arc make project to adjust it specifically 
# for person detection example. In particular:
# - Use Linker command file with better usage of fast memory
# - In case project was generated with MLI usage, reduce scratch buffers.

  person_detection_HDRS += \
  person_detection_patch.txt
  
  person_detection_TEST_HDRS += \
  person_detection_patch.txt
  

%/person_detection_patch.txt: %/emsdp.lcf %/Makefile
	@cp tensorflow/lite/micro/tools/make/targets/arc/emsdp/emsdp_v2.lcf $< 
	@echo emsdp.lcf > $@
	@sed -E -i 's#MLI_ONLY *\?= *false#MLI_ONLY \?= false\n\
	CXXFLAGS += -DSCRATCH_MEM_X_SIZE=0 -DSCRATCH_MEM_Y_SIZE=0 -DSCRATCH_MEM_Z_SIZE=0\
	CCFLAGS += -DSCRATCH_MEM_X_SIZE=0 -DSCRATCH_MEM_Y_SIZE=0 -DSCRATCH_MEM_Z_SIZE=0#'\
	  $(word 2, $^)
	@echo Makefile >> $@

endif
