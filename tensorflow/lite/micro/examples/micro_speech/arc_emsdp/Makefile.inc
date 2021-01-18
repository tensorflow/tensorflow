ifeq ($(TARGET), arc_emsdp)

# Patch of arc make project to adjust it specifically for micro speech example. 
# In particular:
# - Extend Heap and stack size for application needs
# - Use Linker command file with better usage of fast memory
# - Optional (TAGS=reduce_codesize): In case project was 
#   generated with MLI usage, reduce scratch buffers.

  MICRO_SPEECH_HDRS += \
  micro_speech_patch.txt
  
  MICRO_SPEECH_TEST_HDRS += \
  micro_speech_patch.txt
  
  MICRO_SPEECH_MOCK_HDRS += \
  micro_speech_patch.txt

  ARC_MLI_BACKEND_PATH = /tensorflow/lite/micro/kernels/arc_mli

# Apply changes in generated project files. 
# See related comment echoed (@echo <comment>) after each change 
# to get understanding on it's purpose.
%/micro_speech_patch.txt: %/emsdp.lcf %/Makefile %$(ARC_MLI_BACKEND_PATH)/depthwise_conv.cc
	@cp tensorflow/lite/micro/examples/micro_speech/arc_emsdp/emsdp.lcf $< 
	@echo emsdp.lcf: Example specific memory map > $@
	
	@sed -E -i 's#-Hheap=[^ ]*#\-Hheap=16K \-Hstack=16K#g' $(word 2, $^)
	@echo Makefile: Set exact heap/stack size >> $@

	@sed -E -i 's#MLI_ONLY *\?= *false#MLI_ONLY \?= true\n\
	CXXFLAGS += -DSCRATCH_MEM_X_SIZE=0 -DSCRATCH_MEM_Y_SIZE=0 -DSCRATCH_MEM_Z_SIZE=0\
	CCFLAGS += -DSCRATCH_MEM_X_SIZE=0 -DSCRATCH_MEM_Y_SIZE=0 -DSCRATCH_MEM_Z_SIZE=0#'\
	  $(word 2, $^)
	@echo Makefile: set scratch buffers size to 0 >> $@
	@echo Makefile: No Reference fallback for MLI supported functions >> $@


ifneq ($(filter $(ALL_TAGS), reduce_codesize),)
# In case 'reduce_codesize' tag is present, we replace common MLI functions with 
# specializations appropriate for this particular graph. But such changes of code 
# with high probability may not be acceptable for other graphs and will need 
# to be adjusted by the user

	@sed -E -i 's#mli_krn_depthwise_conv2d_hwcn_sa8_sa8_sa32\(# \
	mli_krn_depthwise_conv2d_hwcn_sa8_sa8_sa32_generic\(#g' $(word 3, $^)
	@echo $(word 3, $^): Use generic function >> $@
endif

endif
