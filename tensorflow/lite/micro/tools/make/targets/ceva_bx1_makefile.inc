
ifeq ($(TARGET), ceva)
ifeq ($(TARGET_ARCH), bx1)
 #TARGET_ARCH := 
CXXFLAGS := -std=c++11 -DTF_LITE_STATIC_MEMORY
CCFLAGS  := -std=c11   -DTF_LITE_STATIC_MEMORY


  PLATFORM_ARGS = \
    -c \
	-fmessage-length=0 \
	-fpermissive \
	-Os \
	-g3 \
	-Wall \
	-pedantic \
	-D_LIBCPP_INLINE_VISIBILITY="" \
	-D_LIBCPP_EXTERN_TEMPLATE_INLINE_VISIBILITY="" \
	--target=cevabx1-elf \
	-mcpu=cevabx1v1.0.0 \
	-m32x32 \
	-mgetbits \
	-mloop-buffer-size=10 \
	-mfp=1 \
	-mdpfp=1

  TARGET_TOOLCHAIN_PREFIX := ceva
  CXX_TOOL := clang++
  CC_TOOL := clang
  LD_TOOL := ceva-elf-ld
  LD := ceva-elf-ld

  CXXFLAGS += $(PLATFORM_ARGS)
  CCFLAGS += $(PLATFORM_ARGS)
  LDFLAGS += \
	  -T \
	CEVA_TFLM.ld \
	--no-relax \
	--no-gc-sections \
	-defsym \
	__internal_data_size=512k \
	-defsym \
	__internal_code_size=256k \
#-L/home/yaire/CEVA-ToolBox/V18/BX/cevatools/lib/clang/7.1.0/cevabx1-unknown-unknown-elf/lib/rtlv1.0.0-fp1-dpfp1 \
#	-lc++ -lc++abi -lc -lcompiler-rt
    

endif
endif
