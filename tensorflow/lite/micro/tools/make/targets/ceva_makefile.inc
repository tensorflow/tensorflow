
ifeq ($(TARGET), ceva)

TARGET_ARCH :=

ifeq ($(TARGET_ARCH), )
  $(error TARGET_ARCH must be specified on the command line)

endif# Create a cflag based on the specified TARGET_ARCH. For example:
#   TARGET_ARCH=hifimini --> -DHIFIMINI
#   TARGET_ARCH=fusion_f1 --> -DFUSION_F1
TARGET_ARCH_DEFINES := -D$(shell echo $(TARGET_ARCH) | tr [a-z] [A-Z])

ifeq ($(TARGET_ARCH), CEVA_BX1)
 #TARGET_ARCH := 
CXXFLAGS := -std=c++11 -DTF_LITE_STATIC_MEMORY -I./tensorflow/lite/micro/kernels/ceva
CCFLAGS  := -std=c11   -DTF_LITE_STATIC_MEMORY -I./tensorflow/lite/micro/kernels/ceva


  PLATFORM_ARGS = \
    $(TARGET_ARCH_DEFINES) \
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

ifeq ($(TARGET_ARCH), CEVA_SP500)
#stuff for SP
PLATFORM_ARGS = \
 -c \
 -fmessage-length=0 \
 -Os \
 -g \
 -Wall \
 -pedantic \
 -Wa,--no-rstr-check \
 -D_LIBCPP_INLINE_VISIBILITY="" -DTF_LITE_STATIC_MEMORY \
 -D_LIBCPP_EXTERN_TEMPLATE_INLINE_VISIBILITY="" \
 --target=senspro-elf \
 -mcpu=sensprov1.0.0 \
 -mvu=1 \
 -mno-vld2 \
 -mvmpyv5 \
 -mvmpyext -mnonlinear=1 -mno-vbnn -mvhist \
 -mlvu=1 \
 -mfp=2 \
 -mdpfp=2 \
 -mvfp=1

  TARGET_TOOLCHAIN_PREFIX := ceva
  CXX_TOOL := clang++
  CC_TOOL := clang
  LD_TOOL := ceva-elf-ld
  LD := ceva-elf-ld

  CXXFLAGS += $(PLATFORM_ARGS)
  CCFLAGS += $(PLATFORM_ARGS)
  LDFLAGS += \
--no-relax --no-gc-sections \
 -defsym __internal_code_size=0k \
 -defsym __internal_data_size=512k 
 
endif


MICROLITE_CC_HDRS += \
  tensorflow/lite/micro/kernels/ceva/CEVA_TFLM_lib.h \
  tensorflow/lite/micro/kernels/ceva/types.h \
  tensorflow/lite/micro/kernels/ceva/CEVA_basic_op.h

MICROLITE_CC_SRCS += \
  tensorflow/lite/micro/kernels/ceva/ceva_common.cc


endif
