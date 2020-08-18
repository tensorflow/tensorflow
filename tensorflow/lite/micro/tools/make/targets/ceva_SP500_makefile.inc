
ifeq ($(TARGET), ceva)
ifeq ($(TARGET_ARCH), SP500)
 #TARGET_ARCH := 
CXXFLAGS := -std=c++11 -DTF_LITE_STATIC_MEMORY
CCFLAGS  := -std=c11   -DTF_LITE_STATIC_MEMORY


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
endif
