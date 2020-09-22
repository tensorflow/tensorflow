
TARGET_TOOLCHAIN_ROOT := /home/yaire/CEVA-ToolBox/V18.02/BX
#CC = %{CC_TOOL}%
#CXX = %{CXX_TOOL}%
#LD = %{LD_TOOL}%
CC = ${TARGET_TOOLCHAIN_ROOT}/cevatools/bin/clang
CXX = ${TARGET_TOOLCHAIN_ROOT}/cevatools/bin/clang++
LD = ${TARGET_TOOLCHAIN_ROOT}/cevatools/bin/ceva-elf-ld
AS = ${TARGET_TOOLCHAIN_ROOT}/cevatools/bin/ceva-elf-as
TOOLS_OBJS := \
${TARGET_TOOLCHAIN_ROOT}/cevatools/lib/clang/4.0.1/cevabx1-unknown-unknown-elf/lib/rtlv1.0.0-fp1-dpfp1/crt0.o ${TARGET_TOOLCHAIN_ROOT}/cevatools/lib/clang/4.0.1/cevabx1-unknown-unknown-elf/lib/rtlv1.0.0-fp1-dpfp1/crtn.o
TOOLS_LIBS := \
-lc++ -lc++abi -lc -lcompiler-rt

  LDFLAGS += \
	  -T \
	../../../../../targets/ceva/CEVA_BX1_TFLM_18.0.2.ld \
	--no-relax \
	--no-gc-sections \
	-defsym \
	__internal_data_size=512k \
	-defsym \
	__internal_code_size=256k \
	-L${TARGET_TOOLCHAIN_ROOT}/cevatools/lib/clang/4.0.1/cevabx1-unknown-unknown-elf/lib/rtlv1.0.0-fp1-dpfp1 \
	-lc++ -lc++abi -lc -lcompiler-rt
    

OUT_NAME = %{EXECUTABLE}%

CXXFLAGS += %{CXX_FLAGS}%
CCFLAGS += %{CC_FLAGS}%

#=============================================================
# Files and directories
#=============================================================
SRCS := \
%{SRCS}%

OBJS := \
$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(SRCS)))


#=============================================================
# Common rules
#=============================================================
.PHONY: all app flash clean run debug

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(EXT_CFLAGS) $(INCLUDES) -c $< -o $@

%.o: %.c
	$(CC) $(CCFLAGS) $(EXT_CFLAGS) $(INCLUDES) -c $< -o $@

$(OUT_NAME): $(OBJS)
	$(LD)  -o $@ $(OBJS) $(TOOLS_OBJS) ${TOOLS_LIBS} $(LDFLAGS)

%{EXTRA_APP_RULES}%


#=================================================================
# Global rules
#=================================================================
all: $(OUT_NAME)

app: $(OUT_NAME)

clean: 
	-@$(RM) $(call fix_platform_path,$(OBJS))
	-@$(RM) $(OUT_NAME) %{EXTRA_RM_TARGETS}%


