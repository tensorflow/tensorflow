################################################################################
# Automatically-generated file. Do not edit!
################################################################################


ifeq ($(OS), Windows_NT)
    SHELL:=cmd
endif


VPATH += ../../../../../../../../tensorflow/lite/micro/CEVA_BlueVox/libs;C:\CEVA-ToolBox\V18\BX\cevatools\lib\clang\7.1.0\cevabx1-unknown-unknown-elf\lib\rtlv1.0.0-fp1-dpfp1

-include ../makefile.init

RM := rm -rf

# All of the sources participating in the build are defined here
-include sources.mk
-include src/tensorflow/third_party/kissfft/src/subdir.mk
-include src/tensorflow/lite/micro/memory_planner/subdir.mk
-include src/tensorflow/lite/micro/kernels/subdir.mk
-include src/tensorflow/lite/micro/examples/micro_speech/simple_features/CEVA_BlueVox/subdir.mk
-include src/tensorflow/lite/micro/examples/micro_speech/micro_features/subdir.mk
-include src/tensorflow/lite/micro/examples/micro_speech/CEVA_BlueVox/subdir.mk
-include src/tensorflow/lite/micro/examples/micro_speech/subdir.mk
-include src/tensorflow/lite/micro/subdir.mk
-include src/tensorflow/lite/kernels/internal/subdir.mk
-include src/tensorflow/lite/kernels/subdir.mk
-include src/tensorflow/lite/experimental/microfrontend/lib/subdir.mk
-include src/tensorflow/lite/core/api/subdir.mk
-include src/tensorflow/lite/c/subdir.mk
-include subdir.mk
-include objects.mk

ifneq ($(MAKECMDGOALS),clean)
ifneq ($(strip $(CC_DEPS)),)
-include $(CC_DEPS)
endif
ifneq ($(strip $(C++_DEPS)),)
-include $(C++_DEPS)
endif
ifneq ($(strip $(C_UPPER_DEPS)),)
-include $(C_UPPER_DEPS)
endif
ifneq ($(strip $(CXX_DEPS)),)
-include $(CXX_DEPS)
endif
ifneq ($(strip $(ASM_DEPS)),)
-include $(ASM_DEPS)
endif
ifneq ($(strip $(S_UPPER_DEPS)),)
-include $(S_UPPER_DEPS)
endif
ifneq ($(strip $(C_DEPS)),)
-include $(C_DEPS)
endif
ifneq ($(strip $(CPP_DEPS)),)
-include $(CPP_DEPS)
endif
endif

-include ../makefile.defs

# Add inputs and outputs from these tool invocations to the build variables 

# All Target
all: MICRO_SPEECH_DEMO.elf

# Tool invocations
MICRO_SPEECH_DEMO.elf: $(OBJS) ../BlueVox_18.0.3.ld $(USER_OBJS) $(LIBS)
	@echo 'Building target: $@'
	@echo 'Invoking: CEVA-ELF Linker'
	C:\CEVA-ToolBox\V18\BX/cevatools/bin/ceva-elf-ld -T ../BlueVox_18.0.3.ld --no-relax --no-gc-sections -L../../../../../../../../tensorflow/lite/micro/CEVA_BlueVox/libs -defsym __internal_data_size=512k -defsym __internal_code_size=256k -L"C:/CEVA-ToolBox/V18/BX/cevatools/lib/clang/7.1.0/cevabx1-unknown-unknown-elf/lib/rtlv1.0.0-fp1-dpfp1" $(OBJS) $(USER_OBJS) --start-group $(LIBS) --end-group -o  "MICRO_SPEECH_DEMO.elf"
	@echo 'Finished building target: $@'
	@echo ' '

# Other Targets
clean:
	-$(RM) $(CC_DEPS)$(C++_DEPS)$(EXECUTABLES)$(C_UPPER_DEPS)$(CXX_DEPS)$(OBJS)$(ASM_DEPS)$(S_UPPER_DEPS)$(C_DEPS)$(CPP_DEPS) MICRO_SPEECH_DEMO.elf
	-@echo ' '

.PHONY: all clean dependents

-include ../makefile.targets
