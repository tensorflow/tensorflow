# Settings for not pre-defined ARC processors. 
# User need to specify ARC target with Tool Configuration File (*.tcf). 
# Path to this file must be passed through TCF_FILE variable.
# Otherwise, default em7d_voice_audio configuration is used 

ifeq ($(TARGET_ARCH), arc)

# Known target are specifyed with their own make configurations. 
ifeq ($(filter $(TARGET), emsdp iotdk),)

ARC_TOOLCHAIN := mwdt

ifneq ($(TCF_FILE), )
  TARGET = $(basename $(notdir $(TCF_FILE)))
else
  $(warning TCF_FILE variable is not specifyed. Use default em7d_voice_audio configuration)
  TARGET = em7d_voice_audio
  TCF_FILE = em7d_voice_audio
endif

include $(MAKEFILE_DIR)/targets/arc/arc_common.inc

endif  # $(TARGET)
endif  # $(TARGET_ARCH)...

