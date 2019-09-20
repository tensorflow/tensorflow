# Copyright (c) 2013 Google, Inc.
#
# This software is provided 'as-is', without any express or implied
# warranty.  In no event will the authors be held liable for any damages
# arising from the use of this software.
# Permission is granted to anyone to use this software for any purpose,
# including commercial applications, and to alter it and redistribute it
# freely, subject to the following restrictions:
# 1. The origin of this software must not be misrepresented; you must not
# claim that you wrote the original software. If you use this software
# in a product, an acknowledgment in the product documentation would be
# appreciated but is not required.
# 2. Altered source versions must be plainly marked as such, and must not be
# misrepresented as being the original software.
# 3. This notice may not be removed or altered from any source distribution.

LOCAL_PATH := $(call my-dir)
FLATBUFFERS_ROOT_DIR := $(LOCAL_PATH)/../../..

# FlatBuffers test
include $(CLEAR_VARS)

# Include the FlatBuffer utility function to generate header files from schemas.
include $(FLATBUFFERS_ROOT_DIR)/android/jni/include.mk

LOCAL_MODULE := FlatBufferSample

# Set up some useful variables to identify schema and output directories and
# schema files.
ANDROID_SAMPLE_GENERATED_OUTPUT_DIR := $(LOCAL_PATH)/gen/include
ANDROID_SAMPLE_SCHEMA_DIR := $(LOCAL_PATH)/schemas
ANDROID_SAMPLE_SCHEMA_FILES := $(ANDROID_SAMPLE_SCHEMA_DIR)/animal.fbs

LOCAL_C_INCLUDES := $(ANDROID_SAMPLE_GENERATED_OUTPUT_DIR)

$(info $(LOCAL_C_INCLUDES))

LOCAL_SRC_FILES := main.cpp

LOCAL_CPPFLAGS := -std=c++11 -fexceptions -Wall -Wno-literal-suffix
LOCAL_LDLIBS := -llog -landroid -latomic
LOCAL_ARM_MODE := arm
LOCAL_STATIC_LIBRARIES := android_native_app_glue flatbuffers

ifeq (,$(ANDROID_SAMPLE_RUN_ONCE))
ANDROID_SAMPLE_RUN_ONCE := 1
$(call flatbuffers_header_build_rules,$(ANDROID_SAMPLE_SCHEMA_FILES),$(ANDROID_SAMPLE_SCHEMA_DIR),$(ANDROID_SAMPLE_GENERATED_OUTPUT_DIR),,$(LOCAL_SRC_FILES))
endif

include $(BUILD_SHARED_LIBRARY)

# Path to Flatbuffers root directory.
$(call import-add-path,$(FLATBUFFERS_ROOT_DIR)/..)

$(call import-module,flatbuffers/android/jni)
$(call import-module,android/native_app_glue)
