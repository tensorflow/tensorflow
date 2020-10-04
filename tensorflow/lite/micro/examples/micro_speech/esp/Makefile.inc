# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ifeq ($(TARGET), esp)

# Adding some esp specific files in the main CMakeLists.txt
ESP_MICRO_SPEECH_SRCS := \
tensorflow/lite/micro/examples/micro_speech/esp/ringbuf.c

ESP_MICRO_SPEECH_HDRS := \
tensorflow/lite/micro/examples/micro_speech/esp/ringbuf.h

CCFLAGS += -Wno-nonnull
CXXFLAGS += -Wno-return-type -Wno-strict-aliasing -Wno-ignored-qualifiers
MICRO_SPEECH_SRCS += $(ESP_MICRO_SPEECH_SRCS)
MICRO_SPEECH_HDRS += $(ESP_MICRO_SPEECH_HDRS)
MAIN_SRCS += $(ESP_MICRO_SPEECH_SRCS)

micro_speech_ESP_PROJECT_FILES := \
sdkconfig.defaults

endif
