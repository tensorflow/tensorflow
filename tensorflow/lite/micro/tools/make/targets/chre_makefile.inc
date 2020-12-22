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

# Remove flexbuffers library and detection postprocess kernel from chre build
# due to string dependencies.
EXCLUDED_CC_SRCS := \
  tensorflow/lite/micro/kernels/detection_postprocess.cc \
  tensorflow/lite/micro/kernels/flexbuffers_generated_data.cc

EXCLUDED_TESTS := \
  tensorflow/lite/micro/kernels/detection_postprocess_test.cc

EXCLUDED_HDRS := \
  third_party/flatbuffers/include/flatbuffers/flexbuffers.h

EXCLUDED_KERNEL_HDRS := \
  tensorflow/lite/micro/kernels/flexbuffers_generated_data.h

MICROLITE_CC_KERNEL_SRCS := $(filter-out $(EXCLUDED_CC_SRCS),$(MICROLITE_CC_KERNEL_SRCS))
MICROLITE_TEST_SRCS := $(filter-out $(EXCLUDED_TESTS),$(MICROLITE_TEST_SRCS))
THIRD_PARTY_CC_HDRS := $(filter-out $(EXCLUDED_HDRS),$(THIRD_PARTY_CC_HDRS))
MICROLITE_CC_HDRS := $(filter-out $(EXCLUDED_KERNEL_HDRS),$(MICROLITE_CC_HDRS))
