// Copyright 2025 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/lite/experimental/litert/cc/litert_tensor_buffer_utils.h"

#include <string>

#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer_types.h"

std::string BufferTypeToString(LiteRtTensorBufferType buffer_type) {
  switch (buffer_type) {
    case kLiteRtTensorBufferTypeUnknown:
      return "Unknown";
    case kLiteRtTensorBufferTypeHostMemory:
      return "HostMemory";
    case kLiteRtTensorBufferTypeAhwb:
      return "Ahwb";
    case kLiteRtTensorBufferTypeIon:
      return "Ion";
    case kLiteRtTensorBufferTypeDmaBuf:
      return "DmaBuf";
    case kLiteRtTensorBufferTypeFastRpc:
      return "FastRpc";
    case kLiteRtTensorBufferTypeOpenCl:
      return "OpenCl";
    case kLiteRtTensorBufferTypeGlBuffer:
      return "GlBuffer";
    case kLiteRtTensorBufferTypeGlTexture:
      return "GlTexture";
  }
  LITERT_LOG(LITERT_ERROR, "Unexpected value for LiteRtTensorBufferType: %d",
             static_cast<int>(buffer_type));
  return "UnexpectedBufferType";
}
