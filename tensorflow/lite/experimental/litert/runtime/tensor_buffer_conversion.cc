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

#include "tensorflow/lite/experimental/litert/runtime/tensor_buffer_conversion.h"

#include "absl/strings/str_format.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_tensor_buffer_utils.h"

#if LITERT_HAS_OPENGL_SUPPORT && LITERT_HAS_AHWB_SUPPORT
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl31.h>
#include <GLES3/gl32.h>

#include <cstring>

#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"
#include "tensorflow/lite/experimental/litert/runtime/ahwb_buffer.h"
#include "tensorflow/lite/experimental/litert/runtime/gl_buffer.h"

#endif  // LITERT_HAS_OPENGL_SUPPORT && LITERT_HAS_AHWB_SUPPORT
#include "tensorflow/lite/experimental/litert/runtime/tensor_buffer.h"

namespace litert {
namespace internal {

#if LITERT_HAS_OPENGL_SUPPORT && LITERT_HAS_AHWB_SUPPORT
Expected<void> CopyGlToAhwb(GlBuffer& src, AhwbBuffer& dest) {
  if (src.target() != GL_SHADER_STORAGE_BUFFER) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Unsupported GL target for conversion to AHWB");
  }
  LITERT_ASSIGN_OR_RETURN(size_t ahwb_size, AhwbBuffer::GetSize(dest.ahwb));
  if (src.bytes_size() != ahwb_size) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "GL buffer size does not match AHWB size");
  }
  LITERT_ASSIGN_OR_RETURN(void* host_src, src.Lock<char>());
  LITERT_ASSIGN_OR_RETURN(void* host_dest, AhwbBuffer::Lock(dest.ahwb));
  std::memcpy(host_dest, host_src, src.bytes_size());
  LITERT_RETURN_IF_ERROR(AhwbBuffer::Unlock(dest.ahwb));
  LITERT_RETURN_IF_ERROR(src.Unlock<char>());
  return {};
}

Expected<LiteRtTensorBufferT::Ptr> TensorBufferConvertGlToAhwb(
    LiteRtTensorBufferT& tensor_buffer_gl) {
  // Create a new AHWB tensor buffer.
  LITERT_ASSIGN_OR_RETURN(
      LiteRtTensorBufferT::Ptr tensor_buffer_ahwb,
      LiteRtTensorBufferT::CreateManaged(kLiteRtTensorBufferTypeAhwb,
                                         tensor_buffer_gl.tensor_type(),
                                         tensor_buffer_gl.buffer_size()));
  LITERT_ASSIGN_OR_RETURN(AHardwareBuffer * ahwb,
                          tensor_buffer_ahwb->GetAhwbBuffer());
  AhwbBuffer ahwb_buffer{.ahwb = ahwb};
  LITERT_ASSIGN_OR_RETURN(GlBuffer * gl_buffer, tensor_buffer_gl.GetGlBuffer());
  CopyGlToAhwb(*gl_buffer, ahwb_buffer);
  return tensor_buffer_ahwb;
}
#endif  // LITERT_HAS_OPENGL_SUPPORT && LITERT_HAS_AHWB_SUPPORT

Expected<LiteRtTensorBufferT::Ptr> TensorBufferConvertTo(
    LiteRtTensorBufferType buffer_type, LiteRtTensorBufferT& tensor_buffer) {
  if (tensor_buffer.buffer_type() == kLiteRtTensorBufferTypeGlBuffer &&
      buffer_type == kLiteRtTensorBufferTypeAhwb) {
#if LITERT_HAS_OPENGL_SUPPORT && LITERT_HAS_AHWB_SUPPORT
    return TensorBufferConvertGlToAhwb(tensor_buffer);
#else
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Gl -> Ahwb conversion is not supported.");
#endif
  }
  return Unexpected(
      kLiteRtStatusErrorRuntimeFailure,
      absl::StrFormat("This buffer conversion is not supported: %s -> %s",
                      BufferTypeToString(tensor_buffer.buffer_type()),
                      BufferTypeToString(buffer_type)));
}

}  // namespace internal
}  // namespace litert
