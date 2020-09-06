/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_GL_INTEROP_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_GL_INTEROP_H_

#include <EGL/egl.h>
#include <EGL/eglext.h>

#include <vector>

#include "tensorflow/lite/delegates/gpu/cl/cl_command_queue.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_context.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_device.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_event.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_memory.h"
#include "tensorflow/lite/delegates/gpu/cl/egl_sync.h"
#include "tensorflow/lite/delegates/gpu/cl/environment.h"
#include "tensorflow/lite/delegates/gpu/cl/opencl_wrapper.h"
#include "tensorflow/lite/delegates/gpu/common/access_type.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/gl/portable_gl31.h"
#include "tensorflow/lite/delegates/gpu/spi.h"

namespace tflite {
namespace gpu {
namespace cl {

// Creates an EglSync from OpenCL event. Source event does not need to outlive
// returned sync and could be safely destroyed.
//
// Depends on EGL 1.5.
absl::Status CreateEglSyncFromClEvent(cl_event event, EGLDisplay display,
                                      EglSync* sync);

// Returns true if 'CreateEglSyncFromClEvent' is supported.
bool IsEglSyncFromClEventSupported();

// Creates CL event from EGL sync.
// Created event could only be consumed by AcquiredGlObject::Acquire call as
// a 'wait_event'.
absl::Status CreateClEventFromEglSync(cl_context context,
                                      const EglSync& egl_sync, CLEvent* event);

// Returns true if 'CreateClEventFromEglSync' is supported.
bool IsClEventFromEglSyncSupported(const CLDevice& device);

// Creates new CL memory object from OpenGL buffer.
absl::Status CreateClMemoryFromGlBuffer(GLuint gl_ssbo_id,
                                        AccessType access_type,
                                        CLContext* context, CLMemory* memory);

// Creates new CL memory object from OpenGL texture.
absl::Status CreateClMemoryFromGlTexture(GLenum texture_target,
                                         GLuint texture_id,
                                         AccessType access_type,
                                         CLContext* context, CLMemory* memory);

// Returns true if GL objects could be shared with OpenCL context.
bool IsGlSharingSupported(const CLDevice& device);

// RAII-wrapper for GL objects acquired into CL context.
class AcquiredGlObjects {
 public:
  static bool IsSupported(const CLDevice& device);

  AcquiredGlObjects() : AcquiredGlObjects({}, nullptr) {}

  // Quitely releases OpenGL objects. It is recommended to call Release()
  // explicitly to properly handle potential errors.
  ~AcquiredGlObjects();

  // Acquires memory from the OpenGL context. Memory must be created by either
  // CreateClMemoryFromGlBuffer or CreateClMemoryFromGlTexture calls.
  // If 'acquire_event' is not nullptr, it will be signared once acquisition is
  // complete.
  static absl::Status Acquire(const std::vector<cl_mem>& memory,
                              cl_command_queue queue,
                              const std::vector<cl_event>& wait_events,
                              CLEvent* acquire_event /* optional */,
                              AcquiredGlObjects* objects);

  // Releases OpenCL memory back to OpenGL context. If 'release_event' is not
  // nullptr, it will be signalled once release is complete.
  absl::Status Release(const std::vector<cl_event>& wait_events,
                       CLEvent* release_event /* optional */);

 private:
  AcquiredGlObjects(const std::vector<cl_mem>& memory, cl_command_queue queue)
      : memory_(memory), queue_(queue) {}

  std::vector<cl_mem> memory_;
  cl_command_queue queue_;
};

// Incapsulates all complicated GL-CL synchronization. It manages life time of
// all appropriate events to ensure fast synchronization whenever possible.
class GlInteropFabric {
 public:
  GlInteropFabric(EGLDisplay egl_display, Environment* environment);

  // Ensures proper GL->CL synchronization is in place before
  // GL objects that are mapped to CL objects are used.
  absl::Status Start();

  // Puts appropriate CL->GL synchronization after all work is complete.
  absl::Status Finish();

  // Registers memory to be used from GL context. Such CL memory object must
  // be created with CreateClMemoryFromGlBuffer or CreateClMemoryFromGlTexture
  // call.
  void RegisterMemory(cl_mem memory);

  // Unregisters memory registered with RegisterMemory call.
  void UnregisterMemory(cl_mem memory);

 private:
  bool is_enabled() const { return egl_display_ && !memory_.empty(); }

  bool is_egl_sync_supported_;
  bool is_egl_to_cl_mapping_supported_;
  bool is_cl_to_egl_mapping_supported_;

  const EGLDisplay egl_display_;
  cl_context context_;
  cl_command_queue queue_;
  CLEvent inbound_event_;
  CLEvent outbound_event_;
  std::vector<cl_mem> memory_;
  AcquiredGlObjects gl_objects_;  // transient during Start/Finish calls.
};

// Copies data from(to) GL buffer to(from) CL buffer using CPU.
class GlClBufferCopier : public TensorObjectConverter {
 public:
  static bool IsSupported(const ObjectDef& input, const ObjectDef& output) {
    return input.data_type == output.data_type &&
           input.data_layout == output.data_layout &&
           ((input.object_type == ObjectType::OPENGL_SSBO &&
             output.object_type == ObjectType::OPENCL_BUFFER) ||
            (input.object_type == ObjectType::OPENCL_BUFFER &&
             output.object_type == ObjectType::OPENGL_SSBO));
  }

  GlClBufferCopier(const TensorObjectDef& input_def,
                   const TensorObjectDef& output_def, Environment* environment);

  absl::Status Convert(const TensorObject& input_obj,
                       const TensorObject& output_obj) override;

 private:
  size_t size_in_bytes_;
  CLCommandQueue* queue_ = nullptr;
};

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_GL_INTEROP_H_
