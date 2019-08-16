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

#include <vector>

#include <EGL/egl.h>
#include <EGL/eglext.h>
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

namespace tflite {
namespace gpu {
namespace cl {

// Creates an EglSync from OpenCL event. Source event does not need to outlive
// returned sync and could be safely destroyed.
//
// Depends on EGL 1.5.
Status CreateEglSyncFromClEvent(cl_event event, EGLDisplay display,
                                EglSync* sync);

// Returns true if 'CreateEglSyncFromClEvent' is supported.
bool IsEglSyncFromClEventSupported();

// Creates CL event from EGL sync.
// Created event could only be comsumed by AcquiredGlObject::Acquire call as
// a 'wait_event'.
Status CreateClEventFromEglSync(cl_context context, const EglSync& egl_sync,
                                CLEvent* event);

// Returns true if 'CreateClEventFromEglSync' is supported.
bool IsClEventFromEglSyncSupported(const CLDevice& device);

// Creates new CL memory object from OpenGL buffer.
Status CreateClMemoryFromGlBuffer(GLuint gl_ssbo_id, AccessType access_type,
                                  CLContext* context, CLMemory* memory);

// Creates new CL memory object from OpenGL texture.
Status CreateClMemoryFromGlTexture(GLenum texture_target, GLuint texture_id,
                                   AccessType access_type, CLContext* context,
                                   CLMemory* memory);

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
  static Status Acquire(const std::vector<cl_mem>& memory,
                        cl_command_queue queue,
                        const std::vector<cl_event>& wait_events,
                        CLEvent* acquire_event /* optional */,
                        AcquiredGlObjects* objects);

  // Releases OpenCL memory back to OpenGL context. If 'release_event' is not
  // nullptr, it will be signalled once release is complete.
  Status Release(const std::vector<cl_event>& wait_events,
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
  Status Start();

  // Puts appropriate CL->GL synchronization after all work is complete.
  Status Finish();

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

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_GL_INTEROP_H_
