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

#include "tensorflow/lite/delegates/gpu/cl/gl_interop.h"

#include "absl/strings/str_cat.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_errors.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_call.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_sync.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {

#ifndef EGL_VERSION_1_5
typedef void* EGLSync;
#define EGL_SYNC_CL_EVENT 0x30FE
#define EGL_CL_EVENT_HANDLE 0x309C
#define EGL_NO_SYNC 0
#endif /* EGL_VERSION_1_5 */

// TODO(b/131897059): replace with 64 version when EGL 1.5 is available.
// it should use KHR_cl_event2 extension. More details are in b/129974818.
using PFNEGLCREATESYNCPROC = EGLSync(EGLAPIENTRYP)(
    EGLDisplay dpy, EGLenum type, const EGLAttrib* attrib_list);

PFNEGLCREATESYNCPROC g_eglCreateSync = nullptr;

}  // namespace

Status CreateEglSyncFromClEvent(cl_event event, EGLDisplay display,
                                EglSync* sync) {
  if (!IsEglSyncFromClEventSupported()) {
    return UnimplementedError("CreateEglSyncFromClEvent is not supported");
  }
  EGLSync egl_sync;
  const EGLAttrib attributes[] = {EGL_CL_EVENT_HANDLE,
                                  reinterpret_cast<EGLAttrib>(event), EGL_NONE};
  RETURN_IF_ERROR(TFLITE_GPU_CALL_EGL(g_eglCreateSync, &egl_sync, display,
                                      EGL_SYNC_CL_EVENT, attributes));
  if (egl_sync == EGL_NO_SYNC) {
    return InternalError("Returned empty EGL sync");
  }
  *sync = EglSync(display, egl_sync);
  return OkStatus();
}

bool IsEglSyncFromClEventSupported() {
  // In C++11, static initializers are guaranteed to be evaluated only once.
  static bool supported = []() -> bool {
    // This function requires EGL 1.5 to work
    g_eglCreateSync = reinterpret_cast<PFNEGLCREATESYNCPROC>(
        eglGetProcAddress("eglCreateSync"));
    // eglQueryString accepts EGL_NO_DISPLAY only starting EGL 1.5
    if (!eglQueryString(EGL_NO_DISPLAY, EGL_EXTENSIONS)) {
      g_eglCreateSync = nullptr;
    }
    return (g_eglCreateSync != nullptr);
  }();
  return supported;
}

Status CreateClEventFromEglSync(cl_context context, const EglSync& egl_sync,
                                CLEvent* event) {
  cl_int error_code;
  cl_event new_event = clCreateEventFromEGLSyncKHR(
      context, egl_sync.sync(), egl_sync.display(), &error_code);
  if (error_code != CL_SUCCESS) {
    return InternalError(
        absl::StrCat("Unable to create CL sync from EGL sync. ",
                     CLErrorCodeToString(error_code)));
  }
  *event = CLEvent(new_event);
  return OkStatus();
}

bool IsClEventFromEglSyncSupported(const CLDevice& device) {
  return device.SupportsExtension("cl_khr_egl_event");
}

Status CreateClMemoryFromGlBuffer(GLuint gl_ssbo_id, AccessType access_type,
                                  CLContext* context, CLMemory* memory) {
  cl_int error_code;
  auto mem = clCreateFromGLBuffer(context->context(), ToClMemFlags(access_type),
                                  gl_ssbo_id, &error_code);
  if (error_code != CL_SUCCESS) {
    return InternalError(
        absl::StrCat("Unable to acquire CL buffer from GL buffer. ",
                     CLErrorCodeToString(error_code)));
  }
  *memory = CLMemory(mem, true);
  return OkStatus();
}

Status CreateClMemoryFromGlTexture(GLenum texture_target, GLuint texture_id,
                                   AccessType access_type, CLContext* context,
                                   CLMemory* memory) {
  cl_int error_code;
  auto mem =
      clCreateFromGLTexture(context->context(), ToClMemFlags(access_type),
                            texture_target, 0, texture_id, &error_code);
  if (error_code != CL_SUCCESS) {
    return InternalError(
        absl::StrCat("Unable to create CL buffer from GL texture. ",
                     CLErrorCodeToString(error_code)));
  }
  *memory = CLMemory(mem, true);
  return OkStatus();
}

bool IsGlSharingSupported(const CLDevice& device) {
  return clCreateFromGLBuffer && clCreateFromGLTexture &&
         device.SupportsExtension("cl_khr_gl_sharing");
}

AcquiredGlObjects::~AcquiredGlObjects() { Release({}, nullptr).IgnoreError(); }

Status AcquiredGlObjects::Acquire(const std::vector<cl_mem>& memory,
                                  cl_command_queue queue,
                                  const std::vector<cl_event>& wait_events,
                                  CLEvent* acquire_event,
                                  AcquiredGlObjects* objects) {
  if (!memory.empty()) {
    cl_event new_event;
    cl_int error_code = clEnqueueAcquireGLObjects(
        queue, memory.size(), memory.data(), wait_events.size(),
        wait_events.data(), acquire_event ? &new_event : nullptr);
    if (error_code != CL_SUCCESS) {
      return InternalError(absl::StrCat("Unable to acquire GL object. ",
                                        CLErrorCodeToString(error_code)));
    }
    if (acquire_event) {
      *acquire_event = CLEvent(new_event);
    }
    clFlush(queue);
  }
  *objects = AcquiredGlObjects(memory, queue);
  return OkStatus();
}

Status AcquiredGlObjects::Release(const std::vector<cl_event>& wait_events,
                                  CLEvent* release_event) {
  if (queue_ && !memory_.empty()) {
    cl_event new_event;
    cl_int error_code = clEnqueueReleaseGLObjects(
        queue_, memory_.size(), memory_.data(), wait_events.size(),
        wait_events.data(), release_event ? &new_event : nullptr);
    if (error_code != CL_SUCCESS) {
      return InternalError(absl::StrCat("Unable to release GL object. ",
                                        CLErrorCodeToString(error_code)));
    }
    if (release_event) {
      *release_event = CLEvent(new_event);
    }
    clFlush(queue_);
    queue_ = nullptr;
  }
  return OkStatus();
}

GlInteropFabric::GlInteropFabric(EGLDisplay egl_display,
                                 Environment* environment)
    : is_egl_sync_supported_(true),
      is_egl_to_cl_mapping_supported_(
          IsClEventFromEglSyncSupported(environment->device())),
      is_cl_to_egl_mapping_supported_(IsEglSyncFromClEventSupported()),
      egl_display_(egl_display),
      context_(environment->context().context()),
      queue_(environment->queue()->queue()) {}

void GlInteropFabric::RegisterMemory(cl_mem memory) {
  memory_.push_back(memory);
}

void GlInteropFabric::UnregisterMemory(cl_mem memory) {
  auto it = std::find(memory_.begin(), memory_.end(), memory);
  if (it != memory_.end()) {
    memory_.erase(it);
  }
}

Status GlInteropFabric::Start() {
  if (!is_enabled()) {
    return OkStatus();
  }

  // In GL-CL interoperability, we need to make sure GL finished processing of
  // all commands that might affect GL objects. There are a few ways:
  //   a) glFinish
  //      slow, but portable
  //   b) EglSync + ClientWait
  //      faster alternative for glFinish, but still slow as it stalls GPU
  //      pipeline.
  //   c) EglSync->CLEvent or GlSync->CLEvent mapping
  //      Fast, as it allows to map sync to CL event and use it as a dependency
  //      later without stalling GPU pipeline.
  if (is_egl_sync_supported_) {
    EglSync sync;
    RETURN_IF_ERROR(EglSync::NewFence(egl_display_, &sync));
    if (is_egl_to_cl_mapping_supported_) {
      // (c) EglSync->CLEvent or GlSync->CLEvent mapping
      glFlush();
      RETURN_IF_ERROR(
          CreateClEventFromEglSync(context_, sync, &inbound_event_));
    } else {
      // (b) EglSync + ClientWait
      RETURN_IF_ERROR(sync.ClientWait());
    }
  } else {
    // (a) glFinish / GL fence sync
    RETURN_IF_ERROR(gl::GlActiveSyncWait());
  }

  // Acquire all GL objects needed while processing.
  auto make_acquire_wait = [&]() -> std::vector<cl_event> {
    if (inbound_event_.is_valid()) {
      return {inbound_event_.event()};
    }
    return {};
  };
  return AcquiredGlObjects::Acquire(memory_, queue_, make_acquire_wait(),
                                    nullptr, &gl_objects_);
}

Status GlInteropFabric::Finish() {
  if (!is_enabled()) {
    return OkStatus();
  }
  RETURN_IF_ERROR(gl_objects_.Release({}, &outbound_event_));

  // if (is_egl_sync_supported_ && is_cl_to_egl_mapping_supported_) {
  //   EglSync egl_outbound_sync;
  //   RETURN_IF_ERROR(CreateEglSyncFromClEvent(outbound_event_.event(),
  //                                            egl_display_,
  //                                            &egl_outbound_sync));
  //   // Instruct GL pipeline to wait until corresponding CL event is signaled.
  //   RETURN_IF_ERROR(egl_outbound_sync.ServerWait());
  //   glFlush();
  // } else {
  //   // Slower option if proper sync is not supported. It is equivalent to
  //   // clFinish, but, hopefully, faster.
  //   outbound_event_.Wait();
  // }

  // This slow sync is the only working solution right now. We have to debug why
  // above version is not working fast and reliable.
  outbound_event_.Wait();
  return OkStatus();
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
