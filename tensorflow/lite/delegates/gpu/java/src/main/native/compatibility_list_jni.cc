/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include <jni.h>

#include <memory>

#include "absl/status/status.h"
#include "tensorflow/lite/delegates/gpu/common/gpu_info.h"
#include "tensorflow/lite/delegates/gpu/gl/egl_environment.h"
#include "tensorflow/lite/delegates/gpu/gl/request_gpu_info.h"
#include "tensorflow/lite/experimental/acceleration/compatibility/android_info.h"
#include "tensorflow/lite/experimental/acceleration/compatibility/gpu_compatibility.h"
#include "tensorflow/lite/java/src/main/native/jni_utils.h"

extern "C" {

namespace {
class CompatibilityListHelper {
 public:
  CompatibilityListHelper()
      : compatibility_list_(
            tflite::acceleration::GPUCompatibilityList::Create()) {}
  absl::Status ReadInfo() {
    auto status = tflite::acceleration::RequestAndroidInfo(&android_info_);
    if (!status.ok()) return status;

    if (android_info_.android_sdk_version < "21") {
      // Weakly linked symbols may not be available on pre-21, and the GPU is
      // not supported anyway so return early.
      return absl::OkStatus();
    }

    std::unique_ptr<tflite::gpu::gl::EglEnvironment> env;
    status = tflite::gpu::gl::EglEnvironment::NewEglEnvironment(&env);
    if (!status.ok()) return status;

    status = tflite::gpu::gl::RequestGpuInfo(&gpu_info_);
    if (!status.ok()) return status;

    return absl::OkStatus();
  }

  bool IsDelegateSupportedOnThisDevice() {
    return compatibility_list_->Includes(android_info_, gpu_info_);
  }

 private:
  tflite::acceleration::AndroidInfo android_info_;
  tflite::gpu::GpuInfo gpu_info_;
  std::unique_ptr<tflite::acceleration::GPUCompatibilityList>
      compatibility_list_;
};
}  // namespace

JNIEXPORT jlong JNICALL
Java_org_tensorflow_lite_gpu_CompatibilityList_createCompatibilityList(
    JNIEnv* env, jclass clazz) {
  if (!tflite::jni::CheckJniInitializedOrThrow(env)) return 0;

  CompatibilityListHelper* compatibility_list = new CompatibilityListHelper;
  auto status = compatibility_list->ReadInfo();
  // Errors in ReadInfo should almost always be failures to construct the OpenGL
  // environment. Treating that as "GPU unsupported" is reasonable, and we can
  // swallow the error.
  status.IgnoreError();
  return reinterpret_cast<jlong>(compatibility_list);
}

JNIEXPORT jboolean JNICALL
Java_org_tensorflow_lite_gpu_CompatibilityList_nativeIsDelegateSupportedOnThisDevice(
    JNIEnv* env, jclass clazz, jlong compatibility_list_handle) {
  if (!tflite::jni::CheckJniInitializedOrThrow(env)) return JNI_FALSE;

  CompatibilityListHelper* compatibility_list =
      reinterpret_cast<CompatibilityListHelper*>(compatibility_list_handle);
  return compatibility_list->IsDelegateSupportedOnThisDevice() ? JNI_TRUE
                                                               : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_org_tensorflow_lite_gpu_CompatibilityList_deleteCompatibilityList(
    JNIEnv* env, jclass clazz, jlong compatibility_list_handle) {
  if (!tflite::jni::CheckJniInitializedOrThrow(env)) return;

  CompatibilityListHelper* compatibility_list =
      reinterpret_cast<CompatibilityListHelper*>(compatibility_list_handle);
  delete compatibility_list;
}

}  // extern "C"
