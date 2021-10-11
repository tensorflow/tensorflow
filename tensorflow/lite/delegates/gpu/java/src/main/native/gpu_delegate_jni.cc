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

#include <jni.h>

#include <memory>

#include "absl/status/status.h"
#include "tensorflow/lite/delegates/gpu/common/gpu_info.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"
#include "tensorflow/lite/delegates/gpu/gl/egl_environment.h"
#include "tensorflow/lite/delegates/gpu/gl/request_gpu_info.h"
#include "tensorflow/lite/experimental/acceleration/compatibility/android_info.h"
#include "tensorflow/lite/experimental/acceleration/compatibility/gpu_compatibility.h"

extern "C" {

JNIEXPORT jlong JNICALL Java_org_tensorflow_lite_gpu_GpuDelegate_createDelegate(
    JNIEnv* env, jclass clazz, jboolean precision_loss_allowed,
    jboolean quantized_models_allowed, jint inference_preference,
    jstring serialization_dir, jstring model_token) {
  TfLiteGpuDelegateOptionsV2 options = TfLiteGpuDelegateOptionsV2Default();
  if (precision_loss_allowed == JNI_TRUE) {
    options.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY;
    options.inference_priority2 =
        TFLITE_GPU_INFERENCE_PRIORITY_MIN_MEMORY_USAGE;
    options.inference_priority3 = TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION;
  }
  options.experimental_flags = TFLITE_GPU_EXPERIMENTAL_FLAGS_NONE;
  if (quantized_models_allowed) {
    options.experimental_flags |= TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_QUANT;
  }
  options.inference_preference = static_cast<int32_t>(inference_preference);
  if (serialization_dir) {
    options.serialization_dir =
        env->GetStringUTFChars(serialization_dir, /*isCopy=*/nullptr);
  }
  if (model_token) {
    options.model_token =
        env->GetStringUTFChars(model_token, /*isCopy=*/nullptr);
  }
  if (options.serialization_dir && options.model_token) {
    options.experimental_flags |=
        TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_SERIALIZATION;
  }
  return reinterpret_cast<jlong>(TfLiteGpuDelegateV2Create(&options));
}

JNIEXPORT void JNICALL Java_org_tensorflow_lite_gpu_GpuDelegate_deleteDelegate(
    JNIEnv* env, jclass clazz, jlong delegate) {
  TfLiteGpuDelegateV2Delete(reinterpret_cast<TfLiteDelegate*>(delegate));
}

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
  CompatibilityListHelper* compatibility_list =
      reinterpret_cast<CompatibilityListHelper*>(compatibility_list_handle);
  return compatibility_list->IsDelegateSupportedOnThisDevice() ? JNI_TRUE
                                                               : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_org_tensorflow_lite_gpu_CompatibilityList_deleteCompatibilityList(
    JNIEnv* env, jclass clazz, jlong compatibility_list_handle) {
  CompatibilityListHelper* compatibility_list =
      reinterpret_cast<CompatibilityListHelper*>(compatibility_list_handle);
  delete compatibility_list;
}

}  // extern "C"
