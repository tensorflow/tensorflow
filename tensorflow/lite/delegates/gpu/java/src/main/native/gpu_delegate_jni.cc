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

#include "tensorflow/lite/java/src/main/native/jni_utils.h"

#if TFLITE_DISABLE_SELECT_JAVA_APIS
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/experimental/acceleration/configuration/c/gpu_plugin.h"
#include "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h"
#else
#include "tensorflow/lite/delegates/gpu/delegate.h"
#include "tensorflow/lite/delegates/gpu/delegate_options.h"
#endif

#if TFLITE_DISABLE_SELECT_JAVA_APIS
using flatbuffers::FlatBufferBuilder;
using flatbuffers::Offset;
using flatbuffers::String;
using tflite::GPUSettings;
using tflite::GPUSettingsBuilder;
using tflite::TFLiteSettings;
using tflite::TFLiteSettingsBuilder;
#else
constexpr int kGpuBackendOpenCl = 1;
constexpr int kGpuBackendOpenGl = 2;
#endif

extern "C" {

JNIEXPORT jlong JNICALL Java_org_tensorflow_lite_gpu_GpuDelegate_createDelegate(
    JNIEnv* env, jclass clazz, jboolean precision_loss_allowed,
    jboolean quantized_models_allowed, jint inference_preference,
    jstring serialization_dir, jstring model_token, jint force_backend) {
  if (!tflite::jni::CheckJniInitializedOrThrow(env)) return 0;

#if TFLITE_DISABLE_SELECT_JAVA_APIS
  // Use GPU Delegate via Delegate Plugin API.
  // This approach would work for the !TFLITE_DISABLE_SELECT_JAVA_APIS case too,
  // but may have a slightly higher overhead due to the need to construct
  // a FlatBuffer for the configuration parameters.

  // Construct a FlatBuffer that contains the following:
  //   TFLiteSettings {
  //     GpuSettings {
  //       cache_directory: <serialization_dir>,
  //       model_token: <model_token>,
  //       precision_loss_allowed: <precision_loss_allowed>,
  //       enable_quantized_inference: <quantized_models_allowed>,
  //       inference_preference: <inference_preference>,
  //     }
  //   }
  // where the values in angle brackets are the parameters to this function,
  // except that we only set the fields if they have non-default values.
  FlatBufferBuilder flatbuffer_builder;
  Offset<String> serialization_dir_fb_string = 0;
  if (serialization_dir) {
    const char* serialization_dir_c_string =
        env->GetStringUTFChars(serialization_dir, nullptr);
    serialization_dir_fb_string =
        flatbuffer_builder.CreateString(serialization_dir_c_string);
    env->ReleaseStringUTFChars(serialization_dir, serialization_dir_c_string);
  }
  Offset<String> model_token_fb_string = 0;
  if (model_token) {
    const char* model_token_c_string =
        env->GetStringUTFChars(model_token, nullptr);
    model_token_fb_string =
        flatbuffer_builder.CreateString(model_token_c_string);
    env->ReleaseStringUTFChars(model_token, model_token_c_string);
  }
  GPUSettingsBuilder gpu_settings_builder(flatbuffer_builder);
  if (serialization_dir) {
    gpu_settings_builder.add_cache_directory(serialization_dir_fb_string);
  }
  if (model_token) {
    gpu_settings_builder.add_model_token(model_token_fb_string);
  }
  if (precision_loss_allowed) {
    gpu_settings_builder.add_is_precision_loss_allowed(precision_loss_allowed);
  }
  if (quantized_models_allowed) {
    gpu_settings_builder.add_enable_quantized_inference(
        quantized_models_allowed);
  }
  if (inference_preference) {
    gpu_settings_builder.add_inference_preference(
        static_cast<tflite::GPUInferenceUsage>(inference_preference));
  }
  if (force_backend) {
    gpu_settings_builder.add_force_backend(
        static_cast<tflite::GPUBackend>(force_backend));
  }
  Offset<GPUSettings> gpu_settings = gpu_settings_builder.Finish();
  TFLiteSettingsBuilder tflite_settings_builder(flatbuffer_builder);
  tflite_settings_builder.add_gpu_settings(gpu_settings);
  flatbuffers::Offset<TFLiteSettings> tflite_settings =
      tflite_settings_builder.Finish();
  flatbuffer_builder.Finish(tflite_settings);
  const TFLiteSettings* settings = flatbuffers::GetRoot<TFLiteSettings>(
      flatbuffer_builder.GetBufferPointer());

  // Construct the delegate using the Delegate Plugin C API,
  // and passing in the flatbuffer settings that we constructed above.
  TfLiteOpaqueDelegate* gpu_delegate =
      TfLiteGpuDelegatePluginCApi()->create(settings);
  return reinterpret_cast<jlong>(gpu_delegate);
#else
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
  if (force_backend == kGpuBackendOpenCl) {
    options.experimental_flags |= TFLITE_GPU_EXPERIMENTAL_FLAGS_CL_ONLY;
  } else if (force_backend == kGpuBackendOpenGl) {
    options.experimental_flags |= TFLITE_GPU_EXPERIMENTAL_FLAGS_GL_ONLY;
  }
  return reinterpret_cast<jlong>(TfLiteGpuDelegateV2Create(&options));
#endif  // TFLITE_DISABLE_SELECT_JAVA_APIS
}

JNIEXPORT void JNICALL Java_org_tensorflow_lite_gpu_GpuDelegate_deleteDelegate(
    JNIEnv* env, jclass clazz, jlong delegate) {
  if (!tflite::jni::CheckJniInitializedOrThrow(env)) return;

#if TFLITE_DISABLE_SELECT_JAVA_APIS
  TfLiteGpuDelegatePluginCApi()->destroy(
      reinterpret_cast<TfLiteOpaqueDelegate*>(delegate));
#else
  TfLiteGpuDelegateV2Delete(reinterpret_cast<TfLiteDelegate*>(delegate));
#endif
}

}  // extern "C"
