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
#include <type_traits>

#if TFLITE_DISABLE_SELECT_JAVA_APIS
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/core/shims/c/common.h"
#include "tensorflow/lite/core/shims/c/experimental/acceleration/configuration/nnapi_plugin.h"
#include "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h"
#else
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#endif

#if TFLITE_DISABLE_SELECT_JAVA_APIS
using flatbuffers::FlatBufferBuilder;
using tflite::NNAPISettings;
using tflite::NNAPISettingsBuilder;
using tflite::TFLiteSettings;
using tflite::TFLiteSettingsBuilder;
#else
using tflite::StatefulNnApiDelegate;
#endif

extern "C" {

JNIEXPORT jlong JNICALL
Java_org_tensorflow_lite_nnapi_NnApiDelegateImpl_createDelegate(
    JNIEnv* env, jclass clazz, jint preference, jstring accelerator_name,
    jstring cache_dir, jstring model_token, jint max_delegated_partitions,
    jboolean override_disallow_cpu, jboolean disallow_cpu_value,
    jboolean allow_fp16, jlong nnapi_support_library_handle) {
#if TFLITE_DISABLE_SELECT_JAVA_APIS
  // Use NNAPI Delegate via Delegate Plugin API.
  // This approach would work for the !TFLITE_DISABLE_SELECT_JAVA_APIS case too,
  // but may have a slightly higher overhead due to the need to construct
  // a FlatBuffer for the configuration parameters.

  // Construct a FlatBuffer that contains the following:
  //   TFLiteSettings {
  //     NnapiSettings {
  //       accelerator_name : <accelerator_name>,
  //       cache_directory : <cache_dir>,
  //       model_token : <model_token>,
  //       allow_nnapi_cpu_on_android_10_plus: !<disallow_cpu_value>,
  //       allow_fp16_precision_for_fp32: <allow_fp16>,
  //       support_library_handle: <nnapi_support_library_handle>,
  //     }
  //     max_delegate_partitions: <max_delegated_partitions>
  //   }
  // where the values in angle brackets are the parameters to this function,
  // except that we set the 'allow_nnapi_cpu_on_android_10_plus' field only if
  // <override_disallow_cpu> is true, and that we only set the other fields
  // if they have non-default values.
  FlatBufferBuilder flatbuffer_builder;
  flatbuffers::Offset<flatbuffers::String> accelerator_name_fb_string = 0;
  if (accelerator_name) {
    const char* accelerator_name_c_string =
        env->GetStringUTFChars(accelerator_name, nullptr);
    accelerator_name_fb_string =
        flatbuffer_builder.CreateString(accelerator_name_c_string);
    env->ReleaseStringUTFChars(accelerator_name, accelerator_name_c_string);
  }
  flatbuffers::Offset<flatbuffers::String> cache_directory_fb_string = 0;
  if (cache_dir) {
    const char* cache_directory_c_string =
        env->GetStringUTFChars(cache_dir, nullptr);
    cache_directory_fb_string =
        flatbuffer_builder.CreateString(cache_directory_c_string);
    env->ReleaseStringUTFChars(cache_dir, cache_directory_c_string);
  }
  flatbuffers::Offset<flatbuffers::String> model_token_fb_string = 0;
  if (model_token) {
    const char* model_token_c_string =
        env->GetStringUTFChars(model_token, nullptr);
    model_token_fb_string =
        flatbuffer_builder.CreateString(model_token_c_string);
    env->ReleaseStringUTFChars(model_token, model_token_c_string);
  }
  NNAPISettingsBuilder nnapi_settings_builder(flatbuffer_builder);
  nnapi_settings_builder.add_execution_preference(
      static_cast<tflite::NNAPIExecutionPreference>(preference));
  if (accelerator_name) {
    nnapi_settings_builder.add_accelerator_name(accelerator_name_fb_string);
  }
  if (cache_dir) {
    nnapi_settings_builder.add_cache_directory(cache_directory_fb_string);
  }
  if (model_token) {
    nnapi_settings_builder.add_model_token(model_token_fb_string);
  }
  if (override_disallow_cpu) {
    nnapi_settings_builder.add_allow_nnapi_cpu_on_android_10_plus(
        !disallow_cpu_value);
  }
  if (allow_fp16) {
    nnapi_settings_builder.add_allow_fp16_precision_for_fp32(allow_fp16);
  }
  if (nnapi_support_library_handle) {
    nnapi_settings_builder.add_support_library_handle(
        nnapi_support_library_handle);
  }
  flatbuffers::Offset<NNAPISettings> nnapi_settings =
      nnapi_settings_builder.Finish();
  TFLiteSettingsBuilder tflite_settings_builder(flatbuffer_builder);
  tflite_settings_builder.add_nnapi_settings(nnapi_settings);
  if (max_delegated_partitions >= 0) {
    tflite_settings_builder.add_max_delegated_partitions(
        max_delegated_partitions);
  }
  flatbuffers::Offset<TFLiteSettings> tflite_settings =
      tflite_settings_builder.Finish();
  flatbuffer_builder.Finish(tflite_settings);
  const TFLiteSettings* settings = flatbuffers::GetRoot<TFLiteSettings>(
      flatbuffer_builder.GetBufferPointer());

  // Construct the delegate using the Delegate Plugin C API,
  // and passing in the flatbuffer settings that we constructed above.
  TfLiteOpaqueDelegate* nnapi_delegate =
      TfLiteNnapiDelegatePluginCApi()->create(settings);
  return reinterpret_cast<jlong>(nnapi_delegate);
#else
  // Use NNAPI Delegate directly.

  // Construct an Options object for the parameter settings.
  StatefulNnApiDelegate::Options options = StatefulNnApiDelegate::Options();
  options.execution_preference =
      (StatefulNnApiDelegate::Options::ExecutionPreference)preference;
  if (accelerator_name) {
    options.accelerator_name =
        env->GetStringUTFChars(accelerator_name, nullptr);
  }
  if (cache_dir) {
    options.cache_dir = env->GetStringUTFChars(cache_dir, nullptr);
  }
  if (model_token) {
    options.model_token = env->GetStringUTFChars(model_token, nullptr);
  }
  if (max_delegated_partitions >= 0) {
    options.max_number_delegated_partitions = max_delegated_partitions;
  }
  if (override_disallow_cpu) {
    options.disallow_nnapi_cpu = disallow_cpu_value;
  }
  if (allow_fp16) {
    options.allow_fp16 = allow_fp16;
  }
  // Construct the delegate, using the options object constructed earlier.
  auto delegate =
      nnapi_support_library_handle
          ? new StatefulNnApiDelegate(reinterpret_cast<NnApiSLDriverImplFL5*>(
                                          nnapi_support_library_handle),
                                      options)
          : new StatefulNnApiDelegate(options);
  // Deallocate temporary strings.
  if (options.accelerator_name) {
    env->ReleaseStringUTFChars(accelerator_name, options.accelerator_name);
  }
  if (options.cache_dir) {
    env->ReleaseStringUTFChars(cache_dir, options.cache_dir);
  }
  if (options.model_token) {
    env->ReleaseStringUTFChars(model_token, options.model_token);
  }
  return reinterpret_cast<jlong>(delegate);
#endif
}

JNIEXPORT jint JNICALL
Java_org_tensorflow_lite_nnapi_NnApiDelegateImpl_getNnapiErrno(JNIEnv* env,
                                                               jclass clazz,
                                                               jlong delegate) {
#if TFLITE_DISABLE_SELECT_JAVA_APIS
  TfLiteOpaqueDelegate* nnapi_delegate =
      reinterpret_cast<TfLiteOpaqueDelegate*>(delegate);
  return TfLiteNnapiDelegatePluginCApi()->get_delegate_errno(nnapi_delegate);
#else
  StatefulNnApiDelegate* nnapi_delegate =
      reinterpret_cast<StatefulNnApiDelegate*>(delegate);
  return nnapi_delegate->GetNnApiErrno();
#endif
}

JNIEXPORT void JNICALL
Java_org_tensorflow_lite_nnapi_NnApiDelegateImpl_deleteDelegate(
    JNIEnv* env, jclass clazz, jlong delegate) {
#if TFLITE_DISABLE_SELECT_JAVA_APIS
  TfLiteOpaqueDelegate* nnapi_delegate =
      reinterpret_cast<TfLiteOpaqueDelegate*>(delegate);
  TfLiteNnapiDelegatePluginCApi()->destroy(nnapi_delegate);
#else
  delete reinterpret_cast<StatefulNnApiDelegate*>(delegate);
#endif
}

}  // extern "C"
