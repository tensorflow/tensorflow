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

#include "tensorflow/lite/tools/evaluation/utils.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/string.h"  // from @flatbuffers
#include "tensorflow/lite/tools/delegates/delegate_provider.h"
#include "tensorflow/lite/tools/logging.h"

#if defined(__APPLE__)
#include "TargetConditionals.h"
#if (TARGET_OS_IPHONE && !TARGET_IPHONE_SIMULATOR) || \
    (TARGET_OS_OSX && TARGET_CPU_ARM64)
// Only enable coreml delegate when using a real iPhone device or Apple Silicon.
#define REAL_IPHONE_DEVICE
#include "tensorflow/lite/delegates/coreml/coreml_delegate.h"
#endif
#endif

#ifndef TFLITE_WITHOUT_XNNPACK
#include "tensorflow/lite/acceleration/configuration/c/delegate_plugin.h"
#include "tensorflow/lite/acceleration/configuration/c/xnnpack_plugin.h"
#include "tensorflow/lite/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#endif  // !defined(TFLITE_WITHOUT_XNNPACK)

#if !defined(_WIN32)
#include <dirent.h>
#endif
#include <sys/stat.h>

namespace tflite {
namespace evaluation {

std::string StripTrailingSlashes(const std::string& path) {
  int end = path.size();
  while (end > 0 && path[end - 1] == '/') {
    end--;
  }
  return path.substr(0, end);
}

bool ReadFileLines(const std::string& file_path,
                   std::vector<std::string>* lines_output) {
  if (!lines_output) {
    return false;
  }
  std::ifstream stream(file_path.c_str());
  if (!stream) {
    return false;
  }
  std::string line;
  while (std::getline(stream, line)) {
    lines_output->push_back(line);
  }
  return true;
}

#if !defined(_WIN32)
TfLiteStatus GetSortedFileNames(
    const std::string& directory, std::vector<std::string>* result,
    const std::unordered_set<std::string>& extensions) {
  DIR* dir;
  struct dirent* ent;
  if (result == nullptr) {
    return kTfLiteError;
  }
  result->clear();
  std::string dir_path = StripTrailingSlashes(directory);
  if ((dir = opendir(dir_path.c_str())) != nullptr) {
    while ((ent = readdir(dir)) != nullptr) {
      if (ent->d_type == DT_DIR) continue;
      std::string filename(std::string(ent->d_name));
      size_t lastdot = filename.find_last_of('.');
      std::string ext = lastdot != std::string::npos ? filename.substr(lastdot)
                                                     : std::string();
      std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
      if (!extensions.empty() && extensions.find(ext) == extensions.end()) {
        continue;
      }
      result->emplace_back(dir_path + "/" + filename);
    }
    closedir(dir);
  } else {
    return kTfLiteError;
  }
  std::sort(result->begin(), result->end());
  return kTfLiteOk;
}
#endif

TfLiteDelegatePtr CreateNNAPIDelegate() {
#if TFLITE_SUPPORTS_NNAPI_DELEGATE
  return TfLiteDelegatePtr(
      NnApiDelegate(),
      // NnApiDelegate() returns a singleton, so provide a no-op deleter.
      [](TfLiteDelegate*) {});
#else   // TFLITE_SUPPORTS_NNAPI_DELEGATE
  return tools::CreateNullDelegate();
#endif  // TFLITE_SUPPORTS_NNAPI_DELEGATE
}

#if TFLITE_SUPPORTS_NNAPI_DELEGATE
TfLiteDelegatePtr CreateNNAPIDelegate(StatefulNnApiDelegate::Options options) {
  return TfLiteDelegatePtr(
      new StatefulNnApiDelegate(options), [](TfLiteDelegate* delegate) {
        delete reinterpret_cast<StatefulNnApiDelegate*>(delegate);
      });
}
#endif  // TFLITE_SUPPORTS_NNAPI_DELEGATE

#if TFLITE_SUPPORTS_GPU_DELEGATE
TfLiteDelegatePtr CreateGPUDelegate(TfLiteGpuDelegateOptionsV2* options) {
  return TfLiteDelegatePtr(TfLiteGpuDelegateV2Create(options),
                           &TfLiteGpuDelegateV2Delete);
}
#endif  // TFLITE_SUPPORTS_GPU_DELEGATE

TfLiteDelegatePtr CreateGPUDelegate() {
#if TFLITE_SUPPORTS_GPU_DELEGATE
  TfLiteGpuDelegateOptionsV2 options = TfLiteGpuDelegateOptionsV2Default();
  options.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY;
  options.inference_preference =
      TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;

  return CreateGPUDelegate(&options);
#else
  return tools::CreateNullDelegate();
#endif  // TFLITE_SUPPORTS_GPU_DELEGATE
}

TfLiteDelegatePtr CreateHexagonDelegate(
    const std::string& library_directory_path, bool profiling) {
#if TFLITE_ENABLE_HEXAGON
  TfLiteHexagonDelegateOptions options = {0};
  options.print_graph_profile = profiling;
  return CreateHexagonDelegate(&options, library_directory_path);
#else
  return tools::CreateNullDelegate();
#endif  // TFLITE_ENABLE_HEXAGON
}

#if TFLITE_ENABLE_HEXAGON
TfLiteDelegatePtr CreateHexagonDelegate(
    const TfLiteHexagonDelegateOptions* options,
    const std::string& library_directory_path) {
  if (library_directory_path.empty()) {
    TfLiteHexagonInit();
  } else {
    TfLiteHexagonInitWithPath(library_directory_path.c_str());
  }

  TfLiteDelegate* delegate = TfLiteHexagonDelegateCreate(options);
  if (!delegate) {
    TfLiteHexagonTearDown();
    return tools::CreateNullDelegate();
  }
  return TfLiteDelegatePtr(delegate, [](TfLiteDelegate* delegate) {
    TfLiteHexagonDelegateDelete(delegate);
    TfLiteHexagonTearDown();
  });
}
#endif  // TFLITE_ENABLE_HEXAGON

#ifdef TFLITE_WITHOUT_XNNPACK
TfLiteDelegatePtr CreateXNNPACKDelegate(int num_threads, bool force_fp16,
                                        const char* weight_cache_file_path) {
  return tools::CreateNullDelegate();
}
#else  // !defined(TFLITE_WITHOUT_XNNPACK)
// This method replicates the implementation from
// https://github.com/tensorflow/tensorflow/blob/55e3b5643a791c4cc320746649d455cacfadf6ed/tensorflow/lite/delegates/xnnpack/xnnpack_delegate.cc#L5235
// to avoid having an entire copy of XNNPack.
TfLiteXNNPackDelegateOptions XNNPackDelegateOptionsDefault() {
  TfLiteXNNPackDelegateOptions options = {0};

  // Quantized inference is enabled by default on Web platform
#ifdef XNNPACK_DELEGATE_ENABLE_QS8
  options.flags |= TFLITE_XNNPACK_DELEGATE_FLAG_QS8;
#endif  // XNNPACK_DELEGATE_ENABLE_QS8
#ifdef XNNPACK_DELEGATE_ENABLE_QU8
  options.flags |= TFLITE_XNNPACK_DELEGATE_FLAG_QU8;
#endif  // XNNPACK_DELEGATE_ENABLE_QU8

  // Enable quantized inference for the delegate build used in unit tests.
#ifdef XNNPACK_DELEGATE_TEST_MODE
  options.flags |= TFLITE_XNNPACK_DELEGATE_FLAG_QS8;
  options.flags |= TFLITE_XNNPACK_DELEGATE_FLAG_QU8;
#endif  // XNNPACK_DELEGATE_TEST_MODE
  return options;
}

TfLiteDelegatePtr CreateXNNPACKDelegate() {
  TfLiteXNNPackDelegateOptions xnnpack_options =
      XNNPackDelegateOptionsDefault();
  return CreateXNNPACKDelegate(&xnnpack_options);
}

TfLiteDelegatePtr CreateXNNPACKDelegate(
    const TfLiteXNNPackDelegateOptions* xnnpack_options) {
  flatbuffers::FlatBufferBuilder flatbuffer_builder;
  flatbuffers::Offset<flatbuffers::String> weight_cache_file_path;
  if (xnnpack_options->weight_cache_file_path) {
    TFLITE_LOG(INFO) << "XNNPack file-backed weight cache enabled.";
    weight_cache_file_path = flatbuffer_builder.CreateString(
        xnnpack_options->weight_cache_file_path);
  }

  tflite::XNNPackSettingsBuilder xnnpack_settings_builder(flatbuffer_builder);
  int num_threads = xnnpack_options->num_threads;
  if (num_threads >= 0) {
    xnnpack_settings_builder.add_num_threads(num_threads);
  }
  if (xnnpack_options->flags & TFLITE_XNNPACK_DELEGATE_FLAG_FORCE_FP16) {
    TFLITE_LOG(INFO) << "XNNPack FP16 inference enabled.";
  }
  if (xnnpack_options->flags & TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_SLINKY) {
    TFLITE_LOG(INFO) << "XNNPack Slinky enabled.";
  }
  xnnpack_settings_builder.fbb_.AddElement<int32_t>(
      XNNPackSettings::VT_FLAGS, static_cast<int32_t>(xnnpack_options->flags),
      0);
  xnnpack_settings_builder.fbb_.AddElement<int32_t>(
      XNNPackSettings::VT_RUNTIME_FLAGS,
      static_cast<int32_t>(xnnpack_options->runtime_flags), 0);
  xnnpack_settings_builder.add_weight_cache_file_path(weight_cache_file_path);
  flatbuffers::Offset<tflite::XNNPackSettings> xnnpack_settings =
      xnnpack_settings_builder.Finish();
  tflite::TFLiteSettingsBuilder tflite_settings_builder(flatbuffer_builder);
  tflite_settings_builder.add_xnnpack_settings(xnnpack_settings);
  tflite_settings_builder.add_delegate(tflite::Delegate_XNNPACK);
  flatbuffers::Offset<tflite::TFLiteSettings> tflite_settings =
      tflite_settings_builder.Finish();
  flatbuffer_builder.Finish(tflite_settings);
  const tflite::TFLiteSettings* tflite_settings_flatbuffer =
      flatbuffers::GetRoot<tflite::TFLiteSettings>(
          flatbuffer_builder.GetBufferPointer());
  // Create an XNNPack delegate plugin using the settings from the flatbuffer.
  const TfLiteOpaqueDelegatePlugin* delegate_plugin =
      TfLiteXnnpackDelegatePluginCApi();
  TfLiteOpaqueDelegate* delegate =
      delegate_plugin->create(tflite_settings_flatbuffer);
  void (*delegate_deleter)(TfLiteOpaqueDelegate*) = delegate_plugin->destroy;
  return TfLiteDelegatePtr(delegate, delegate_deleter);
}

TfLiteDelegatePtr CreateXNNPACKDelegate(int num_threads, bool force_fp16,
                                        const char* weight_cache_file_path) {
  auto opts = XNNPackDelegateOptionsDefault();
  // Note that we don't want to use the thread pool for num_threads == 1.
  opts.num_threads = num_threads > 1 ? num_threads : 0;
  if (force_fp16) {
    opts.flags |= TFLITE_XNNPACK_DELEGATE_FLAG_FORCE_FP16;
  }
  if (weight_cache_file_path && weight_cache_file_path[0] != '\0') {
    opts.weight_cache_file_path = weight_cache_file_path;
  }
  return CreateXNNPACKDelegate(&opts);
}
#endif

TfLiteDelegatePtr CreateCoreMlDelegate() {
#ifdef REAL_IPHONE_DEVICE
  TfLiteCoreMlDelegateOptions coreml_options = {
      .enabled_devices = TfLiteCoreMlDelegateAllDevices};
  TfLiteDelegate* delegate = TfLiteCoreMlDelegateCreate(&coreml_options);
  if (!delegate) {
    return tools::CreateNullDelegate();
  }
  return TfLiteDelegatePtr(delegate, &TfLiteCoreMlDelegateDelete);
#else
  return tools::CreateNullDelegate();
#endif  // REAL_IPHONE_DEVICE
}

}  // namespace evaluation
}  // namespace tflite
