/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/delegates/serialization.h"

#if defined(_WIN32)
#else
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif  // defined(_WIN32)

#include <time.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/minimal_logging.h"
#include <farmhash.h>

namespace tflite {
namespace delegates {
namespace {

// Farmhash Fingerprint
inline uint64_t CombineFingerprints(uint64_t l, uint64_t h) {
  // Murmur-inspired hashing.
  const uint64_t kMul = 0x9ddfea08eb382d69ULL;
  uint64_t a = (l ^ h) * kMul;
  a ^= (a >> 47);
  uint64_t b = (h ^ a) * kMul;
  b ^= (b >> 44);
  b *= kMul;
  b ^= (b >> 41);
  b *= kMul;
  return b;
}

inline std::string JoinPath(const std::string& path1,
                            const std::string& path2) {
  return (path1.back() == '/') ? (path1 + path2) : (path1 + "/" + path2);
}

inline std::string GetFilePath(const std::string& cache_dir,
                               const std::string& model_token,
                               const uint64_t fingerprint) {
  auto file_name = (model_token + "_" + std::to_string(fingerprint) + ".bin");
  return JoinPath(cache_dir, file_name);
}

}  // namespace

std::string StrFingerprint(const void* data, const size_t num_bytes) {
  return std::to_string(
      ::util::Fingerprint64(reinterpret_cast<const char*>(data), num_bytes));
}

SerializationEntry::SerializationEntry(const std::string& cache_dir,
                                       const std::string& model_token,
                                       const uint64_t fingerprint)
    : cache_dir_(cache_dir),
      model_token_(model_token),
      fingerprint_(fingerprint) {}

TfLiteStatus SerializationEntry::SetData(TfLiteContext* context,
                                         const char* data,
                                         const size_t size) const {
  auto filepath = GetFilePath(cache_dir_, model_token_, fingerprint_);
  // Temporary file to write data to.
  const std::string temp_filepath =
      JoinPath(cache_dir_, (model_token_ + std::to_string(fingerprint_) +
                            std::to_string(time(nullptr))));

#if defined(_WIN32)
  std::ofstream out_file(temp_filepath.c_str());
  if (!out_file) {
    TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "Could not create file: %s",
                    temp_filepath.c_str());
    return kTfLiteDelegateDataWriteError;
  }
  out_file.write(data, size);
  out_file.flush();
  out_file.close();
  // rename is an atomic operation in most systems.
  if (rename(temp_filepath.c_str(), filepath.c_str()) < 0) {
    TF_LITE_KERNEL_LOG(context, "Failed to rename to %s", filepath.c_str());
    return kTfLiteDelegateDataWriteError;
  }
#else   // !defined(_WIN32)
  // This method only works on unix/POSIX systems.
  const int fd = open(temp_filepath.c_str(),
                      O_WRONLY | O_APPEND | O_CREAT | O_CLOEXEC, 0600);
  if (fd < 0) {
    TF_LITE_KERNEL_LOG(context, "Failed to open for writing: %s",
                       temp_filepath.c_str());
    return kTfLiteDelegateDataWriteError;
  }
  // Loop until all bytes written.
  ssize_t len = 0;
  const char* buf = data;
  do {
    ssize_t ret = write(fd, buf, size);
    if (ret <= 0) {
      close(fd);
      TF_LITE_KERNEL_LOG(context, "Failed to write data to: %s, error: %d",
                         temp_filepath.c_str(), errno);
      return kTfLiteDelegateDataWriteError;
    }

    len += ret;
    buf += ret;
  } while (len < static_cast<ssize_t>(size));
  // Use fsync to ensure data is on disk before renaming temp file.
  if (fsync(fd) < 0) {
    close(fd);
    TF_LITE_KERNEL_LOG(context, "Could not fsync: %s, error: %d",
                       temp_filepath.c_str(), errno);
    return kTfLiteDelegateDataWriteError;
  }
  if (close(fd) < 0) {
    TF_LITE_KERNEL_LOG(context, "Could not close fd: %s, error: %d",
                       temp_filepath.c_str(), errno);
    return kTfLiteDelegateDataWriteError;
  }
  if (rename(temp_filepath.c_str(), filepath.c_str()) < 0) {
    TF_LITE_KERNEL_LOG(context, "Failed to rename to %s, error: %d",
                       filepath.c_str(), errno);
    return kTfLiteDelegateDataWriteError;
  }
#endif  // defined(_WIN32)

  TFLITE_LOG(TFLITE_LOG_INFO, "Wrote serialized data for model %s (%d B) to %s",
             model_token_.c_str(), size, filepath.c_str());

  return kTfLiteOk;
}

TfLiteStatus SerializationEntry::GetData(TfLiteContext* context,
                                         std::string* data) const {
  if (!data) return kTfLiteError;
  auto filepath = GetFilePath(cache_dir_, model_token_, fingerprint_);

  // TODO(b/188704640): Benchmark this file IO to optimize it further?
  std::ifstream cache_stream(filepath,
                             std::ios_base::in | std::ios_base::binary);
  if (cache_stream.good()) {
    cache_stream.seekg(0, cache_stream.end);
    int cache_size = cache_stream.tellg();
    cache_stream.seekg(0, cache_stream.beg);

    data->resize(cache_size);
    cache_stream.read(&(*data)[0], cache_size);
    cache_stream.close();
  }

  if (!data->empty()) {
    TFLITE_LOG(TFLITE_LOG_INFO, "Data found at %s: %d bytes", filepath.c_str(),
               data->size());
    return kTfLiteOk;
  } else {
    TF_LITE_KERNEL_LOG(context, "No serialized data found: %s",
                       filepath.c_str());
    return kTfLiteDelegateDataNotFound;
  }
}

SerializationEntry Serialization::GetEntryImpl(
    const std::string& custom_key, TfLiteContext* context,
    const TfLiteDelegateParams* delegate_params) {
  // First incorporate model_token.
  // We use Fingerprint64 instead of std::hash, since the latter isn't
  // guaranteed to be stable across runs. See b/172237993.
  uint64_t fingerprint =
      ::util::Fingerprint64(model_token_.c_str(), model_token_.size());

  // Incorporate custom_key.
  const uint64_t custom_str_fingerprint =
      ::util::Fingerprint64(custom_key.c_str(), custom_key.size());
  fingerprint = CombineFingerprints(fingerprint, custom_str_fingerprint);

  // Incorporate context details, if provided.
  // A quick heuristic that considers the number of tensors & execution plan
  // to 'fingerprint' a tflite::Subgraph.
  if (context) {
    std::vector<int32_t> context_data;
    TfLiteIntArray* execution_plan;
    if (context->GetExecutionPlan(context, &execution_plan) != kTfLiteOk) {
      TF_LITE_KERNEL_LOG(context, "Could not get execution plan from context");
      return SerializationEntry(cache_dir_, model_token_, fingerprint);
    }
    context_data.reserve(execution_plan->size + 2);
    context_data.push_back(context->tensors_size);
    context_data.push_back(execution_plan->size);
    context_data.insert(context_data.end(), execution_plan->data,
                        execution_plan->data + execution_plan->size);
    const uint64_t context_fingerprint =
        ::util::Fingerprint64(reinterpret_cast<char*>(context_data.data()),
                                context_data.size() * sizeof(int32_t));
    fingerprint = CombineFingerprints(fingerprint, context_fingerprint);
  }

  // Incorporate delegated partition details, if provided.
  // A quick heuristic that considers the nodes & I/O tensor sizes to
  // fingerprint TfLiteDelegateParams.
  if (delegate_params) {
    std::vector<int32_t> partition_data;
    auto* nodes = delegate_params->nodes_to_replace;
    auto* input_tensors = delegate_params->input_tensors;
    auto* output_tensors = delegate_params->output_tensors;
    partition_data.reserve(nodes->size + input_tensors->size +
                           output_tensors->size);
    partition_data.insert(partition_data.end(), nodes->data,
                          nodes->data + nodes->size);
    for (int i = 0; i < input_tensors->size; ++i) {
      auto& tensor = context->tensors[input_tensors->data[i]];
      partition_data.push_back(tensor.bytes);
    }
    for (int i = 0; i < output_tensors->size; ++i) {
      auto& tensor = context->tensors[output_tensors->data[i]];
      partition_data.push_back(tensor.bytes);
    }
    const uint64_t partition_fingerprint =
        ::util::Fingerprint64(reinterpret_cast<char*>(partition_data.data()),
                                partition_data.size() * sizeof(int32_t));
    fingerprint = CombineFingerprints(fingerprint, partition_fingerprint);
  }

  // Get a fingerprint-specific lock that is passed to the SerializationKey, to
  // ensure noone else gets access to an equivalent SerializationKey.
  return SerializationEntry(cache_dir_, model_token_, fingerprint);
}

}  // namespace delegates
}  // namespace tflite
