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

#include "tensorflow/lite/logger.h"

#if defined(_WIN32)
#include <fstream>
#include <iostream>
#else
#include <errno.h>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#endif  // defined(_WIN32)

#include <time.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/minimal_logging.h"
#include <farmhash.h>

namespace tflite {
namespace delegates {
namespace {

static const char kDelegatedNodesSuffix[] = "_dnodes";

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
  std::ofstream out_file(temp_filepath.c_str(), std::ios_base::binary);
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
      TF_LITE_KERNEL_LOG(context, "Failed to write data to: %s, error: %s",
                         temp_filepath.c_str(), std::strerror(errno));
      return kTfLiteDelegateDataWriteError;
    }

    len += ret;
    buf += ret;
  } while (len < static_cast<ssize_t>(size));
  // Use fsync to ensure data is on disk before renaming temp file.
  if (fsync(fd) < 0) {
    close(fd);
    TF_LITE_KERNEL_LOG(context, "Could not fsync: %s, error: %s",
                       temp_filepath.c_str(), std::strerror(errno));
    return kTfLiteDelegateDataWriteError;
  }
  if (close(fd) < 0) {
    TF_LITE_KERNEL_LOG(context, "Could not close fd: %s, error: %s",
                       temp_filepath.c_str(), std::strerror(errno));
    return kTfLiteDelegateDataWriteError;
  }
  if (rename(temp_filepath.c_str(), filepath.c_str()) < 0) {
    TF_LITE_KERNEL_LOG(context, "Failed to rename to %s, error: %s",
                       filepath.c_str(), std::strerror(errno));
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

#if defined(_WIN32)
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
#else   // !defined(_WIN32)
  // This method only works on unix/POSIX systems, but is more optimized & has
  // lower size overhead for Android binaries.
  data->clear();
  // O_CLOEXEC is needed for correctness, as another thread may call
  // popen() and the callee inherit the lock if it's not O_CLOEXEC.
  int fd = open(filepath.c_str(), O_RDONLY | O_CLOEXEC, 0600);
  if (fd < 0) {
    TF_LITE_KERNEL_LOG(context, "File %s couldn't be opened for reading: %s",
                       filepath.c_str(), std::strerror(errno));
    return kTfLiteDelegateDataNotFound;
  }
  int lock_status = flock(fd, LOCK_EX);
  if (lock_status < 0) {
    close(fd);
    TF_LITE_KERNEL_LOG(context, "Could not flock %s: %s", filepath.c_str(),
                       std::strerror(errno));
    return kTfLiteDelegateDataReadError;
  }

  struct stat file_stat;
  if (fstat(fd, &file_stat) < 0) {
    close(fd);
    TF_LITE_KERNEL_LOG(context, "Could not fstat %s: %s", filepath.c_str(),
                       std::strerror(errno));
    return kTfLiteDelegateDataReadError;
  }
  data->resize(file_stat.st_size);

  size_t total_read = 0;
  while (total_read < data->size()) {
    ssize_t bytes_read =
        read(fd, data->data() + total_read, data->size() - total_read);
    total_read += bytes_read;

    if (bytes_read < 0) {
      close(fd);
      TF_LITE_KERNEL_LOG(context, "Error reading %s: %s", filepath.c_str(),
                         std::strerror(errno));
      return kTfLiteDelegateDataReadError;
    }
  }

  close(fd);
#endif  // defined(_WIN32)

  TFLITE_LOG_PROD(TFLITE_LOG_INFO,
                  "Found serialized data for model %s (%d B) at %s",
                  model_token_.c_str(), data->size(), filepath.c_str());

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
  // A quick heuristic involving graph tensors to 'fingerprint' a
  // tflite::Subgraph. We don't consider the execution plan, since it could be
  // in flux if the delegate uses this method during
  // ReplaceNodeSubsetsWithDelegateKernels (eg in kernel Init).
  if (context) {
    std::vector<int32_t> context_data;
    // Number of tensors can be large.
    const int tensors_to_consider = std::min<int>(context->tensors_size, 100);
    context_data.reserve(1 + tensors_to_consider);
    context_data.push_back(context->tensors_size);
    for (int i = 0; i < tensors_to_consider; ++i) {
      context_data.push_back(context->tensors[i].bytes);
    }
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

TfLiteStatus SaveDelegatedNodes(TfLiteContext* context,
                                Serialization* serialization,
                                const std::string& delegate_id,
                                const TfLiteIntArray* node_ids) {
  if (!node_ids) return kTfLiteError;
  std::string cache_key = delegate_id + kDelegatedNodesSuffix;
  auto entry = serialization->GetEntryForDelegate(cache_key, context);
  return entry.SetData(context, reinterpret_cast<const char*>(node_ids),
                       (1 + node_ids->size) * sizeof(int));
}

TfLiteStatus GetDelegatedNodes(TfLiteContext* context,
                               Serialization* serialization,
                               const std::string& delegate_id,
                               TfLiteIntArray** node_ids) {
  if (!node_ids) return kTfLiteError;
  std::string cache_key = delegate_id + kDelegatedNodesSuffix;
  auto entry = serialization->GetEntryForDelegate(cache_key, context);

  std::string read_buffer;
  TF_LITE_ENSURE_STATUS(entry.GetData(context, &read_buffer));
  if (read_buffer.empty()) return kTfLiteOk;
  *node_ids = TfLiteIntArrayCopy(
      reinterpret_cast<const TfLiteIntArray*>(read_buffer.data()));
  return kTfLiteOk;
}

}  // namespace delegates
}  // namespace tflite
