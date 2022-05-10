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
#if GOOGLE_CUDA && GOOGLE_TENSORRT

#include "tensorflow/compiler/tf2tensorrt/convert/timing_cache.h"

#include <memory>
#include <string>
#include <unordered_map>

#include "tensorflow/compiler/tf2tensorrt/common/utils.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/lib/io/record_writer.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/util/env_var.h"
#include "third_party/tensorrt/NvInfer.h"

namespace tensorflow {
namespace tensorrt {
namespace convert {

constexpr absl::string_view kCachePathEnvVarName = "TF_TRT_TIMING_CACHE_DIR";

TimingCacheRegistry::TimingCacheRegistry() {
  string dir;
  Status s = ReadStringFromEnvVar(kCachePathEnvVarName, "", &dir);
  if (s.ok() && !dir.empty()) {
    cache_dir_ = dir;
  }
}

static string CreateCachePath(string dir, string cache_name) {
  // TODO(cbate): since timing information is specific to hardware, this should
  // be expanded to include information about the GPU (such as compute
  // capability) in order to reduce liklihood that the user can re-use cache
  // files inappropriately.
  return io::JoinPath(dir, cache_name + ".trt.timing_cache");
}

static StatusOr<TimingCacheRegistry::TimingCachePtr> LoadCacheFromPath(
    string filename, nvinfer1::IBuilderConfig* builder_config) {
  std::unique_ptr<RandomAccessFile> file;
  auto* env = Env::Default();

  if (!env->FileExists(filename).ok()) {
    return errors::Internal("path dooes not exist");
  }

  TF_RETURN_IF_ERROR(env->NewRandomAccessFile(filename, &file));
  auto reader = std::make_unique<io::RecordReader>(file.get());
  uint64 offset{0};
  tstring record;
  TF_RETURN_IF_ERROR(reader->ReadRecord(&offset, &record));
  return std::unique_ptr<nvinfer1::ITimingCache>(
      builder_config->createTimingCache(record.data(), record.size()));
}

StatusOr<TimingCacheRegistry::TimingCachePtr> TimingCacheRegistry::LookUp(
    const string& name, nvinfer1::IBuilderConfig* builder_config) {
#if IS_TRT_VERSION_GE(8, 0, 0, 0)
  TRT_ENSURE(builder_config != nullptr);
  mutex_lock scoped_lock(mu_);

  if (map_.find(name) != map_.end()) {
    const std::vector<uint8_t>& data = map_[name];
    return std::unique_ptr<nvinfer1::ITimingCache>(
        builder_config->createTimingCache(data.data(), data.size()));
  }

  // Check if we have this cache on disk.
  if (cache_dir_.has_value()) {
    string filename = CreateCachePath(*cache_dir_, name);
    StatusOr<TimingCacheRegistry::TimingCachePtr> cache =
        LoadCacheFromPath(filename, builder_config);
    // If we loaded the cache, return it. Otherwise, fallback to creating a new
    // cache below.
    if (cache.ok()) {
      return cache;
    }
  }

  // If no such timing cache exists, create a new timing cache.
  return std::unique_ptr<nvinfer1::ITimingCache>(
      builder_config->createTimingCache(nullptr, 0));
#endif  // IS_TRT_VERSION_GE(8, 0, 0, 0)
  return errors::Unimplemented(
      "serializable timing cache does not exist in TensorRT versions < 8.0");
}

void TimingCacheRegistry::Upsert(const string& name, TimingCache* cache) {
  if (cache == nullptr) {
    LOG(WARNING) << "No timing cache to serialize";
    return;
  }
#if IS_TRT_VERSION_GE(8, 0, 0, 0)
  nvinfer1::IHostMemory* memory = cache->serialize();
  if (memory == nullptr) {
    LOG(WARNING) << "Could not serialize timing cache.";
    return;
  }

  if (map_.find(name) == map_.end()) {
    // If the timing cache with the given name does not exist, emplace the
    // serialized buffer.
    std::vector<uint8_t> mem(memory->size());
    std::copy_n(static_cast<uint8_t*>(memory->data()), memory->size(),
                mem.begin());
    {
      mutex_lock scoped_lock(mu_);
      map_.emplace(name, std::move(mem));
    }
  } else {
    // If the timing cache does exist, use the existing buffer.
    mutex_lock scoped_lock(mu_);
    std::vector<uint8_t>& mem = map_[name];
    mem.resize(memory->size());
    std::copy_n(static_cast<uint8_t*>(memory->data()), memory->size(),
                mem.begin());
  }

  // Write the data to disk.
  if (cache_dir_.has_value()) {
    auto* env = Env::Default();
    string filename = CreateCachePath(*cache_dir_, name);
    LOG(INFO) << "Saving timing cache to " << filename;
    std::unique_ptr<WritableFile> file{nullptr};
    auto s = env->NewWritableFile(filename, &file);
    if (!s.ok()) {
      LOG(WARNING) << "Failed to write cache to " << filename << ": "
                   << s.error_message();
    } else {
      io::RecordWriter writer(file.get());
      s = writer.WriteRecord(StringPiece(
          reinterpret_cast<const char*>(memory->data()), memory->size()));
      if (!s.ok()) {
        LOG(WARNING) << "Failed to write to cache file: " << s.error_message();
      }
    }
  }

  memory->destroy();
#endif  // IS_TRT_VERSION_GE(8, 0, 0, 0)
}

TimingCacheRegistry* GetTimingCacheRegistry() {
  static TimingCacheRegistry* registry = new TimingCacheRegistry();
  return registry;
}

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
