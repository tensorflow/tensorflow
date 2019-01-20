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

#ifndef TENSORFLOW_CONTRIB_TENSORRT_RESOURCES_TRT_LRU_CACHE_H_
#define TENSORFLOW_CONTRIB_TENSORRT_RESOURCES_TRT_LRU_CACHE_H_

#include <list>
#include <unordered_map>

#include "tensorflow/contrib/tensorrt/convert/utils.h"
#include "tensorflow/contrib/tensorrt/resources/trt_allocator.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/core/errors.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT
#include "tensorrt/include/NvInfer.h"
#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT

namespace tensorflow {
namespace tensorrt {

template <class Key, class Value, class HashFunction>
class LRUCache {
 public:
  typedef Value value_type;
  typedef Key key_type;
  typedef HashFunction hasher;
  typedef typename std::unordered_map<key_type, value_type, hasher> map_type;
  typedef typename map_type::iterator iterator;
  typedef typename map_type::const_iterator const_iterator;

  LRUCache() : capacity_(0) {}
  explicit LRUCache(size_t capacity) : capacity_(capacity) {}

  size_t capacity() const { return capacity_; }

  void reserve(size_t capacity) {
    capacity_ = capacity;
    DiscardOld();
  }

  size_t size() const { return objects_.size(); }

  size_t count(const key_type& key) const { return objects_.count(key); }

  value_type& at(const key_type& key) { return Touch(key); }

  const_iterator begin() const { return objects_.begin(); }
  const_iterator end() const { return objects_.end(); }

  iterator begin() { return objects_.begin(); }
  iterator end() { return objects_.end(); }

  template <typename... Args>
  std::pair<iterator, bool> emplace(Args&&... args) {
    DiscardOld(1);
    std::pair<iterator, bool> result =
        objects_.emplace(std::forward<Args>(args)...);
    key_type key = result.first->first;
    if (result.second) {
      keys_.push_front(key);
    } else {
      TouchNoCheck(key);  // The key must exist in this case.
    }
    return result;
  }

 private:
  std::unordered_map<key_type, value_type, hasher> objects_;
  std::list<key_type> keys_;
  size_t capacity_;
  value_type not_found_value_;

  value_type& Touch(const key_type& key) {
    // Check that the key exists, and let it return std::out_of_range error if
    // not.
    value_type& value = objects_.at(key);
    TouchNoCheck(key);
    return value;
  }

  void TouchNoCheck(const key_type& key) {
    auto rank = std::find(keys_.begin(), keys_.end(), key);
    if (rank != keys_.begin()) {
      keys_.erase(rank);
      keys_.push_front(key);
    }
  }

  // Creates n free positions in cache
  tensorflow::Status DiscardOld(size_t n = 0) {
    if (n > capacity_) {
      return tensorflow::errors::Internal(
          "Insufficient capacity in cache (capacity = ", capacity_,
          ", requested ", n, ")");
    }
    while (objects_.size() > (capacity_ - n)) {
      key_type discard_key = keys_.back();
      keys_.pop_back();
      objects_.erase(discard_key);
    }
    return tensorflow::Status::OK();
  }
};

// Define a hash function for vector<TensorShape> because it is used as the key
// for the engine cache.
struct VectorTensorShapeHasher {
  std::size_t operator()(
      const std::vector<tensorflow::TensorShape>& key) const {
    return std::hash<std::string>()(TensorShapeUtils::ShapeListString(key));
  }
};

#if GOOGLE_CUDA
#if GOOGLE_TENSORRT

struct EngineContext {
  EngineContext() {}  // Creates an empty context.
  EngineContext(
      TrtUniquePtrType<nvinfer1::ICudaEngine>&& input_cuda_engine,
      TrtUniquePtrType<nvinfer1::IExecutionContext>&& input_execution_context)
      : cuda_engine(std::move(input_cuda_engine)),
        execution_context(std::move(input_execution_context)) {}

  mutex mu;
  TrtUniquePtrType<nvinfer1::ICudaEngine> cuda_engine;
  TrtUniquePtrType<nvinfer1::IExecutionContext> execution_context
      GUARDED_BY(mu);
};

class TRTEngineCacheResource : public tensorflow::ResourceBase {
 public:
  TRTEngineCacheResource(OpKernelContext* ctx, size_t capacity)
      : cache_(capacity) {
    auto device = ctx->device();
    auto alloc = device->GetAllocator(tensorflow::AllocatorAttributes());
    if (!alloc) {
      LOG(ERROR) << "Can't find device allocator for gpu device "
                 << device->name();
      allocator_ = nullptr;
    } else {
      allocator_.reset(new TRTDeviceAllocator(alloc));
    }
  }

  string DebugString() const override {
    std::stringstream oss;
    using std::dec;
    using std::endl;
    using std::hex;
    oss << "TRTEngineCacheResource: ";
    oss << "TRTBaseAllocator = " << hex << allocator_.get() << dec << ", ";
    oss << "LRUCache = " << hex << &cache_ << dec << endl;
    oss << "Containing " << cache_.size() << " entries: " << endl;
    for (const auto& item : cache_) {
      oss << TensorShapeUtils::ShapeListString(item.first) << ": " << hex
          << "ICudaEngine: " << item.second.get()->cuda_engine.get() << ", "
          << "IExecutionContext: " << item.second.get()->execution_context.get()
          << dec << endl;
    }
    return oss.str();
  }

  // Keep device allocator for TRT.
  std::unique_ptr<TRTBaseAllocator> allocator_;

  // Declare cache after allocator so that it is destroyed before allocator is.
  LRUCache<std::vector<TensorShape>, std::unique_ptr<EngineContext>,
           VectorTensorShapeHasher>
      cache_;
};

#endif  // GOOGLE_TENSORRT
#endif  // GOOGLE_CUDA

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_TENSORRT_RESOURCES_TRT_LRU_CACHE_H_
