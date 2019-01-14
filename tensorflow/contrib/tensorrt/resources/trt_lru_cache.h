#ifndef TENSORFLOW_CONTRIB_TENSORRT_TRT_LRU_CACHE_H_
#define TENSORFLOW_CONTRIB_TENSORRT_TRT_LRU_CACHE_H_

#include <list>
#include <unordered_map>

#include "tensorflow/contrib/tensorrt/convert/utils.h"
#include "tensorflow/contrib/tensorrt/resources/trt_allocator.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/core/errors.h"

#if GOOGLE_CUDA
#if GOOGLE_TENSORRT
#include "tensorrt/include/NvInfer.h"

namespace tensorflow {
namespace tensorrt {

template <class Key, class Value, class HashFunction>
class LRUCache {
 public:
  typedef Value value_type;
  typedef Key key_type;
  typedef HashFunction hasher;
  typedef typename std::unordered_map<key_type, value_type, hasher>::iterator
      map_iterator;

  LRUCache() : capacity_(0) {}
  explicit LRUCache(size_t capacity) : capacity_(capacity) {}

  size_t capacity() const { return capacity_; }

  void reserve(size_t capacity) {
    capacity_ = capacity;
    DiscardOld();
  }

  size_t size() const { return objects_.size(); }

  size_t count(const key_type& k) const { return objects_.count(k); }

  value_type& at(key_type k, tensorflow::Status* status_ptr = nullptr) {
    tensorflow::Status status = Touch(k);
    if (!status.ok()) {
      if (status_ptr) {
        *status_ptr = status;
      }
      return not_found_value_;
    }
    return objects_.at(k);
  }

  map_iterator begin() { return objects_.begin(); }

  map_iterator end() { return objects_.end(); }

  template <typename... Args>
  std::pair<map_iterator, bool> emplace(Args&&... args) {
    DiscardOld(1);
    std::pair<map_iterator, bool> result =
        objects_.emplace(std::forward<Args>(args)...);
    key_type key = result.first->first;
    if (result.second) {
      keys_.push_front(key);
    } else {
      Touch(key);
    }
    return result;
  }

 private:
  std::unordered_map<key_type, value_type, hasher> objects_;
  std::list<key_type> keys_;
  size_t capacity_;
  value_type not_found_value_;

  tensorflow::Status Touch(const key_type& key) {
    if (!count(key)) {
      return tensorflow::errors::NotFound("Key not found in cache.");
    }
    auto rank = std::find(keys_.begin(), keys_.end(), key);
    if (rank != keys_.begin()) {
      keys_.erase(rank);
      keys_.push_front(key);
    }
    return tensorflow::Status::OK();
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
    for (auto& item : cache_) {
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

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_TENSORRT
#endif  // GOOGLE_CUDA

#endif  // TENSORFLOW_CONTRIB_TENSORRT_TRT_LRU_CACHE_H_
