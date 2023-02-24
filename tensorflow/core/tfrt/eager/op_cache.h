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
#ifndef TENSORFLOW_CORE_TFRT_EAGER_OP_CACHE_H_
#define TENSORFLOW_CORE_TFRT_EAGER_OP_CACHE_H_

#include "tensorflow/core/common_runtime/eager/attr_builder.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/tfrt/utils/utils.h"
#include "tfrt/core_runtime/core_runtime.h"  // from @tf_runtime
#include "tfrt/core_runtime/core_runtime_op.h"  // from @tf_runtime
#include "tfrt/core_runtime/op_attrs.h"  // from @tf_runtime
#include "tfrt/core_runtime/op_handler.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#include "tfrt/support/error_util.h"  // from @tf_runtime
#include "tfrt/support/mutex.h"  // from @tf_runtime
#include "tfrt/support/ref_count.h"  // from @tf_runtime
#include "tfrt/support/string_util.h"  // from @tf_runtime

namespace tfrt {
namespace tf {

class ContextInterface;
class OperationInterface;

// Cache for a single core runtime op. Thread safe.
class OpCache {
 public:
  // Helper function to look up the cache. If miss, insert the CoreRuntimeOp
  // to the cache.
  Expected<CoreRuntimeOp*> GetOrAddOp(string_view op_name,
                                      OpHandler* op_handler,
                                      string_view device_name,
                                      llvm::SmallVector<string_view, 4> dtypes,
                                      OperationInterface* const op_interface)
      TFRT_EXCLUDES(cache_mu_);

  // Compile with XLA is currently supported via fallback, and the compilation
  // result is a CoreRuntimeOp.
  // TODO(tfrt-devs): Native support of compile_with_xla.
  Expected<CoreRuntimeOp*> GetOrAddXlaOp(string_view op_name,
                                         ContextInterface* context)
      TFRT_EXCLUDES(cache_mu_);

  // The following helper functions are for debugging and testing only.
  size_t Size() const {
    mutex_lock l(cache_mu_);
    return cache_.size();
  }

  bool Contains(string_view op_name, OpHandler* op_handler,
                string_view device_name,
                llvm::SmallVector<string_view, 4> dtypes) const {
    const CacheKey& cache_key{op_name, op_handler,
                              (op_handler == nullptr ? device_name : ""),
                              dtypes};
    mutex_lock l(cache_mu_);
    return cache_.find(cache_key) != cache_.end();
  }

 private:
  class CacheKey {
   public:
    CacheKey(string_view op_name, OpHandler* op_handler,
             string_view device_name, llvm::SmallVector<string_view, 4> dtypes)
        : op_handler_(op_handler),
          op_name_(op_name),
          device_name_(device_name),
          dtypes_(dtypes) {}

    CacheKey(const CacheKey& other)
        : op_handler_(other.op_handler_),
          op_name_(other.op_name_),
          device_name_(other.device_name_),
          dtypes_(other.dtypes_) {
      // Copy the concrete strings if the key is concrete, and set the
      // string_views to refer to the concrete strings.
      if (other.is_concrete_) {
        op_name_concrete_ = other.op_name_concrete_;
        op_name_ = op_name_concrete_.data();
        device_name_concrete_ = other.device_name_concrete_;
        device_name_ = device_name_concrete_.data();
        size_t n = other.dtypes_concrete_.size();
        dtypes_concrete_.reserve(n);
        dtypes_.clear();
        for (size_t i = 0; i < n; ++i) {
          dtypes_concrete_.push_back(other.dtypes_concrete_[i]);
          dtypes_.push_back(dtypes_concrete_[i].data());
        }
        is_concrete_ = true;
      }
    }

    // Make the cache key concrete by copying the key components (strings) to
    // internal storage.
    void MakeConcrete() {
      op_name_concrete_ = op_name_.str();
      device_name_concrete_ = device_name_.str();
      dtypes_concrete_.reserve(dtypes_.size());
      for (const auto& dtype : dtypes_) dtypes_concrete_.push_back(dtype.str());
      is_concrete_ = true;
    }

    bool operator==(const CacheKey& other) const {
      // During comparing keys, self or other can be either concrete or not.
      // If a CacheKey is concrete, it's likely that the string_view fields
      // are not valid (for example the key is obtained from the cache). We
      // need to make the string_view fields refer to the concrete fields
      // by constructing copies of them.
      CacheKey lhs{*this};
      CacheKey rhs{other};

      if (lhs.op_handler_ != rhs.op_handler_) return false;
      if (lhs.dtypes_.size() != rhs.dtypes_.size()) return false;

      for (size_t i = 0, n = lhs.dtypes_.size(); i < n; ++i) {
        if (lhs.dtypes_[i] != rhs.dtypes_[i]) return false;
      }
      return (lhs.op_name_ == rhs.op_name_ &&
              lhs.device_name_ == rhs.device_name_);
    }

    string_view OpName() { return op_name_; }

    string_view DeviceName() { return device_name_; }

    const llvm::SmallVector<string_view, 4>& Dtypes() { return dtypes_; }

   private:
    class OpHandler* op_handler_;
    // friend size_t CacheKeyHash::operator()(const CacheKey& input_key);
    // string_view is used for efficient cache look up to avoid string copy.
    string_view op_name_, device_name_;
    llvm::SmallVector<string_view, 4> dtypes_;

    // Concrete string is used for storing cache key, since the lifetime
    // of the strings should be the same as the container.
    bool is_concrete_ = false;
    std::string op_name_concrete_, device_name_concrete_;
    llvm::SmallVector<std::string, 4> dtypes_concrete_;
  };

  class CacheKeyHash {
   public:
    size_t operator()(const CacheKey& input_key) const {
      CacheKey key{input_key};
      tensorflow::Fprint128 hash = tensorflow::Fingerprint128(
          {key.OpName().data(), key.OpName().size()});
      hash = tsl::FingerprintCat128(
          hash, tensorflow::Fingerprint128(
                    {key.DeviceName().data(), key.DeviceName().size()}));
      for (const auto& dtype : key.Dtypes())
        hash = tsl::FingerprintCat128(
            hash, tensorflow::Fingerprint128({dtype.data(), dtype.size()}));
      return hash.high64 ^ hash.low64;
    }
  };

  mutable mutex cache_mu_;
  std::unordered_map<CacheKey, CoreRuntimeOp, CacheKeyHash> cache_
      TFRT_GUARDED_BY(cache_mu_);
};

}  // namespace tf
}  // namespace tfrt

#endif  // TENSORFLOW_CORE_TFRT_EAGER_OP_CACHE_H_
