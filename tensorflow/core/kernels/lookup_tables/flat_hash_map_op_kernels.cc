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

#include <memory>
#include <type_traits>
#include "absl/base/attributes.h"
#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/lookup_tables/op_kernel_templates.h"
#include "tensorflow/core/kernels/lookup_tables/resource_interface_templates.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/fingerprint.h"

namespace tensorflow {
namespace tables {

using errors::InvalidArgument;

// absl::flat_hash_map<HeterogeneousKeyType, ValueType> backed table with inline
// fallback to x -> (Fingerprint64(x) % num_oov_buckets) + offset when looked
// up keys are not in the flat_hash_map. Inlining the fallback table turns out
// to be quite efficient in comparison to virtual dispatch for the fallback
// lookup.
template <typename ValueType>
class StaticStringFlatHashMap final
    : public virtual LookupInterface<ValueType*, const absl::string_view&>,
      public virtual LookupInterface<ValueType*, const string&>,
      public virtual LookupWithPrefetchInterface<
          absl::Span<ValueType>, absl::Span<const absl::string_view>>,
      public virtual LookupWithPrefetchInterface<absl::Span<ValueType>,
                                                 absl::Span<const string>>,
      public virtual KeyValueTableInitializerInterface<
          absl::Span<const ValueType>, absl::Span<const absl::string_view>>,
      public virtual KeyValueTableInitializerInterface<
          absl::Span<const ValueType>, absl::Span<const string>>,
      public virtual SizeInterface {
 public:
  using value_type = ValueType;

  StaticStringFlatHashMap(bool enable_synchronization, int64 num_oov_buckets)
      : num_oov_buckets_(num_oov_buckets) {
    if (enable_synchronization) {
      mutex_ = absl::make_unique<mutex>();
    }
  }

  Status Initialize(absl::Span<const absl::string_view> keys,
                    absl::Span<const ValueType> values) override {
    if (ABSL_PREDICT_FALSE(keys.size() != values.size())) {
      return errors::InvalidArgument(
          "keys and values do not have the same number of elements (found ",
          keys.size(), " vs ", values.size(), ").");
    }

    table_.reserve(table_.size() + keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
      table_.insert_or_assign(string(keys[i]), values[i]);
    }
    return Status::OK();
  }

  Status Initialize(absl::Span<const string> keys,
                    absl::Span<const ValueType> values) override {
    if (ABSL_PREDICT_FALSE(keys.size() != values.size())) {
      return errors::InvalidArgument(
          "keys and values do not have the same number of elements (found ",
          keys.size(), " vs ", values.size(), ").");
    }

    table_.reserve(table_.size() + keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
      table_.insert_or_assign(keys[i], values[i]);
    }
    return Status::OK();
  }

  Status Lookup(const absl::string_view& key, ValueType* value) const override {
    *value = LookupHelper(key);
    return Status::OK();
  }

  Status Lookup(const string& key, ValueType* value) const override {
    *value = LookupHelper(key);
    return Status::OK();
  }

  // keys and values are guaranteed to have the same size by convention.
  Status Lookup(absl::Span<const absl::string_view> keys,
                absl::Span<ValueType> values,
                int64 prefetch_lookahead) const override {
    const auto keys_size = keys.size();
    if (prefetch_lookahead <= 0 || prefetch_lookahead >= keys_size) {
      for (size_t i = 0; i < keys_size; ++i) {
        values[i] = LookupHelper(keys[i]);
      }
    } else {
      for (size_t i = 0; i < keys_size; ++i) {
        if (i + prefetch_lookahead < keys.size()) {
          table_.prefetch(keys[i + prefetch_lookahead]);
        }
        values[i] = LookupHelper(keys[i]);
      }
    }
    return Status::OK();
  }

  // keys and values are guaranteed to have the same size by convention.
  Status Lookup(absl::Span<const string> keys, absl::Span<ValueType> values,
                int64 prefetch_lookahead) const override {
    const auto keys_size = keys.size();
    if (prefetch_lookahead <= 0 || prefetch_lookahead >= keys_size) {
      for (size_t i = 0; i < keys_size; ++i) {
        values[i] = LookupHelper(keys[i]);
      }
    } else {
      for (size_t i = 0; i < keys_size; ++i) {
        if (i + prefetch_lookahead < keys.size()) {
          table_.prefetch(keys[i + prefetch_lookahead]);
        }
        values[i] = LookupHelper(keys[i]);
      }
    }
    return Status::OK();
  }

  uint64 Size() const override { return table_.size(); }

  mutex* GetMutex() const override { return mutex_.get(); }

  string DebugString() const override { return __PRETTY_FUNCTION__; }

 private:
  template <typename T>
  ABSL_ATTRIBUTE_ALWAYS_INLINE ValueType
  LookupHelper(const T& key_to_find) const {
    auto it = table_.find(key_to_find);
    if (it != table_.end()) {
      return it->second;
    } else {
      return static_cast<ValueType>(Fingerprint64(key_to_find) %
                                    num_oov_buckets_) +
             StaticStringFlatHashMap::Size();
    }
  }

  const int64 num_oov_buckets_;
  std::unique_ptr<mutex> mutex_;
  // The underlying table.
  absl::flat_hash_map<string, ValueType> table_;
  TF_DISALLOW_COPY_AND_ASSIGN(StaticStringFlatHashMap);
};

// Used to allocate StaticStringFlatHashMap objects via the AllocateContainer
// method.
template <typename StaticStringFlatHashMap>
struct StaticStringFlatHashMapFactory {
  struct Functor {
    using resource_type = StaticStringFlatHashMap;

    template <typename StaticStringFlatHashMapBase>
    static Status AllocateContainer(OpKernelContext* ctx, OpKernel* kernel,
                                    StaticStringFlatHashMapBase** container) {
      OpInputList table_int64_args;
      TF_RETURN_IF_ERROR(
          ctx->input_list("table_int64_args", &table_int64_args));
      const size_t variadic_arg_size = table_int64_args.size();
      if (ABSL_PREDICT_FALSE(variadic_arg_size != 2)) {
        return errors::InvalidArgument(
            "table_int64_args should have 2 elements (found ",
            variadic_arg_size,
            "). Set the first element to 1 to enable synchronized table use "
            "and to 0 otherwise. The second element should be "
            "num_oov_buckets.");
      }

      const bool enable_synchronization = ctx->input(0).scalar<int64>()() != 0;
      const int64 num_oov_buckets = ctx->input(1).scalar<int64>()();
      if (ABSL_PREDICT_FALSE(num_oov_buckets <= 0)) {
        return errors::InvalidArgument(
            "num_oov_buckets must be positive. Found: ", num_oov_buckets);
      }
      auto* non_virtual_container =
          new StaticStringFlatHashMap(enable_synchronization, num_oov_buckets);
      *container = non_virtual_container;
      const Tensor& keys = ctx->input(table_int64_args.size());
      const Tensor& values = ctx->input(table_int64_args.size() + 1);
      if (keys.NumElements() == 0) {
        return Status::OK();
      } else if (keys.dtype() == DT_STRING) {
        return Functor::Initialize(
            keys.flat<string>(),
            values.flat<typename StaticStringFlatHashMap::value_type>(),
            non_virtual_container);
      } else if (keys.dtype() == DT_VARIANT) {
        auto keys_flat = keys.flat<Variant>();
        if (keys_flat(0).get<absl::string_view>() == nullptr) {
          return errors::InvalidArgument(
              "Variant keys tensor must have subtype absl::string_view.");
        }
        return Functor::Initialize(
            keys.flat<Variant>(),
            values.flat<typename StaticStringFlatHashMap::value_type>(),
            non_virtual_container);
      }
      return errors::InvalidArgument(
          "keys tensor must have type DT_STRING or type DT_VARIANT with "
          "subtype absl::string_view.");
    }

    static Status Initialize(
        const absl::Span<const string> keys,
        const absl::Span<const typename StaticStringFlatHashMap::value_type>
            values,
        StaticStringFlatHashMap* container) {
      return container->Initialize(keys, values);
    }

    static Status Initialize(
        const absl::Span<const Variant> keys,
        const absl::Span<const typename StaticStringFlatHashMap::value_type>
            values,
        StaticStringFlatHashMap* container) {
      std::vector<typename absl::string_view> keys_vec;
      keys_vec.reserve(keys.size());
      for (size_t i = 0; i < keys.size(); ++i) {
        keys_vec.push_back(*keys[i].get<absl::string_view>());
      }
      return container->Initialize(keys_vec, values);
    }
  };
};

template <typename ValueType>
using ResourceOp = ResourceConstructionOp<
    typename StaticStringFlatHashMapFactory<
        StaticStringFlatHashMap<ValueType>>::Functor,
    // These are the aliases.
    LookupInterface<ValueType*, const absl::string_view&>,
    LookupWithPrefetchInterface<absl::Span<ValueType>,
                                absl::Span<const absl::string_view>>,
    LookupInterface<ValueType*, const string&>,
    LookupWithPrefetchInterface<absl::Span<ValueType>,
                                absl::Span<const string>>,
    SizeInterface>;

#define REGISTER_STRING_KERNEL(table_value_dtype)                  \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("StaticStringFlatHashMap")                              \
          .Device(DEVICE_CPU)                                      \
          .TypeConstraint<Variant>("heterogeneous_key_dtype")      \
          .TypeConstraint<table_value_dtype>("table_value_dtype"), \
      ResourceOp<table_value_dtype>);

REGISTER_STRING_KERNEL(int32);
REGISTER_STRING_KERNEL(int64);

#undef REGISTER_STRING_KERNEL

}  // namespace tables
}  // namespace tensorflow
