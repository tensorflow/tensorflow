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

#include <type_traits>
#include "absl/strings/string_view.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/kernels/lookup_tables/op_kernel_templates.h"
#include "tensorflow/core/kernels/lookup_tables/resource_interface_templates.h"
#include "tensorflow/core/kernels/string_view_variant_wrapper.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace tables {

template <typename KeyType, typename ValueType>
struct TensorInsertFactory {
  class Functor {
   public:
    // If KeyType is not 'valid' then use the value it wraps as the table key
    // type.
    using resource_type = InsertOrAssignInterface<
        absl::Span<const ValueType>,
        typename absl::conditional_t<
            IsValidDataType<KeyType>::value, absl::Span<const KeyType>,
            absl::Span<const typename KeyType::value_type>>>;

    static Status TensorInsert(const Tensor& keys, const Tensor& values,
                               resource_type* table) {
      if (keys.NumElements() != values.NumElements()) {
        return errors::InvalidArgument(
            "OpKernel tried to map keys vector of size ", keys.NumElements(),
            " to values vector of size ", values.NumElements());
      }
      return TensorInsertHelper(keys, values, table);
    }

   private:
    // keys and *values arguments to TensorInsert must have the same number of
    // elements. This is guaranteed above.

    // 'Simple' types below are types which are natively supported in TF.
    // Non-variant KeyType which is the same as Container::key_type.
    // No need to static_cast.
    template <typename SfinaeArg = KeyType>
    static absl::enable_if_t<IsValidDataType<SfinaeArg>::value, Status>
    TensorInsertHelper(const Tensor& keys, const Tensor& values,
                       resource_type* table) {
      return table->InsertOrAssign(keys.flat<KeyType>(),
                                   values.flat<ValueType>());
    }

    // Variant KeyType; the wrapped type is convertible to
    // Container::key_type.
    template <typename VariantSubType = KeyType>
    static absl::enable_if_t<!IsValidDataType<VariantSubType>::value, Status>
    TensorInsertHelper(const Tensor& keys, const Tensor& values,
                       resource_type* table) {
      const auto keys_flat = keys.flat<Variant>();
      std::vector<typename VariantSubType::value_type> keys_vec;
      keys_vec.reserve(keys_flat.size());
      for (size_t i = 0; i < keys_flat.size(); ++i) {
        keys_vec.emplace_back(
            *keys_flat(i).get<typename VariantSubType::value_type>());
      }
      return table->InsertOrAssign(keys_vec, values.flat<ValueType>());
    }
  };
};

template <typename KeyType, typename ValueType>
using InsertOp = LookupTableInsertOp<
    typename TensorInsertFactory<KeyType, ValueType>::Functor>;

template <typename KeyType, typename ValueType>
struct TensorLookupFactory {
  class Functor {
   public:
    // If KeyType is not 'valid' then use the value it wraps as the table key
    // type.
    using resource_type = LookupWithPrefetchInterface<
        absl::Span<ValueType>,
        typename absl::conditional_t<
            IsValidDataType<KeyType>::value, absl::Span<const KeyType>,
            absl::Span<const typename KeyType::value_type>>>;

    static Status TensorLookup(const resource_type& table, const Tensor& keys,
                               const int64 prefetch_lookahead,
                               const int64 num_keys_per_thread,
                               thread::ThreadPool* threadpool, Tensor* values) {
      if (keys.NumElements() != values->NumElements()) {
        return errors::InvalidArgument(
            "OpKernel tried to map keys vector of size ", keys.NumElements(),
            " to values vector of size ", values->NumElements());
      }
      return TensorLookupHelper(table, keys, prefetch_lookahead,
                                num_keys_per_thread, threadpool, values);
    }

   private:
    // keys and *values arguments to TensorLookup must have the same number of
    // elements. This is guaranteed above.

    // 'Simple' types below are types which are natively supported in TF.
    template <typename SfinaeArg = KeyType>
    static absl::enable_if_t<IsValidDataType<SfinaeArg>::value, Status>
    TensorLookupHelper(const resource_type& table, const Tensor& keys,
                       const int64 prefetch_lookahead,
                       const int64 num_keys_per_thread,
                       thread::ThreadPool* threadpool, Tensor* values) {
      const auto keys_flat = keys.flat<KeyType>();
      auto key_span = absl::MakeSpan(keys_flat);
      auto value_span = absl::MakeSpan(values->flat<ValueType>().data(),
                                       values->NumElements());
      return MultithreadedTensorLookup(table, prefetch_lookahead,
                                       num_keys_per_thread, key_span,
                                       value_span, threadpool);
    }

    // Non-simple KeyType. We'll try an implicit conversion to
    // Container::key_type.
    template <typename VariantSubType = KeyType>
    static absl::enable_if_t<!IsValidDataType<VariantSubType>::value, Status>
    TensorLookupHelper(const resource_type& table, const Tensor& keys,
                       const int64 prefetch_lookahead,
                       const int64 num_keys_per_thread,
                       thread::ThreadPool* threadpool, Tensor* values) {
      const auto keys_flat = keys.flat<Variant>();
      std::vector<typename VariantSubType::value_type> keys_vec;
      const auto keys_size = keys_flat.size();
      keys_vec.reserve(keys_size);
      for (size_t i = 0; i < keys_size; ++i) {
        keys_vec.emplace_back(*keys_flat(i).get<VariantSubType>()->get());
      }
      absl::Span<const typename VariantSubType::value_type> key_span(keys_vec);
      auto value_span = absl::MakeSpan(values->flat<ValueType>().data(),
                                       values->NumElements());
      return MultithreadedTensorLookup(table, prefetch_lookahead,
                                       num_keys_per_thread, key_span,
                                       value_span, threadpool);
    }

    // Wrapper around table.BatchLookup which permits sharding across cores.
    template <typename K, typename V>
    static Status MultithreadedTensorLookup(const resource_type& table,
                                            int64 prefetch_lookahead,
                                            int64 num_keys_per_thread, K keys,
                                            V values,
                                            thread::ThreadPool* threadpool) {
      mutex temp_mutex;  // Protect status.
      Status status;
      auto lookup_keys = [&](int64 begin, int64 end) {
        auto temp_status = table.Lookup(keys.subspan(begin, end - begin),
                                        values.subspan(begin, end - begin),
                                        prefetch_lookahead);
        if (ABSL_PREDICT_FALSE(!temp_status.ok())) {
          mutex_lock lock(temp_mutex);
          status.Update(temp_status);
        }
      };
      threadpool->TransformRangeConcurrently(
          num_keys_per_thread /* block_size */, keys.size(), lookup_keys);
      return status;
    }
  };
};

template <typename KeyType, typename ValueType>
using LookupOp = LookupTableFindOp<
    typename TensorLookupFactory<KeyType, ValueType>::Functor>;

struct TableSizeFunctor {
  using resource_type = SizeInterface;

  static Status Size(const SizeInterface& table, uint64* size) {
    *size = table.Size();
    return Status::OK();
  }
};

#define REGISTER_STRING_KERNEL(table_value_dtype)                     \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("LookupTableInsertOrAssignOp")                             \
          .Device(DEVICE_CPU)                                         \
          .TypeConstraint<string>("insert_key_tensor_dtype")          \
          .TypeConstraint<table_value_dtype>("table_value_dtype"),    \
      InsertOp<string, table_value_dtype>);                           \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("LookupTableInsertOrAssignOp")                             \
          .Device(DEVICE_CPU)                                         \
          .TypeConstraint<Variant>("insert_key_tensor_dtype")         \
          .TypeConstraint<table_value_dtype>("table_value_dtype"),    \
      InsertOp<StringViewVariantWrapper, table_value_dtype>);         \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("LookupTableFindOp")                                       \
          .Device(DEVICE_CPU)                                         \
          .TypeConstraint<string>("lookup_key_tensor_dtype")          \
          .TypeConstraint<table_value_dtype>("table_value_dtype"),    \
      LookupOp<string, table_value_dtype>);                           \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("LookupTableFindOp")                                       \
          .Device(DEVICE_CPU)                                         \
          .TypeConstraint<Variant>("lookup_key_tensor_dtype")         \
          .TypeConstraint<table_value_dtype>("table_value_dtype"),    \
      LookupOp<StringViewVariantWrapper, table_value_dtype>);         \
  REGISTER_KERNEL_BUILDER(Name("ContainerSizeOp").Device(DEVICE_CPU), \
                          ContainerSizeOp<TableSizeFunctor>);

REGISTER_STRING_KERNEL(int32);
REGISTER_STRING_KERNEL(int64);

#undef REGISTER_STRING_KERNEL

}  // namespace tables
}  // namespace tensorflow
