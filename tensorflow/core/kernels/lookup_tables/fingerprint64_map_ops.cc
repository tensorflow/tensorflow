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

#include "absl/strings/string_view.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/lookup_tables/table_op_utils.h"
#include "tensorflow/core/kernels/lookup_tables/table_resource_utils.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {
namespace tables {

// Map x -> (Fingerprint64(x) % num_oov_buckets) + offset.
// num_oov_buckets and offset are node attributes provided at construction
// time.
template <class HeterogeneousKeyType, class ValueType>
class Fingerprint64Map final
    : public LookupTableInterface<HeterogeneousKeyType, ValueType> {
 public:
  Fingerprint64Map(int64 num_oov_buckets, int64 offset)
      : num_oov_buckets_(num_oov_buckets), offset_(offset) {}

  mutex* GetMutex() const override { return nullptr; }

  bool UnsafeInsertOrAssign(const HeterogeneousKeyType& key,
                            const ValueType& value) override {
    return true;
  }

  Status TableUnbatchedInsertStatus() const override {
    return errors::Unimplemented("Fingerprint64Map does not support inserts.");
  }

  Status BatchInsertOrAssign(absl::Span<const HeterogeneousKeyType> keys,
                             absl::Span<const ValueType> values) override {
    return errors::Unimplemented("Fingerprint64Map does not support inserts.");
  }

  ValueType UnsafeLookupKey(
      const HeterogeneousKeyType& key_to_find) const override {
    // This can cause a downcast.
    return static_cast<ValueType>(Fingerprint64(key_to_find) %
                                  num_oov_buckets_) +
           offset_;
  }

  Status TableUnbatchedLookupStatus() const override { return Status::OK(); }

  Status BatchLookup(absl::Span<const HeterogeneousKeyType> keys,
                     absl::Span<ValueType> values,
                     int64 prefetch_lookahead) const override {
    if (ABSL_PREDICT_FALSE(keys.size() != values.size())) {
      return errors::InvalidArgument(
          "keys and values do not have the same number of elements (found ",
          keys.size(), " vs ", values.size(), ").");
    }
    for (size_t i = 0; i < keys.size(); ++i) {
      values[i] = Fingerprint64Map::UnsafeLookupKey(keys[i]);
    }
    return Status::OK();
  }

  const absl::optional<const ValueType> DefaultValue() const override {
    return {};
  }

  void UnsafePrefetchKey(
      const HeterogeneousKeyType& key_to_find) const override {}

  size_t UnsafeSize() const override { return 0; }

  Status SizeStatus() const override {
    return errors::Unimplemented(
        "Fingerprint64Map does not have a concept of size.");
  }

  bool UnsafeContainsKey(
      const HeterogeneousKeyType& key_to_find) const override {
    return true;
  }

 private:
  const int64 num_oov_buckets_;
  const int64 offset_;
  TF_DISALLOW_COPY_AND_ASSIGN(Fingerprint64Map);
};

template <typename Fingerprint64Map>
struct Fingerprint64MapFactory {
  struct Functor {
    template <typename ContainerBase>
    static Status AllocateContainer(OpKernelContext* ctx, OpKernel* kernel,
                                    ContainerBase** container) {
      int64 num_oov_buckets;
      int64 offset;
      TF_RETURN_IF_ERROR(
          GetNodeAttr(kernel->def(), "num_oov_buckets", &num_oov_buckets));
      TF_RETURN_IF_ERROR(GetNodeAttr(kernel->def(), "offset", &offset));
      *container = new Fingerprint64Map(num_oov_buckets, offset);
      return Status::OK();
    }
  };
};

#define REGISTER_STRING_KERNEL(table_value_dtype)                             \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("Fingerprint64Map")                                                \
          .Device(DEVICE_CPU)                                                 \
          .TypeConstraint<Variant>("heterogeneous_key_dtype")                 \
          .TypeConstraint<table_value_dtype>("table_value_dtype"),            \
      ResourceConstructionOp<                                                 \
          LookupTableInterface<absl::string_view, table_value_dtype>,         \
          Fingerprint64MapFactory<Fingerprint64Map<                           \
              absl::string_view, table_value_dtype>>::Functor>);              \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("Fingerprint64Map")                                                \
          .Device(DEVICE_CPU)                                                 \
          .TypeConstraint<string>("heterogeneous_key_dtype")                  \
          .TypeConstraint<table_value_dtype>("table_value_dtype"),            \
      ResourceConstructionOp<LookupTableInterface<string, table_value_dtype>, \
                             Fingerprint64MapFactory<Fingerprint64Map<        \
                                 string, table_value_dtype>>::Functor>);

REGISTER_STRING_KERNEL(int32);
REGISTER_STRING_KERNEL(int64);

#undef REGISTER_STRING_KERNEL

}  // namespace tables
}  // namespace tensorflow
