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
#include "tensorflow/core/kernels/lookup_tables/op_kernel_templates.h"
#include "tensorflow/core/kernels/lookup_tables/resource_interface_templates.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {
namespace tables {

// Map x -> (Fingerprint64(x) % num_oov_buckets) + offset.
// num_oov_buckets and offset are node attributes provided at construction
// time.
template <typename KeyType, typename ValueType>
class Fingerprint64Map final
    : public virtual LookupInterface<ValueType*, const KeyType&>,
      public virtual LookupWithPrefetchInterface<absl::Span<ValueType>,
                                                 absl::Span<const KeyType>> {
 public:
  using key_type = KeyType;

  Fingerprint64Map(int64 num_oov_buckets, int64 offset)
      : num_oov_buckets_(num_oov_buckets), offset_(offset) {}

  Status Lookup(const KeyType& key_to_find, ValueType* value) const override {
    *value = LookupHelper(key_to_find);
    return Status::OK();
  }

  Status Lookup(absl::Span<const KeyType> keys, absl::Span<ValueType> values,
                int64 prefetch_lookahead) const override {
    if (ABSL_PREDICT_FALSE(keys.size() != values.size())) {
      return errors::InvalidArgument(
          "keys and values do not have the same number of elements (found ",
          keys.size(), " vs ", values.size(), ").");
    }
    for (size_t i = 0; i < keys.size(); ++i) {
      values[i] = LookupHelper(keys[i]);
    }
    return Status::OK();
  }

  mutex* GetMutex() const override { return nullptr; }

  string DebugString() const override { return __PRETTY_FUNCTION__; }

 private:
  ABSL_ATTRIBUTE_ALWAYS_INLINE ValueType
  LookupHelper(const KeyType& key_to_find) const {
    // This can cause a downcast.
    return static_cast<ValueType>(Fingerprint64(key_to_find) %
                                  num_oov_buckets_) +
           offset_;
  }

  const int64 num_oov_buckets_;
  const int64 offset_;
  TF_DISALLOW_COPY_AND_ASSIGN(Fingerprint64Map);
};

template <typename Fingerprint64Map>
struct Fingerprint64MapFactory {
  struct Functor {
    using resource_type = Fingerprint64Map;

    static Status AllocateContainer(OpKernelContext* ctx, OpKernel* kernel,
                                    Fingerprint64Map** container) {
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

template <typename KeyType, typename ValueType>
using ResourceOp = ResourceConstructionOp<
    typename Fingerprint64MapFactory<
        Fingerprint64Map<KeyType, ValueType>>::Functor,
    // These are the aliases.
    LookupInterface<ValueType*, const KeyType&>,
    LookupWithPrefetchInterface<absl::Span<ValueType>,
                                absl::Span<const KeyType>>>;

#define REGISTER_STRING_KERNEL(ValueType)                     \
  REGISTER_KERNEL_BUILDER(                                    \
      Name("Fingerprint64Map")                                \
          .Device(DEVICE_CPU)                                 \
          .TypeConstraint<Variant>("heterogeneous_key_dtype") \
          .TypeConstraint<ValueType>("table_value_dtype"),    \
      ResourceOp<absl::string_view, ValueType>);              \
  REGISTER_KERNEL_BUILDER(                                    \
      Name("Fingerprint64Map")                                \
          .Device(DEVICE_CPU)                                 \
          .TypeConstraint<string>("heterogeneous_key_dtype")  \
          .TypeConstraint<ValueType>("table_value_dtype"),    \
      ResourceOp<string, ValueType>);

REGISTER_STRING_KERNEL(int32);
REGISTER_STRING_KERNEL(int64);

#undef REGISTER_STRING_KERNEL

}  // namespace tables
}  // namespace tensorflow
