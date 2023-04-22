/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include <functional>
#include <unordered_map>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/bfloat16.h"

namespace tensorflow {
namespace {

typedef Eigen::ThreadPoolDevice CPUDevice;

// `UniqueOpHashMap` defines the map type that is used when elements of type
// `T` are to be uniquified. By default, we use `absl::flat_hash_map<T, TIndex>`
// as the map type. Subsequent specializations are provided for
// performance and/or correctness.
template <typename T, typename TIndex>
struct UniqueOpHashMap {
  using map_type = absl::flat_hash_map<T, TIndex>;
};

// NOTE(mrry): For `tstring` elements, we use an `absl::string_view` key to
// avoid copying the input strings into the map.
template <typename TIndex>
struct UniqueOpHashMap<tstring, TIndex> {
  using map_type = absl::flat_hash_map<absl::string_view, TIndex>;
};

// NOTE(mrry): `absl::flat_hash_map<float, ...>` does not allow `NaN` as a key,
// because `NaN != NaN`, so we fall back to `std::unordered_map<>` for
// floating-point types.
template <typename TIndex>
struct UniqueOpHashMap<float, TIndex> {
  using map_type = std::unordered_map<float, TIndex>;
};
template <typename TIndex>
struct UniqueOpHashMap<double, TIndex> {
  using map_type = std::unordered_map<double, TIndex>;
};
template <typename TIndex>
struct UniqueOpHashMap<Eigen::half, TIndex> {
  using map_type = std::unordered_map<Eigen::half, TIndex>;
};
template <typename TIndex>
struct UniqueOpHashMap<bfloat16, TIndex> {
  using map_type = std::unordered_map<bfloat16, TIndex>;
};

// `UniqueOp` computes the unique elements in the input tensor.
//
// * `T` is the element type.
// * `TIndex` is the type used to represent indices in the output, either
//   `int32` or `int64`.
template <typename T, typename TIndex>
class UniqueOp : public OpKernel {
 public:
  explicit UniqueOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    // TODO(dga):  Make unique polymorphic for returning int32 and int64
    // vectors to support large tensors.
    OP_REQUIRES(context,
                input.NumElements() <= std::numeric_limits<int32>::max(),
                errors::InvalidArgument(
                    "unique does not support input tensors larger than ",
                    std::numeric_limits<int32>::max(), " elements"));

    int64 axis = 0;
    std::vector<int64> new_sizes{1, input.NumElements(), 1};
    if (context->num_inputs() == 1) {
      OP_REQUIRES(context, TensorShapeUtils::IsVector(input.shape()),
                  errors::InvalidArgument("unique expects a 1D vector."));
    } else {
      // In case of UniqueV2, the axis is a 1D vector. The purpose is
      // to allow specifying either "no axis" or "axis". The `[]` means
      // "no axis", while `[x]` means `axis = x`.
      const Tensor& axis_tensor = context->input(1);
      OP_REQUIRES(context, TensorShapeUtils::IsVector(axis_tensor.shape()),
                  errors::InvalidArgument("axis expects a 1D vector."));
      OP_REQUIRES(
          context, axis_tensor.NumElements() <= 1,
          errors::InvalidArgument(
              "axis does not support input tensors larger than 1 elements"));
      if (axis_tensor.NumElements() == 0) {
        OP_REQUIRES(context, TensorShapeUtils::IsVector(input.shape()),
                    errors::InvalidArgument("unique expects a 1D vector."));
      } else {
        OP_REQUIRES(context,
                    (axis_tensor.dtype() == DT_INT32 ||
                     axis_tensor.dtype() == DT_INT64),
                    errors::InvalidArgument(
                        "axis tensor should be int32 or int64, but got ",
                        DataTypeString(axis_tensor.dtype())));
        if (axis_tensor.dtype() == DT_INT32) {
          axis = internal::SubtleMustCopy(axis_tensor.scalar<int32>()());
        } else {
          axis = internal::SubtleMustCopy(axis_tensor.scalar<int64>()());
        }
        axis = axis < 0 ? axis + input.dims() : axis;
        OP_REQUIRES(context, 0 <= axis && axis < input.dims(),
                    errors::InvalidArgument("axis has to be between [0, ",
                                            input.dims(), ")"));
        if (axis > 0) {
          for (int64 i = 0; i < axis; i++) {
            new_sizes[0] *= input.dim_size(i);
          }
        }
        new_sizes[1] = input.dim_size(axis);
        if (axis + 1 < input.dims()) {
          for (int64 i = axis + 1; i < input.dims(); i++) {
            new_sizes[2] *= input.dim_size(i);
          }
        }
      }
    }

    Tensor* idx = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                1, TensorShape({new_sizes[1]}), &idx));
    auto idx_vec = idx->template vec<TIndex>();

    int64 uniq_size;
    if (new_sizes[0] == 1 && new_sizes[2] == 1) {
      // Specialized and faster implementation when unique is run over single
      // elements. Here we put T directly into the map rather than ints pointing
      // to them as in the general case.
      auto Tin = input.flat<T>();
      const int64 N = static_cast<int64>(Tin.size());

      typename UniqueOpHashMap<T, TIndex>::map_type uniq;
      uniq.reserve(2 * N);
      for (Eigen::Index i = 0, j = 0; i < N; ++i) {
        auto it = uniq.emplace(Tin(i), j);
        idx_vec(i) = it.first->second;
        if (it.second) {
          ++j;
        }
      }

      uniq_size = static_cast<int64>(uniq.size());
      TensorShape output_shape(input.shape());
      output_shape.set_dim(axis, uniq_size);
      Tensor* output = nullptr;
      OP_REQUIRES_OK(context,
                     context->allocate_output(0, output_shape, &output));
      auto Tout = output->flat<T>();

      for (const auto& it : uniq) {
        Tout(it.second) = it.first;
      }
    } else {
      // General implementation when unique is run over multiple elements.
      auto Tin = input.shaped<T, 3>(new_sizes);

      auto hash_fn = [&Tin](const Eigen::Index& key) {
        size_t h = 0;
        for (Eigen::Index i = 0; i < Tin.dimension(0); i++) {
          for (Eigen::Index j = 0; j < Tin.dimension(2); j++) {
            h = Hash64Combine(h, hash<T>{}(Tin(i, key, j)));
          }
        }
        return h;
      };

      auto equal_to_fn = [&Tin](const Eigen::Index& lhs,
                                const Eigen::Index& rhs) {
        for (Eigen::Index i = 0; i < Tin.dimension(0); i++) {
          for (Eigen::Index j = 0; j < Tin.dimension(2); j++) {
            if (Tin(i, lhs, j) != Tin(i, rhs, j)) {
              return false;
            }
          }
        }
        return true;
      };

      absl::flat_hash_map<int64, int64, decltype(hash_fn),
                          decltype(equal_to_fn)>
          uniq(0, hash_fn, equal_to_fn);

      uniq.reserve(2 * Tin.dimension(1));

      for (int64 i = 0, j = 0; i < Tin.dimension(1); ++i) {
        auto it = uniq.emplace(i, j);
        idx_vec(i) = it.first->second;
        if (it.second) {
          ++j;
        }
      }

      uniq_size = static_cast<int64>(uniq.size());
      new_sizes[1] = uniq_size;
      TensorShape output_shape(input.shape());
      output_shape.set_dim(axis, uniq_size);
      Tensor* output = nullptr;
      OP_REQUIRES_OK(context,
                     context->allocate_output(0, output_shape, &output));
      auto Tout = output->shaped<T, 3>(new_sizes);

      for (auto it : uniq) {
        Tout.chip(it.second, 1) = Tin.chip(it.first, 1);
      }
    }

    if (num_outputs() > 2) {
      Tensor* output = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(
                                  2, TensorShape({uniq_size}), &output));
      auto count_output_vec = output->template vec<TIndex>();
      count_output_vec.setZero();
      const int N = idx_vec.size();
      for (int64 i = 0; i < N; ++i) {
        count_output_vec(idx_vec(i))++;
      }
    }
  }
};

#define REGISTER_UNIQUE(type)                                    \
  REGISTER_KERNEL_BUILDER(Name("Unique")                         \
                              .Device(DEVICE_CPU)                \
                              .TypeConstraint<type>("T")         \
                              .TypeConstraint<int32>("out_idx"), \
                          UniqueOp<type, int32>);                \
  REGISTER_KERNEL_BUILDER(Name("Unique")                         \
                              .Device(DEVICE_CPU)                \
                              .TypeConstraint<type>("T")         \
                              .TypeConstraint<int64>("out_idx"), \
                          UniqueOp<type, int64>);                \
  REGISTER_KERNEL_BUILDER(Name("UniqueV2")                       \
                              .Device(DEVICE_CPU)                \
                              .TypeConstraint<type>("T")         \
                              .TypeConstraint<int32>("out_idx"), \
                          UniqueOp<type, int32>);                \
  REGISTER_KERNEL_BUILDER(Name("UniqueV2")                       \
                              .Device(DEVICE_CPU)                \
                              .TypeConstraint<type>("T")         \
                              .TypeConstraint<int64>("out_idx"), \
                          UniqueOp<type, int64>);                \
  REGISTER_KERNEL_BUILDER(Name("UniqueWithCounts")               \
                              .Device(DEVICE_CPU)                \
                              .TypeConstraint<type>("T")         \
                              .TypeConstraint<int32>("out_idx"), \
                          UniqueOp<type, int32>)                 \
  REGISTER_KERNEL_BUILDER(Name("UniqueWithCounts")               \
                              .Device(DEVICE_CPU)                \
                              .TypeConstraint<type>("T")         \
                              .TypeConstraint<int64>("out_idx"), \
                          UniqueOp<type, int64>);                \
  REGISTER_KERNEL_BUILDER(Name("UniqueWithCountsV2")             \
                              .Device(DEVICE_CPU)                \
                              .TypeConstraint<type>("T")         \
                              .TypeConstraint<int32>("out_idx"), \
                          UniqueOp<type, int32>)                 \
  REGISTER_KERNEL_BUILDER(Name("UniqueWithCountsV2")             \
                              .Device(DEVICE_CPU)                \
                              .TypeConstraint<type>("T")         \
                              .TypeConstraint<int64>("out_idx"), \
                          UniqueOp<type, int64>)
TF_CALL_REAL_NUMBER_TYPES(REGISTER_UNIQUE);
REGISTER_UNIQUE(tstring)
REGISTER_UNIQUE(bool)
#undef REGISTER_UNIQUE

}  // namespace
}  // namespace tensorflow
