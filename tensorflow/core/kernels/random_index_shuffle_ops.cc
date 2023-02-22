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

#include <algorithm>
#include <array>
#include <memory>

#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/random_index_shuffle.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace tensorflow {
namespace {

constexpr absl::string_view kRounds = "rounds";

template <typename DType>
std::array<uint32_t, 3> CastSeedFrom(const Tensor& seed_t, const int row) {
  const auto seed_vals = seed_t.flat<DType>();
  return {static_cast<uint32_t>(seed_vals(3 * row)),
          static_cast<uint32_t>(seed_vals(3 * row + 1)),
          static_cast<uint32_t>(seed_vals(3 * row + 2))};
}

Status GetSeed(const Tensor& seed_t, const int row,
               std::array<uint32_t, 3>* seed) {
  if (seed_t.dtype() == DT_INT32) {
    *seed = CastSeedFrom<int32_t>(seed_t, row);
  } else if (seed_t.dtype() == DT_UINT32) {
    *seed = CastSeedFrom<uint32_t>(seed_t, row);
  } else if (seed_t.dtype() == DT_INT64) {
    *seed = CastSeedFrom<int64_t>(seed_t, row);
  } else if (seed_t.dtype() == DT_UINT64) {
    *seed = CastSeedFrom<uint64_t>(seed_t, row);
  } else {
    return errors::InvalidArgument("Invalid seed type: ",
                                   DataTypeString(seed_t.dtype()));
  }
  return OkStatus();
}

template <typename IntType>
class RandomIndexShuffleOp : public OpKernel {
 public:
  explicit RandomIndexShuffleOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr(kRounds, &rounds_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& index_t = context->input(0);
    const Tensor& seed_t = context->input(1);
    const Tensor& max_index_t = context->input(2);

    const bool all_scalar =
        index_t.dims() == 0 && seed_t.dims() == 1 && max_index_t.dims() == 0;
    const int64_t num_outputs =
        std::max(std::max(index_t.NumElements(), max_index_t.NumElements()),
                 seed_t.NumElements() / 3);

    // Check shapes.
    OP_REQUIRES(context,
                index_t.dims() == 0 ||
                    (index_t.dims() == 1 && index_t.dim_size(0) == num_outputs),
                errors::InvalidArgument("Index bust be a scalar or vector."));
    OP_REQUIRES(context,
                (seed_t.dims() == 1 && seed_t.dim_size(0) == 3) ||
                    (seed_t.dims() == 2 && seed_t.dim_size(0) == num_outputs &&
                     seed_t.dim_size(1) == 3),
                errors::InvalidArgument(absl::StrFormat(
                    "Seed must be a vector of size [3] "
                    "or a matrix of size [%d, 3] but got %s.",
                    num_outputs, seed_t.shape().DebugString())));
    OP_REQUIRES(
        context,
        max_index_t.dims() == 0 ||
            (max_index_t.dims() == 1 && max_index_t.dim_size(0) == num_outputs),
        errors::InvalidArgument(
            absl::StrFormat("Maxval must be a scalar or a vector of "
                            "the same size as index but got %s",
                            max_index_t.shape().DebugString())));

    // Create output tensor.
    Tensor* new_index_t;
    if (all_scalar) {
      OP_REQUIRES_OK(
          context, context->allocate_output(0, index_t.shape(), &new_index_t));
    } else {
      TensorShape new_index_shape({num_outputs});
      OP_REQUIRES_OK(
          context, context->allocate_output(0, new_index_shape, &new_index_t));
    }

    for (int64_t i = 0; i < num_outputs; ++i) {
      const auto index =
          static_cast<uint64_t>(index_t.dims() ? index_t.vec<IntType>()(i)
                                               : index_t.scalar<IntType>()());
      const auto max_index = static_cast<uint64_t>(
          max_index_t.dims() ? max_index_t.vec<IntType>()(i)
                             : max_index_t.scalar<IntType>()());
      std::array<uint32_t, 3> seed;
      OP_REQUIRES_OK(context,
                     GetSeed(seed_t, seed_t.dims() == 1 ? 0 : i, &seed));
      const auto new_index =
          tensorflow::random::index_shuffle(index, seed, max_index, rounds_);
      new_index_t->flat<IntType>()(i) = static_cast<IntType>(new_index);
    }
  }

 private:
  int32_t rounds_;  // Number of rounds for the block cipher.

  TF_DISALLOW_COPY_AND_ASSIGN(RandomIndexShuffleOp);
};

#define REGISTER(TYPE)                                        \
  REGISTER_KERNEL_BUILDER(Name("RandomIndexShuffle")          \
                              .Device(DEVICE_CPU)             \
                              .TypeConstraint<TYPE>("dtype"), \
                          RandomIndexShuffleOp<TYPE>);

TF_CALL_int32(REGISTER);
TF_CALL_int64(REGISTER);
TF_CALL_uint32(REGISTER);
TF_CALL_uint64(REGISTER);

}  // namespace
}  // namespace tensorflow
