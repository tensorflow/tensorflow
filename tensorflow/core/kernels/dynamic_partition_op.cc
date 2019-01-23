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

// See docs in ../ops/data_flow_ops.cc.

#include <vector>
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/util/util.h"

namespace tensorflow {

// Shared code that is not dependent on the type of T.  We do this to reduce
// code size by not duplicating all this for all T (float, double, int32, etc.)
class DynamicPartitionOp_Shared : public OpKernel {
 public:
  explicit DynamicPartitionOp_Shared(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("num_partitions", &num_partitions_));
    //   QUESTION: It'd be nice to support DT_INT16, DT_UINT8, etc.
    //   to input[1].  Should we have the framework do some sort of
    //   integer promotion automatically, or should that be something
    //   that users have to do explicitly with a conversion operator
    //   in the graph?
  }

  void ValidateAndAllocateOutputs(OpKernelContext* c, const Tensor** data,
                                  const Tensor** partitions,
                                  OpOutputList* Tout) {
    OP_REQUIRES_OK(c, c->input("data", data));
    OP_REQUIRES_OK(c, c->input("partitions", partitions));
    OP_REQUIRES(
        c,
        TensorShapeUtils::StartsWith((*data)->shape(), (*partitions)->shape()),
        errors::InvalidArgument(
            "data.shape must start with partitions.shape, ",
            "got data.shape = ", (*data)->shape().DebugString(),
            ", partitions.shape = ", (*partitions)->shape().DebugString()));

    // Count how many occurrences of each partition id we have in partitions
    gtl::InlinedVector<int, 32> partition_count(num_partitions_);
    auto e_partitions = (*partitions)->flat<int32>();
    const int64 N = e_partitions.dimension(0);
    for (int64 i = 0; i < N; i++) {
      const int32 p = internal::SubtleMustCopy(e_partitions(i));
      OP_REQUIRES(c, FastBoundsCheck(p, num_partitions_),
                  errors::InvalidArgument(
                      "partitions", SliceDebugString((*partitions)->shape(), i),
                      " = ", p, " is not in [0, ", num_partitions_, ")"));
      partition_count[p]++;
    }

    // Allocate output tensors of the right size
    OP_REQUIRES_OK(c, c->output_list("outputs", Tout));
    for (int p = 0; p < num_partitions_; p++) {
      TensorShape shape;
      shape.AddDim(partition_count[p]);
      for (int i = (*partitions)->dims(); i < (*data)->dims(); i++) {
        shape.AddDim((*data)->dim_size(i));
      }
      Tensor* out;
      OP_REQUIRES_OK(c, Tout->allocate(p, shape, &out));
    }
  }

 protected:
  int num_partitions_;
};

template <class T>
class DynamicPartitionOp : public DynamicPartitionOp_Shared {
 public:
  explicit DynamicPartitionOp(OpKernelConstruction* c)
      : DynamicPartitionOp_Shared(c) {}
  void Compute(OpKernelContext* c) override {
    const Tensor* data;
    const Tensor* partitions;
    OpOutputList outputs;
    ValidateAndAllocateOutputs(c, &data, &partitions, &outputs);
    if (!c->status().ok()) return;
    if (num_partitions_ == 0 || data->NumElements() == 0) return;

    auto e_partitions = partitions->flat<int32>();
    const int64 N = e_partitions.dimension(0);
    gtl::InlinedVector<int, 32> output_index(num_partitions_);

    if (partitions->dims() == data->dims()) {
      // Walk through data and copy the data to the appropriate output tensor
      const auto data_flat = data->flat<T>();
      std::vector<Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>,
                                   Eigen::Aligned> >
          out_vec;
      out_vec.reserve(num_partitions_);
      for (int p = 0; p < num_partitions_; p++) {
        out_vec.push_back(outputs[p]->vec<T>());
      }
      for (int64 i = 0; i < N; i++) {
        const int32 p = internal::SubtleMustCopy(e_partitions(i));
        OP_REQUIRES(
            c, FastBoundsCheck(p, num_partitions_),
            errors::InvalidArgument("indices[", i, "] is out of range"));
        auto oi = output_index[p];
        OP_REQUIRES(c, FastBoundsCheck(oi, out_vec[p].size()),
                    errors::InvalidArgument(
                        "out_vec[", p, "] size: ", out_vec[p].size(),
                        " is not LTE output_index[", p, "] : ", oi));
        out_vec[p](oi) = data_flat(i);
        output_index[p]++;
      }
    } else {
      // If data has extra dimensions, use Eigen slices
      std::vector<Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>,
                                   Eigen::Aligned> >
          out_flat;
      out_flat.reserve(num_partitions_);
      for (int p = 0; p < num_partitions_; p++) {
        out_flat.push_back(outputs[p]->flat_outer_dims<T>());
      }

      // Walk through data and copy the data to the appropriate output tensor
      const int64 slice_size = data->NumElements() / N;
      const auto data_flat = data->shaped<T, 2>({N, slice_size});
      Eigen::DSizes<Eigen::DenseIndex, 2> sizes(1, slice_size);
      for (int64 i = 0; i < N; i++) {
        // outputs[p][output_index[p]++] = data[i]
        const int32 p = internal::SubtleMustCopy(e_partitions(i));
        OP_REQUIRES(
            c, FastBoundsCheck(p, num_partitions_),
            errors::InvalidArgument("indices[", i,
                                    "] has been asynchronously overwritten and "
                                    "is no longer in range!"));
        auto oi = output_index[p];
        OP_REQUIRES(c, FastBoundsCheck(oi, out_flat[p].dimension(0)),
                    errors::InvalidArgument("Size of output_index: ", oi,
                                            " is no longer in range."));
        Eigen::DSizes<Eigen::DenseIndex, 2> out_indices(oi, 0);
        Eigen::DSizes<Eigen::DenseIndex, 2> data_indices(i, 0);
        out_flat[p].slice(out_indices, sizes) =
            data_flat.slice(data_indices, sizes);
        output_index[p]++;
      }
    }
  }
};

#define REGISTER_DYNAMIC_PARTITION(T)                                     \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("DynamicPartition").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      DynamicPartitionOp<T>)

TF_CALL_ALL_TYPES(REGISTER_DYNAMIC_PARTITION);
#undef REGISTER_DYNAMIC_PARTITION

}  // namespace tensorflow
