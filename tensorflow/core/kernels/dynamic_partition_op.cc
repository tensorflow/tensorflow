// See docs in ../ops/data_flow_ops.cc.

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/public/tensor.h"

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
    OP_REQUIRES(c, TensorShapeUtils::StartsWith((*data)->shape(),
                                                (*partitions)->shape()),
                errors::InvalidArgument(
                    "data.shape must start with partitions.shape, ",
                    "got data.shape = ", (*data)->shape().ShortDebugString(),
                    ", partitions.shape = ",
                    (*partitions)->shape().ShortDebugString()));

    // Count how many occurrences of each partition id we have in partitions
    gtl::InlinedVector<int, 32> partition_count(num_partitions_);
    auto e_partitions = (*partitions)->flat<int32>();
    const int64 N = e_partitions.dimension(0);
    for (int64 i = 0; i < N; i++) {
      const int32 p = e_partitions(i);
      OP_REQUIRES(c, p >= 0 && p < num_partitions_,
                  errors::InvalidArgument(
                      "partitions", SliceString((*partitions)->shape(), i),
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

  static string SliceString(const TensorShape& shape, const int64 flat) {
    // Special case rank 0 and 1
    const int dims = shape.dims();
    if (dims == 0) return "";
    if (dims == 1) return strings::StrCat("[", flat, "]");

    // Compute strides
    gtl::InlinedVector<int64, 32> strides(dims);
    strides.back() = 1;
    for (int i = dims - 2; i >= 0; i--) {
      strides[i] = strides[i + 1] * shape.dim_size(i + 1);
    }

    // Unflatten index
    int64 left = flat;
    string result;
    for (int i = 0; i < dims; i++) {
      strings::StrAppend(&result, i ? "," : "[", left / strides[i]);
      left %= strides[i];
    }
    strings::StrAppend(&result, "]");
    return result;
  }
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
                                   Eigen::Aligned> > out_vec;
      for (int p = 0; p < num_partitions_; p++) {
        out_vec.push_back(outputs[p]->vec<T>());
      }
      for (int64 i = 0; i < N; i++) {
        const int32 p = e_partitions(i);
        out_vec[p](output_index[p]) = data_flat(i);
        output_index[p]++;
      }
    } else {
      // If data has extra dimensions, use Eigen slices
      std::vector<Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>,
                                   Eigen::Aligned> > out_flat;
      for (int p = 0; p < num_partitions_; p++) {
        out_flat.push_back(outputs[p]->flat_outer_dims<T>());
      }

      // Walk through data and copy the data to the appropriate output tensor
      const int64 slice_size = data->NumElements() / N;
      const auto data_flat = data->shaped<T, 2>({N, slice_size});
      Eigen::DSizes<Eigen::DenseIndex, 2> sizes(1, slice_size);
      for (int64 i = 0; i < N; i++) {
        const int32 p = e_partitions(i);
        // outputs[p][output_index[p]++] = data[i]
        Eigen::DSizes<Eigen::DenseIndex, 2> out_indices(output_index[p], 0);
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
