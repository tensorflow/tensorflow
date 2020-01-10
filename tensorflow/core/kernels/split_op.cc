// See docs in ../ops/array_ops.cc.

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/split_op.h"

#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/public/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/public/tensor.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class SplitOp : public OpKernel {
 public:
  explicit SplitOp(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* context) override {
    const int32 split_dim = context->input(0).flat<int32>()(0);
    const int32 num_split = num_outputs();
    const Tensor& input = context->input(1);
    const TensorShape& input_shape = input.shape();

    OP_REQUIRES(
        context, 0 <= split_dim && split_dim < input_shape.dims(),
        errors::InvalidArgument("0 <= split_dim < number of input dimensions (",
                                input_shape.dims(), "), but got ", split_dim));

    OP_REQUIRES(
        context, num_split > 0,
        errors::InvalidArgument(
            "Number of ways to split should be > 0, but got ", num_split));

    OP_REQUIRES(context, input_shape.dim_size(split_dim) % num_split == 0,
                errors::InvalidArgument(
                    "Number of ways to split should evenly divide the split "
                    "dimension, but got split_dim ",
                    split_dim, " (size = ", input_shape.dim_size(split_dim),
                    ") ", "and num_split ", num_split));

    // Special case 1: num_split == 1. Nothing to do.
    if (num_split == 1) {
      VLOG(1) << "Split identity";
      context->set_output(0, context->input(1));
      return;
    }

    // Special case 2: split along the 1st dimension. We can share the
    // underlying buffer.
    //
    // Apply this optimization conservatively: if input is aligned,
    // the resulting tensors must be aligned. It's conservative
    // because if the immediate consumer of the resulting tensors are
    // not using eigen for computation, its perfectly fine to avoid
    // the copying.
    if ((split_dim == 0) && IsInnerDimsSizeAligned<T>(input_shape)) {
      VLOG(1) << "Slice dim 0: " << input_shape.DebugString();
      const int64 delta = input_shape.dim_size(0) / num_split;
      for (int i = 0; i < num_split; ++i) {
        context->set_output(i, input.Slice(i * delta, (i + 1) * delta));
      }
      return;
    }

    int32 prefix_dim_size = 1;
    for (int i = 0; i < split_dim; ++i) {
      prefix_dim_size *= input_shape.dim_size(i);
    }

    int32 split_dim_size = input_shape.dim_size(split_dim);

    int32 suffix_dim_size = 1;
    for (int i = split_dim + 1; i < input_shape.dims(); ++i) {
      suffix_dim_size *= input_shape.dim_size(i);
    }

    auto input_reshaped =
        input.shaped<T, 3>({prefix_dim_size, split_dim_size, suffix_dim_size});

    const int32 split_dim_output_size = split_dim_size / num_split;
    TensorShape output_shape(input_shape);
    output_shape.set_dim(split_dim, split_dim_output_size);

    Eigen::DSizes<ptrdiff_t, 3> indices{0, 0, 0};
    Eigen::DSizes<ptrdiff_t, 3> sizes{prefix_dim_size, split_dim_output_size,
                                      suffix_dim_size};

    for (int i = 0; i < num_split; ++i) {
      Tensor* result = nullptr;
      OP_REQUIRES_OK(context,
                     context->allocate_output(i, output_shape, &result));
      if (prefix_dim_size * split_dim_output_size * suffix_dim_size > 0) {
        Eigen::DSizes<ptrdiff_t, 3> slice_indices;
        Eigen::DSizes<ptrdiff_t, 3> slice_sizes;
        for (int j = 0; j < 3; ++j) {
          slice_indices[j] = indices[j];
          slice_sizes[j] = sizes[j];
        }

        auto result_shaped = result->shaped<T, 3>(
            {prefix_dim_size, split_dim_output_size, suffix_dim_size});

        functor::Split<Device, T>()(context->eigen_device<Device>(),
                                    result_shaped, input_reshaped,
                                    slice_indices, slice_sizes);
      }
      indices[1] += split_dim_output_size;
    }
  }
};

#define REGISTER_SPLIT(type)                             \
  REGISTER_KERNEL_BUILDER(Name("Split")                  \
                              .Device(DEVICE_CPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("split_dim"),  \
                          SplitOp<CPUDevice, type>)

TF_CALL_ALL_TYPES(REGISTER_SPLIT);

#undef REGISTER_SPLIT

#if GOOGLE_CUDA

#define REGISTER_GPU(type)                               \
  REGISTER_KERNEL_BUILDER(Name("Split")                  \
                              .Device(DEVICE_GPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("split_dim"),  \
                          SplitOp<GPUDevice, type>)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU);
#undef REGISTER_GPU

#endif  // GOOGLE_CUDA

}  // end namespace tensorflow
