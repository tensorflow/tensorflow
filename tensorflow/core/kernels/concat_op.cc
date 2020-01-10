// See docs in ../ops/array_ops.cc.

#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/concat_op.h"
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/public/tensor.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/public/status.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

// --------------------------------------------------------------------------
template <typename Device, typename T>
class ConcatOp : public OpKernel {
 public:
  typedef std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>
      ConstMatrixVector;

  explicit ConcatOp(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* c) override {
    const Tensor* concat_dim_tensor;
    OP_REQUIRES_OK(c, c->input("concat_dim", &concat_dim_tensor));
    OP_REQUIRES(
        c, TensorShapeUtils::IsLegacyScalar(concat_dim_tensor->shape()),
        errors::InvalidArgument(
            "Concat dim tensor should be a scalar integer, but got shape ",
            concat_dim_tensor->shape().DebugString()));
    const int32 concat_dim = concat_dim_tensor->scalar<int32>()();
    OpInputList values;
    OP_REQUIRES_OK(c, c->input_list("values", &values));
    const int N = values.size();
    const int input_dims = values[0].dims();
    const TensorShape& input_shape = values[0].shape();
    OP_REQUIRES(
        c, (0 <= concat_dim && concat_dim < input_dims) ||
               (kAllowLegacyScalars && concat_dim == 0),
        errors::InvalidArgument(
            "ConcatOp : Expected concatenating dimensions in the range [", 0,
            ", ", input_dims, "), but got ", concat_dim));

    // Note that we reduce the concat of n-dimensional tensors into a two
    // dimensional concat. Assuming the dimensions of any input/output
    // tensor are {x0, x1,...,xn-1, y0, y1,...,ym-1}, where the concat is along
    // the dimension indicated with size y0, we flatten it to {x, y}, where y =
    // Prod_i(yi) and x = ((n > 0) ? Prod_i(xi) : 1).
    ConstMatrixVector inputs_flat;
    inputs_flat.reserve(N);
    int64 inputs_flat_dim0 = 1;
    for (int d = 0; d < concat_dim; ++d) {
      inputs_flat_dim0 *= input_shape.dim_size(d);
    }
    int output_concat_dim = 0;
    const bool input_is_scalar = TensorShapeUtils::IsLegacyScalar(input_shape);
    for (int i = 0; i < N; ++i) {
      const auto in = values[i];
      const bool in_is_scalar = TensorShapeUtils::IsLegacyScalar(in.shape());
      OP_REQUIRES(
          c, in.dims() == input_dims || (input_is_scalar && in_is_scalar),
          errors::InvalidArgument(
              "ConcatOp : Ranks of all input tensors should match: shape[0] = ",
              input_shape.ShortDebugString(), " vs. shape[", i, "] = ",
              in.shape().ShortDebugString()));
      for (int j = 0; j < input_dims; ++j) {
        if (j == concat_dim) {
          continue;
        }
        OP_REQUIRES(
            c, in.dim_size(j) == input_shape.dim_size(j),
            errors::InvalidArgument(
                "ConcatOp : Dimensions of inputs should match: shape[0] = ",
                input_shape.ShortDebugString(), " vs. shape[", i, "] = ",
                in.shape().ShortDebugString()));
      }
      if (in.NumElements() > 0) {
        int64 inputs_flat_dim1 = in.NumElements() / inputs_flat_dim0;
        inputs_flat.emplace_back(new typename TTypes<T, 2>::ConstMatrix(
            in.shaped<T, 2>({inputs_flat_dim0, inputs_flat_dim1})));
      }
      // TODO(irving): Remove check once !kAllowLegacyScalars
      output_concat_dim += in.dims() > 0 ? in.dim_size(concat_dim) : 1;
    }

    TensorShape output_shape(input_shape);
    // TODO(irving): Remove rank 0 case once !kAllowLegacyScalars
    if (output_shape.dims() == 0) {
      output_shape.AddDim(output_concat_dim);
    } else {
      output_shape.set_dim(concat_dim, output_concat_dim);
    }
    Tensor* output = nullptr;
    OP_REQUIRES_OK(c, c->allocate_output(0, output_shape, &output));
    if (output->NumElements() > 0) {
      int64 output_dim1 = output->NumElements() / inputs_flat_dim0;
      auto output_flat = output->shaped<T, 2>({inputs_flat_dim0, output_dim1});
      if (std::is_same<Device, GPUDevice>::value) {
        ConcatGPU<T>(c->eigen_gpu_device(), inputs_flat, &output_flat);
      } else {
        ConcatCPU<T>(c->device(), inputs_flat, &output_flat);
      }
    }
  }
};

#define REGISTER_CONCAT(type)                            \
  REGISTER_KERNEL_BUILDER(Name("Concat")                 \
                              .Device(DEVICE_CPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("concat_dim"), \
                          ConcatOp<CPUDevice, type>)

TF_CALL_ALL_TYPES(REGISTER_CONCAT);
REGISTER_CONCAT(quint8);
REGISTER_CONCAT(qint8);
REGISTER_CONCAT(qint32);
REGISTER_CONCAT(bfloat16);

#undef REGISTER_CONCAT

#if GOOGLE_CUDA

#define REGISTER_GPU(type)                               \
  REGISTER_KERNEL_BUILDER(Name("Concat")                 \
                              .Device(DEVICE_GPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("concat_dim"), \
                          ConcatOp<GPUDevice, type>)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU);
#undef REGISTER_GPU

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("Concat")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("T")
                            .HostMemory("concat_dim")
                            .HostMemory("values")
                            .HostMemory("output"),
                        ConcatOp<CPUDevice, int32>);

#endif  // GOOGLE_CUDA

}  // namespace tensorflow
