// See docs in ../ops/image_ops.cc
#define EIGEN_USE_THREADS

#include <memory>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/public/status.h"
#include "tensorflow/core/public/tensor.h"
#include "tensorflow/core/public/tensor_shape.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Device, typename T>
class ResizeNearestNeighborOp : public OpKernel {
 public:
  explicit ResizeNearestNeighborOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    OP_REQUIRES(context, input.dims() == 4,
                errors::InvalidArgument("input must be 4-dimensional",
                                        input.shape().ShortDebugString()));
    const Tensor& shape_t = context->input(1);
    OP_REQUIRES(context, shape_t.dims() == 1,
                errors::InvalidArgument("shape_t must be 1-dimensional",
                                        shape_t.shape().ShortDebugString()));
    OP_REQUIRES(context, shape_t.NumElements() == 2,
                errors::InvalidArgument("shape_t must have two elements",
                                        shape_t.shape().ShortDebugString()));

    auto Svec = shape_t.vec<int32>();
    // Initialize shape to the batch size of the input, then add
    // the rest of the dimensions
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, TensorShape({input.dim_size(0), Svec(0),
                                                Svec(1), input.dim_size(3)}),
                                &output));

    const int64 batch_size = input.dim_size(0);
    const int64 in_height = input.dim_size(1);
    const int64 in_width = input.dim_size(2);
    const int64 channels = input.dim_size(3);
    const int64 out_height = output->dim_size(1);
    const int64 out_width = output->dim_size(2);

    typename TTypes<T, 4>::ConstTensor input_data = input.tensor<T, 4>();
    typename TTypes<T, 4>::Tensor output_data = output->tensor<T, 4>();

    const float height_scale = in_height / static_cast<float>(out_height);
    const float width_scale = in_width / static_cast<float>(out_width);

    for (int b = 0; b < batch_size; ++b) {
      for (int y = 0; y < out_height; ++y) {
        const int in_y = std::min(static_cast<int64>(floorf(y * height_scale)),
                                  (in_height - 1));
        for (int x = 0; x < out_width; ++x) {
          const int in_x = std::min(static_cast<int64>(floorf(x * width_scale)),
                                    (in_width - 1));
          for (int c = 0; c < channels; ++c) {
            output_data(b, y, x, c) = input_data(b, in_y, in_x, c);
          }
        }
      }
    }
  }
};

#define REGISTER_KERNEL(T)                              \
  REGISTER_KERNEL_BUILDER(Name("ResizeNearestNeighbor") \
                              .Device(DEVICE_CPU)       \
                              .TypeConstraint<T>("T")   \
                              .HostMemory("size"),      \
                          ResizeNearestNeighborOp<CPUDevice, T>);

REGISTER_KERNEL(uint8);
REGISTER_KERNEL(int8);
REGISTER_KERNEL(int32);
REGISTER_KERNEL(float);
REGISTER_KERNEL(double);
#undef REGISTER_KERNEL

}  // namespace tensorflow
