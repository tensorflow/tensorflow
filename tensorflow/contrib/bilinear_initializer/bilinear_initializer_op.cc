#include "tensorflow/contrib/bilinear_initializer/bilinear_initializer_op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
namespace bilinear_initializer {

using shape_inference::ShapeHandle;

template<typename T>
BilinearInitializerOp<T>::BilinearInitializerOp(OpKernelConstruction* context)
    : OpKernel(context) {}

template<typename T>
void BilinearInitializerOp<T>::Compute(OpKernelContext* context) {
    const Tensor& input_tensor = context->input(0);
    const int* shape = &input_tensor.unaligned_flat<int>()(0);

    int factor = ceil(shape[0] / 2.);
    T center = (2 * factor - factor % 2 - 1) / 2.;

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(
      context,
      context->allocate_output(
        0,
        {shape[0], shape[1], shape[2], shape[3]},
        &output_tensor));

    T* output_ptr = &output_tensor->unaligned_flat<T>()(0);
    size_t num = shape[0] * shape[1] * shape[2] * shape[3];
    memset(output_ptr, 0, num * sizeof(T));

    auto output = output_tensor->tensor<T, 4>();
    // The four dimensions of the kernel are in the order
    // [filter_height, filter_width, input_depth, output_depth]
    for (int i = 0; i < shape[0]; i++) {      // height
      for (int j = 0; j < shape[1]; j++) {    // width
        for (int k = 0; k < shape[2]; k++) {  // input feature map
          // Since interpolation is done per-feature-map and there should be
          // no exchange of information between feature maps, only k-th input
          // feature map affects the k-th output feature map. Thus, only
          // output(i, j, k, k) contains non-zero.
          // The weight value at position (i, j) is inversely proportional to
          // its distance to the center of the kernel map.
          output(i, j, k, k) =
            (1 - fabs(i - center) / factor) *  // vertical distance
            (1 - fabs(j - center) / factor);   // horizontal distance
        }
      }
    }
  }

REGISTER_OP("BilinearInitializer")
    .Attr("T: {float, double} = DT_FLOAT")
    .Input("shape: int32")
    .Output("filter: T")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        ShapeHandle out;
        TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(0, &out));
        c->set_output(0, out);
        return Status::OK();
      })
    .Doc(R"doc(
Generates the initial value with bilinear interpolation.

Specifically, it generates a 4D kernel with shape
[filter_height, filter_width, input_depth, output_depth]

The weights are inversely proportional to their distances (w.r.t. width and
height respectively) to the center of the kernel. When applying this kernel map
in a convolution, it conducts a per-feature-map spatial weighted average, i.e.,
bilinear interpolation.

shape: A 4D tensor representing the shape of the filter to be initialized.
filter: The tensor generated with bilinear interpolation.
)doc");

REGISTER_KERNEL_BUILDER(
    Name("BilinearInitializer")
      .Device(DEVICE_CPU)
      .TypeConstraint<float>("T"),
    BilinearInitializerOp<float>);

REGISTER_KERNEL_BUILDER(
    Name("BilinearInitializer")
      .Device(DEVICE_CPU)
      .TypeConstraint<double>("T"),
    BilinearInitializerOp<double>);

template class BilinearInitializerOp<float>;
template class BilinearInitializerOp<double>;

} // namespace bilinear_initializer
} // namespace tensorflow
