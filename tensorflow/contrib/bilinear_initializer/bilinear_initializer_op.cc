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

    auto output = output_tensor->tensor<T, 4>();

    for (int i = 0; i < shape[0]; i++) {
      for (int j = 0; j < shape[1]; j++) {
        for (int k = 0; k < shape[2]; k++) {
          for (int l = 0; l < shape[3]; l++) {
            if (k == l) {
              output(i, j, k, l) =
                (1 - fabs(i - center) / factor) *
                (1 - fabs(j - center) / factor);
            } else {
              output(i, j, k, l) = 0.0;
            }
          }
        }
      }
    }
  }

REGISTER_OP("BilinearInitializer")
    .Attr("T: {float, double}")
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

shape: A tensor representing the shape of the filter to be initialized.
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
