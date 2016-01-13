#include "tensorflow/core/framework/op.h"

REGISTER_OP("GetDiag").Input("l: T").Output("g: T").Attr( "T: {float, double}").Doc("Get diagonal of a square matrix.");

#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;
template <typename T>
class GetDiag : public OpKernel {
    public:
    
    explicit GetDiag(OpKernelConstruction* context) : OpKernel(context) {}
    
    void Compute(OpKernelContext* context) override {

    const Tensor& input_tensor = context->input(0); 
    
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(input_tensor.shape()), errors::InvalidArgument("In[0] is not a matrix"));
    OP_REQUIRES(context, input_tensor.dim_size(0) == input_tensor.dim_size(1), errors::InvalidArgument("Input matrix must be square."));        

    const int N = input_tensor.dim_size(0);

    Tensor* output_tensor = NULL;
    
    const TensorShape output_shape = TensorShape({ N }); 
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));    
    
    // Create an output tensor
    auto output = output_tensor->template flat<T>();
    auto input = input_tensor.template matrix<T>();
       
    //Could presumably be vectorized if this became a bottleneck.
    for (int i = 0; i < N; i++) {
      output(i) = input(i,i);
    }
    
    }
};

REGISTER_KERNEL_BUILDER(
    Name("GetDiag")
    .Device(DEVICE_CPU)
    .TypeConstraint<float>("T"),
    GetDiag<float>);
    
REGISTER_KERNEL_BUILDER(
    Name("GetDiag")
    .Device(DEVICE_CPU)
    .TypeConstraint<double>("T"),
    GetDiag<double>);
