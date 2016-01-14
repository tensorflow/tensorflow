//Copyright (c) 2016, Alexander G. de G. Matthews and James Hensman. All rights reserved.

#include "tensorflow/core/framework/op.h"

REGISTER_OP("Triangle").Input("a: T").Output("x: T").Attr( "T: {float, double}").Attr( "Case: {'upper','lower' }" ).Doc("Gives upper or lower triangular half of a square matrix.");

#include "tensorflow/core/framework/op_kernel.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"

using namespace tensorflow;

template <typename T>
class Triangle : public OpKernel {

    public:
    explicit Triangle(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("Case", &Case));
    }
    
    using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using ConstMatrixMap = Eigen::Map<const Matrix>;
    using MatrixMap = Eigen::Map<Matrix>;
    
    void Compute(OpKernelContext* context) override {
    
    const Tensor & a_tensor = context->input(0); 

    //Check that a_tensor represents a matrix.
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(a_tensor.shape()),
                errors::InvalidArgument("In[0] is not a matrix"));

    //Check that a_tensor matrix is square. 
    OP_REQUIRES(context, a_tensor.dim_size(0) == a_tensor.dim_size(1), errors::InvalidArgument("Input matrix must be square."));
                
    Tensor* output_tensor = NULL;
    
    //Allocate space for output
    OP_REQUIRES_OK(context, context->allocate_output(0, a_tensor.shape(),
                                                     &output_tensor));
    
    if (output_tensor->NumElements() == 0) {
      // the output shape is a 0-element matrix, so there is nothing to do.
      return;
    }
    
    //The next three lines are necessary to get Eigen matrix behaviour.
    const ConstMatrixMap input_a(a_tensor.flat<T>().data(), a_tensor.dim_size(0), a_tensor.dim_size(1));
    MatrixMap output_matrix(output_tensor->flat<T>().data(), a_tensor.dim_size(0), a_tensor.dim_size(1) );
    
    if (Case=="upper")
    {
        output_matrix = input_a.template triangularView<Eigen::Upper>();    
    }
    else
    {
        output_matrix = input_a.template triangularView<Eigen::Lower>();           
    }
        
    }
    
    string Case;
};

REGISTER_KERNEL_BUILDER(
    Name("Triangle")
    .Device(DEVICE_CPU)
    .TypeConstraint<float>("T"),
    Triangle<float>);
    
REGISTER_KERNEL_BUILDER(
    Name("Triangle")
    .Device(DEVICE_CPU)
    .TypeConstraint<double>("T"),
    Triangle<double>);
