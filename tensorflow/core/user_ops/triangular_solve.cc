//Copyright (c) 2016, Alexander G. de G. Matthews and James Hensman. All rights reserved.

#include "tensorflow/core/framework/op.h"

//Solve ax = y for x with a being an upper or lower triangular matrix and y being a compatible matrix or vector.

REGISTER_OP("TriangularSolve").Input("a: T").Input("y: T").Output("x: T").Attr( "T: {float, double}").Attr( "Case: {'upper','lower' }" ).Doc("Solves linear system ax=y for x where a is an upper or lower triangular matrix and y is a matrix.");

#include "tensorflow/core/framework/op_kernel.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"

using namespace tensorflow;

template <typename T>
class TriangularSolve : public OpKernel {

    public:
    explicit TriangularSolve(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("Case", &Case));
    }
    
    using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using ConstMatrixMap = Eigen::Map<const Matrix>;
    using MatrixMap = Eigen::Map<Matrix>;
    
    void Compute(OpKernelContext* context) override {
    
    const Tensor & a_tensor = context->input(0); 
    const Tensor & y_tensor = context->input(1); 

    //Check that a_tensor represents a matrix.
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(a_tensor.shape()),
                errors::InvalidArgument("In[0] is not a matrix"));

    //Check that y_tensor represents a matrix.
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(y_tensor.shape()),
                errors::InvalidArgument("In[1] is not a matrix"));    
                
    //Check that a_tensor matrix is square. 
    OP_REQUIRES(context, a_tensor.dim_size(0) == a_tensor.dim_size(1), errors::InvalidArgument("Input matrix must be square."));
                
    //Check that inner y_tensor size is the same as a_tensor size.
    OP_REQUIRES(context, a_tensor.dim_size(1) == y_tensor.dim_size(0), errors::InvalidArgument("Inner indeces must be the same."))
    
    Tensor* output_tensor = NULL;
    
    //Allocate space for output
    OP_REQUIRES_OK(context, context->allocate_output(0, y_tensor.shape(),
                                                     &output_tensor));
    
    if (output_tensor->NumElements() == 0) {
      // the output shape is a 0-element matrix, so there is nothing to do.
      return;
    }
    
    //The next three lines are necessary to get Eigen matrix behaviour.
    const ConstMatrixMap input_a(a_tensor.flat<T>().data(), a_tensor.dim_size(0), a_tensor.dim_size(1));
    const ConstMatrixMap input_y(y_tensor.flat<T>().data(), y_tensor.dim_size(0), y_tensor.dim_size(1));
    MatrixMap output_matrix(output_tensor->flat<T>().data(), y_tensor.dim_size(0), y_tensor.dim_size(1) );
    
    if (Case=="upper")
    {
        output_matrix = input_a.template triangularView<Eigen::Upper>().solve( input_y );    
    }
    else
    {
        output_matrix = input_a.template triangularView<Eigen::Lower>().solve( input_y );           
    }
        
    }
    
    string Case;
};

REGISTER_KERNEL_BUILDER(
    Name("TriangularSolve")
    .Device(DEVICE_CPU)
    .TypeConstraint<float>("T"),
    TriangularSolve<float>);
    
REGISTER_KERNEL_BUILDER(
    Name("TriangularSolve")
    .Device(DEVICE_CPU)
    .TypeConstraint<double>("T"),
    TriangularSolve<double>);
