//Copyright (c) 2016, Alexander G. de G. Matthews and James Hensman. All rights reserved.

//Based on paper "Differentiation of the Cholesky Algorithm" by S. P. Smith (1995). Journal of Computational and Graphical Statistics.

//Also inspired by implementation from GPy by James Hensman and Alan Saul, 2015
//and implementation from AutoGrad by Dougal Maclaurin, David Duvenaud and Matt Johnson, 2015.

#include "tensorflow/core/framework/op.h"
#include "third_party/eigen3/Eigen/Core"

REGISTER_OP("CholeskyGrad").Input("l: T").Input("l_bar: T").Output("a_bar: T").Attr( "T: {float, double}").Doc("Cholesky backpropagation where l is output of Cholesky algorithm and f is gradient of some loss wrt l");

#include "tensorflow/core/framework/op_kernel.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"

using namespace tensorflow;

template <typename T>
class CholeskyGrad : public OpKernel {
    public:
    
    explicit CholeskyGrad(OpKernelConstruction* context) : OpKernel(context) {}

    using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using ConstMatrixMap = Eigen::Map<const Matrix>;
    using MatrixMap = Eigen::Map<Matrix>;
    using ConstRef = Eigen::Ref<const Matrix>;
    using Ref = Eigen::Ref<Matrix>;

    void Compute(OpKernelContext* context) override {

    const Tensor& input_tensor_l = context->input(0); ;
    const Tensor& input_tensor_l_bar = context->input(1); 

    //Check that input tensors represent a matrix.
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(input_tensor_l.shape()), errors::InvalidArgument("In[0] is not a matrix"));
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(input_tensor_l_bar.shape()), errors::InvalidArgument("In[1] is not a matrix"));

    //Check that input tensors are square.
    OP_REQUIRES(context, input_tensor_l.dim_size(0) == input_tensor_l.dim_size(1), errors::InvalidArgument("Input matrix must be square."));
    OP_REQUIRES(context, input_tensor_l_bar.dim_size(0) == input_tensor_l_bar.dim_size(1), errors::InvalidArgument("Input matrix must be square."));

    //Check that input tensors are of same size.
    OP_REQUIRES(context, input_tensor_l.dim_size(0) == input_tensor_l_bar.dim_size(0), errors::InvalidArgument("Input matrices must be of same size."));    

    // Create an output tensor
    Tensor* output_tensor = NULL;
    
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor_l_bar.shape(), &output_tensor));

    if (output_tensor->NumElements() == 0) {
      // the output shape is a 0-element matrix, so there is nothing to do.
      return;
    }
    
    //The next three lines are necessary to get Eigen matrix behaviour.
    const ConstMatrixMap input_matrix_l(input_tensor_l.flat<T>().data(), input_tensor_l.dim_size(0), input_tensor_l.dim_size(1));
    const ConstMatrixMap input_matrix_l_bar(input_tensor_l_bar.flat<T>().data(), input_tensor_l_bar.dim_size(0), input_tensor_l_bar.dim_size(1));
    MatrixMap output_matrix(output_tensor->template flat<T>().data(), input_tensor_l.dim_size(0), input_tensor_l.dim_size(1) );    

    const int N = input_matrix_l.rows();

    output_matrix = input_matrix_l_bar.template triangularView<Eigen::Lower>();

    chol_rev_unblocked( input_matrix_l, output_matrix );
    
    //TODO: what to do if input_matrix_l isn't lower triangular?
    
    //This symmetrizes the result effectively assigning equal contribution to the gradient for the two symmetric halves.
    //output_matrix = (0.5 * ( output_matrix +  output_matrix.transpose() )).eval();
    //output_matrix.template triangularView<Eigen::Upper>() =  output_matrix.template triangularView<Eigen::Lower>().adjoint() ; 

    }
    
    void chol_rev_unblocked(const ConstRef l_block, Ref l_bar_block)
    {
        const int N = l_block.rows();
        
        for ( int k = N-1; k>=0; k--)
        {
            l_bar_block(k,k) -= (l_block.block( k+1,k , N-(k+1), 1 ).adjoint() * l_bar_block.block( k+1,k , N-(k+1), 1 ) )(0,0) / l_block(k,k);
            l_bar_block.block(k,k,N-k,1) /= l_block( k,k) ;
            l_bar_block.block(k,0,1,k) -=l_bar_block.block(k,k,N-k,1).adjoint() * l_block.block(k,0,N-k,k);
            l_bar_block.block(k+1,0,N-(k+1),k) -= l_bar_block.block(k+1, k , N-(k+1), 1 ) * l_block.block( k, 0, 1, k ) ; 
            l_bar_block(k,k) *= 0.5;
        }
    }
    
};

REGISTER_KERNEL_BUILDER(
    Name("CholeskyGrad")
    .Device(DEVICE_CPU)
    .TypeConstraint<float>("T"),
    CholeskyGrad<float>);
    
REGISTER_KERNEL_BUILDER(
    Name("CholeskyGrad")
    .Device(DEVICE_CPU)
    .TypeConstraint<double>("T"),
    CholeskyGrad<double>);
