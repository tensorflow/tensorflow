//Copyright (c) 2016, Alexander G. de G. Matthews and James Hensman. All rights reserved.

//Based on paper "Differentiation of the Cholesky Algorithm" by S. P. Smith (1995). Journal of Computational and Graphical Statistics.

//Also inspired by implementation from GPy by James Hensman and Alan Saul, 2015
//and implementation from AutoGrad by Dougal Maclaurin, David Duvenaud and Matt Johnson, 2015.

#include "tensorflow/core/framework/op.h"

REGISTER_OP("CholeskyGrad").Input("l: T").Input("l_bar: T").Output("a_bar: T").Attr( "T: {float, double}").Doc("Cholesky backpropagation where l is output of Cholesky algorithm and f is gradient of some loss wrt l");

/*
CHOL_REV_UNBLOCKED push derivatives back through a Cholesky decomposition

     A_bar = chol_rev_unblocked(L, L_bar)

 Back-propagate derivatives of some function f, through the Cholesky
 decomposition:
 A_bar = tril(df/dA), when L = chol(A, 'lower') and L_bar = df/dL.

 Inputs:
          L NxN lower-triangular matrix, resulting from chol(A, 'lower'),
                where A is a symmetric +ve definite matrix
      L_bar NxN df/dL for some scalar function f

 Outputs:
      A_bar NxN tril(df/dA)
*/

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

    /*
    for J = N:-1:1
        L_bar(J,J) = L_bar(J,J) - L(J+1:N,J)'*L_bar(J+1:N,J) / L(J,J);
        L_bar(J:N,J) = L_bar(J:N,J) / L(J,J);
        L_bar(J,1:J-1) = L_bar(J,1:J-1) - L_bar(J:N,J)'*L(J:N,1:J-1);
        L_bar(J+1:N,1:J-1) = L_bar(J+1:N,1:J-1) - L_bar(J+1:N,J)*L(J,1:J-1);
        L_bar(J,J) = 0.5 * L_bar(J,J); % can take out of loop if like.
    end
    */
    
    //TODO: what to do if input_matrix_l isn't lower triangular?
    
    for ( int k = N-1; k>=0; k--)
    {
        output_matrix(k,k) -= (input_matrix_l.block( k+1,k , N-(k+1), 1 ).adjoint() * output_matrix.block( k+1,k , N-(k+1), 1 ) )(0,0) / input_matrix_l(k,k);
        output_matrix.block(k,k,N-k,1) /= input_matrix_l( k,k) ;
        output_matrix.block(k,0,1,k) -=output_matrix.block(k,k,N-k,1).adjoint() * input_matrix_l.block(k,0,N-k,k);
        output_matrix.block(k+1,0,N-(k+1),k) -= output_matrix.block(k+1, k , N-(k+1), 1 ) * input_matrix_l.block( k, 0, 1, k ) ; 
        output_matrix(k,k) *= 0.5;
    }

    //This symmetrizes the result effectively assigning equal contribution to the gradient for the two symmetric halves.
    //output_matrix = (0.5 * ( output_matrix +  output_matrix.transpose() )).eval();
    //output_matrix.template triangularView<Eigen::Upper>() =  output_matrix.template triangularView<Eigen::Lower>().adjoint() ; 

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
