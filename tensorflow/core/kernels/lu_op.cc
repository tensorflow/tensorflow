/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// See docs in ../ops/linalg_ops.cc.
#include <vector>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/linalg_ops_common.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/Eigen/LU"


namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

/*    
ConstEigenMatrixMap in_mat(
      tensor_in.flat<T>().data(), params.depth,
      params.tensor_in_cols * params.tensor_in_rows * params.tensor_in_batch);
  EigenMatrixMap out_mat(
      output->flat<T>().data(), params.depth,
      params.out_width * params.out_height * params.tensor_in_batch);
  EigenIndexMatrixMap out_arg_max_mat(
      output_arg_max->flat<int64>().data(), params.depth,
      params.out_width * params.out_height * params.tensor_in_batch);
*/

static const char kErrMsg[] =
    "LU decomposition was not successful. The input might not be valid.";

template <typename Device, typename T>
class LuOp : public OpKernel {
 public:

  /*
  using Matrix =
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using ConstMatrixMap = Eigen::Map<const Matrix>;
  using MatrixMap = Eigen::Map<Matrix>;
  //using ConstMatrixMaps = gtl::InlinedVector<ConstMatrixMap, 4>;
  using MatrixMaps = gtl::InlinedVector<MatrixMap, 4>;
  //using RealScalar = typename Eigen::NumTraits<T>::Real;
 
  using tensorflow::MakeUnique;
  */

  explicit LuOp(OpKernelConstruction* context) : OpKernel(context) {}
  
  void Compute(OpKernelContext* context) override {   
    /*  
    typedef Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
        ConstEigenMatrixMap;
    typedef Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
        EigenMatrixMap;
    typedef Eigen::Map<Eigen::Matrix<int64, Eigen::Dynamic, Eigen::Dynamic>>
        EigenIndexMatrixMap; 
    */
    const Tensor & in = context->input(0);  
    TensorShape mtx_shape = in.shape();  
    //const ConstMatrixMap & input = in;        
    // zhuangh: assume square at this moment        
    auto matrix = in.matrix<T>();
    
    auto & input = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 
                          Eigen::RowMajor>::Map(
                            matrix.data(), matrix.dimension(0), matrix.dimension(1));

    Eigen::PartialPivLU<Eigen::Matrix<T, Eigen::Dynamic, 
                        Eigen::Dynamic, Eigen::RowMajor>> 
                          lu_decomposition(input);
            
    TensorShape perm_shape({});
    perm_shape.AddDim(mtx_shape.dim_size(0));

    // Create the output tensors
    Tensor * output_l = nullptr;
    Tensor * output_u = nullptr;
    Tensor * perm_idx = nullptr;
    
    OP_REQUIRES_OK(context, context->allocate_output(0, mtx_shape, &output_l));
    OP_REQUIRES_OK(context, context->allocate_output(1, mtx_shape, &output_u));
    OP_REQUIRES_OK(context, context->allocate_output(2, perm_shape, &perm_idx));
    
    T * ltensor = output_l->flat<T>().data();
    //auto ltensor = output_l->flat<T>();
    //auto ltensor = output_l->matrix<T>();
    //auto d = context->template eigen_device<Device>();
    //const CPUDevice& d = context->eigen_device<CPUDevice>();
    //ltensor.device(d) = 
    auto &l = lu_decomposition.matrixLU().template triangularView<Eigen::UnitLower>();
    auto lm = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 
                Eigen::Dynamic, Eigen::RowMajor> >(ltensor, l.rows(), l.cols());
    lm = l;
    //ltensor = Eigen::Matrix<T, Eigen::Dynamic, 
    //                        Eigen::Dynamic, Eigen::RowMajor>::Map(l, l.rows(), l.cols());

    T * utensor = output_u->flat<T>().data();
    auto &u = lu_decomposition.matrixLU().template triangularView<Eigen::Upper>();
    auto um = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 
                Eigen::Dynamic, Eigen::RowMajor> >(utensor, u.rows(), u.cols());
    um = u;
    int32 * ptensor = perm_idx->flat<int32>().data();
    auto & p = lu_decomposition.permutationP().indices().array(); 
    
    for(int i = 0; i < p.size(); i++){
      ptensor[i] = p[i];
    }
    
    
    /*
    EigenMatrixMap out_mat(
      output_l->flat<T>().data(), params.depth,
      params.out_width * params.out_height * params.tensor_in_batch);

    T * utensor = output_u->flat<T>().data();
    auto & u = lu_decomposition.matrixLU().template triangularView<Eigen::Upper>();
    utensor = Eigen::Matrix<T, Eigen::Dynamic, 
                            Eigen::Dynamic, Eigen::RowMajor>::Map(u, u.rows(), u.cols());
    int * ptensor = perm_idx->flat<int>().data();
    auto & p = lu_decomposition.permutationP().indices().array();         
    for(int i = 0; i < p.size(); i++){
        ptensor[i] = p[i];
    }    
    */
  }
};

#define REGISTER_KERNEL(T)                                                  \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("Lu").Device(DEVICE_CPU).TypeConstraint<T>("T"),              \
      LuOp<CPUDevice, T>)
 
REGISTER_KERNEL(float);
REGISTER_KERNEL(double);

#undef REGISTER_KERNEL
}  // namespace tensorflow






/*
namespace tensorflow {


static const char kErrMsg[] =
    "LU decomposition was not successful. The input might not be valid.";

template <class Scalar>
class LuOp : public LinearAlgebraOp<Scalar> {
 public:
  INHERIT_LINALG_TYPEDEFS(Scalar);
 explicit LuOp(OpKernelConstruction* context) : Base(context) {}

  TensorShapes GetOutputMatrixShapes(
      const TensorShapes& input_matrix_shapes) const final {
    int64 m = input_matrix_shapes[0].dim_size(0);  
    // only square matrix is supported for now.
    return TensorShapes({TensorShape({m, m}), 
                         TensorShape({m, m}),
                         TensorShape({m})}); // 1, m
  }


  void ComputeMatrix(OpKernelContext* context, const ConstMatrixMaps& inputs,
                     MatrixMaps* outputs) final {
    const ConstMatrixMap& input = inputs[0];
    if (input.rows() == 0) {
      return;
    }

    // Perform the actual LU decomposition.
    Eigen::PartialPivLU<
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        lu_decomposition(input);

    //OP_REQUIRES(context, lu_decomposition.isInvertible() == true,
    //            errors::InvalidArgument(kErrMsg));
    // Output the lower triangular in a dense form.
    outputs->at(0) =
        lu_decomposition.matrixLU().template triangularView<Eigen::UnitLower>();
    outputs->at(1) =
        lu_decomposition.matrixLU().template triangularView<Eigen::Upper>();        
    //outputs->at(2) = lu_decomposition.permutationP();//.indices().data();    
    //Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>  perm(input.rows());
    Eigen::VectorXd perm(input.rows()); 
    outputs->at(2) = perm.cast<Scalar>();
    //outputs->at(2) = perm.cast<int64>();
    
    auto & indices = lu_decomposition.permutationP().indices();    
    for(int i = 0; i < indices.size(); i++){
        perm(i) = indices(i);        
    }
    //using namespace std;
    //cout<<perm<<endl;   
    //lu_decomposition.permutationP();//.indices().data();    
    //outputs->at(2) = lu_decomposition.permutationP().indices().array();        
    //using namespace std;
    //cout<<"permutation matrix"<<endl;
    //cout<<lu_decomposition.permutationP().indices()<<endl;//.cast<int>().array();
    //int n = input.rows();
    //Tensor perm_vec;//(n, 1);
    //outputs->at(2) = perm_vec;    
    //Eigen::ArrayXi perm = lu_decomposition.permutationP().indices().cast<int>().array();
    //for(const auto & it:perm) cout<<it<<" ";
  }
};

REGISTER_LINALG_OP("Lu", (LuOp<float>), float);
REGISTER_LINALG_OP("Lu", (LuOp<double>), double);
REGISTER_LINALG_OP("Lu", (LuOp<complex64>), complex64);
REGISTER_LINALG_OP("Lu", (LuOp<complex128>), complex128);
}  // namespace tensorflow  
*/