// See docs in ../ops/linalg_ops.cc.
#include <cmath>

#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/linalg_ops_common.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/public/tensor_shape.h"
#include "third_party/eigen3/Eigen/LU"

namespace tensorflow {

template <class Scalar, bool SupportsBatchOperationT>
class MatrixInverseOp
    : public LinearAlgebraOp<Scalar, SupportsBatchOperationT> {
 public:
  explicit MatrixInverseOp(OpKernelConstruction* context)
      : LinearAlgebraOp<Scalar, SupportsBatchOperationT>(context) {}
  ~MatrixInverseOp() override {}

  TensorShape GetOutputMatrixShape(
      const TensorShape& input_matrix_shape) override {
    return input_matrix_shape;
  }

  int64 GetCostPerUnit(const TensorShape& input_matrix_shape) override {
    const int64 rows = input_matrix_shape.dim_size(0);
    if (rows > (1LL << 20)) {
      // A big number to cap the cost in case overflow.
      return kint32max;
    } else {
      return rows * rows * rows;
    }
  }

  using typename LinearAlgebraOp<Scalar, SupportsBatchOperationT>::MatrixMap;
  using
      typename LinearAlgebraOp<Scalar, SupportsBatchOperationT>::ConstMatrixMap;

  void ComputeMatrix(OpKernelContext* context, const ConstMatrixMap& input,
                     MatrixMap* output) override {
    OP_REQUIRES(context, input.rows() == input.cols(),
                errors::InvalidArgument("Input matrix must be square."));
    if (input.rows() == 0) {
      // By definition, an empty matrix's inverse is an emptry matrix.
      return;
    }
    Eigen::FullPivLU<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic,
                                   Eigen::RowMajor>> lu_decomposition(input);
    OP_REQUIRES(context, lu_decomposition.isInvertible(),
                errors::InvalidArgument("Input is not invertible."));
    *output = lu_decomposition.inverse();
  }
};

REGISTER_LINALG_OP("MatrixInverse", (MatrixInverseOp<float, false>), float);
REGISTER_LINALG_OP("MatrixInverse", (MatrixInverseOp<double, false>), double);
REGISTER_LINALG_OP("BatchMatrixInverse", (MatrixInverseOp<float, true>), float);
REGISTER_LINALG_OP("BatchMatrixInverse", (MatrixInverseOp<double, true>),
                   double);

}  // namespace tensorflow
