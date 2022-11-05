
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"

#include "tensorflow/core/kernels/sparse_indices_to_ragged_row_splits_op.h"


using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;


template <typename IndexType>
struct SparseIndicesToRaggedRowSplitsFunctor<CPUDevice, IndexType> {
    Status operator()(
            OpKernelContext* context,
            const CPUDevice& d,
            int num_nonzero, // total number of nonzero values in tensor
            bool validate_ragged_right, // enable validation of input tensor format
            const IndexType* indices_flat_2d, // array of length 2*num_nonzero
            const IndexType* dense_shape,
            int32_t* invalid_flag // single bool, will be set to 1 if input tensor is invalid
            ){
        auto num_rows = dense_shape[0];
        auto num_cols = dense_shape[1];

        Tensor* output;
        TF_RETURN_IF_ERROR(context, context->allocate_output("row_splits", TensorShape({num_rows + 1}), &output));
        IndexType* row_splits = output->flat<IndexType>().data();

        *invalid_flag = 0;

        int prev_row = -1;
        int prev_col = -1;
        int n = 0;
        for (; n < num_nonzero; ++n) { // for each pair of value + indices in structure
            int curr_row = indices_flat_2d[2*n];
            if (validate_ragged_right) {
                if (curr_row != prev_row) { // when row is changing
                    // rows idx must increase monotonically and not exceed dense size
                    // (to ensure indices are in order)
                    if ((curr_row < prev_row) || (curr_row >= num_rows)) {
                        *invalid_flag = 1;
                        return Status::OK();
                    }
                    prev_col = -1;
                }
                // within a row, column values must always be one greater than the previous
                // (to ensure that tensor is ragged-right and indices are in order)
                int curr_col = indices_flat_2d[2*n+1];
                if ((curr_col != prev_col + 1) || (curr_col >= num_cols)) {
                    *invalid_flag = 1;
                    return Status::OK();
                } else {
                    prev_col = curr_col;
                }
            }
            // simply fill in row splits; loop used to fill all if a row is empty
            for (int r = prev_row; r < curr_row; ++r) {
                row_splits[r+1] = n;
            }
            prev_row = curr_row;
        }
        // fill final row split + any trailing empty rows
        for (int r = prev_row; r < num_rows; ++r) {
            row_splits[r+1] = n;
        }

        return Status::OK();
    }
};

template <typename Device, typename IndexType>
class SparseIndicesToRaggedRowSplitsOp : public OpKernel {
    private:
        bool validate_ragged_right;

    public:
        explicit SparseIndicesToRaggedRowSplitsOp(OpKernelConstruction* context) : OpKernel(context) {
            OP_REQUIRES_OK(context, context->GetAttr("validate_ragged_right", &validate_ragged_right));
        }
        void Compute(OpKernelContext* context) override {
            // inputs
            const Tensor& indices = context->input(0);
            const Tensor& dense_shape = context->input(1);

            auto num_nonzero = indices.dim_size(0);

            Tensor* output_invalid = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output("invalid_flag", TensorShape({}), &output_invalid));

            OP_REQUIRES_OK(context,
                    SparseIndicesToRaggedRowSplitsFunctor<Device, IndexType>()(
                        context,
                        context->eigen_device<Device>(),
                        num_nonzero,
                        validate_ragged_right,
                        indices.flat<IndexType>().data(),
                        dense_shape.flat<IndexType>().data(),
                        output_invalid->flat<int32_t>().data()
            ));
      }
};

#ifdef GOOGLE_CUDA
#define REGISTER_GPU(IndexType) \
  REGISTER_KERNEL_BUILDER( \
      Name("SparseIndicesToRaggedRowSplits").Device(DEVICE_GPU).TypeConstraint<IndexType>("IndexType"), \
      SparseIndicesToRaggedRowSplitsOp<GPUDevice, IndexType>);
REGISTER_GPU(int32);
REGISTER_GPU(int64);
#undef REGISTER_GPU

#define REGISTER_CPU(IndexType) \
  REGISTER_KERNEL_BUILDER( \
      Name("SparseIndicesToRaggedRowSplits").Device(DEVICE_CPU).TypeConstraint<IndexType>("IndexType"), \
      SparseIndicesToRaggedRowSplitsOp<CPUDevice, IndexType>);
REGISTER_CPU(int32);
REGISTER_CPU(int64);
#undef REGISTER_CPU

#endif  // GOOGLE_CUDA


