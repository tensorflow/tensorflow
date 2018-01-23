/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include <array>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/threadpool.h"

#include "tensorflow/contrib/nearest_neighbor/kernels/hyperplane_lsh_probes.h"

namespace tensorflow {

using errors::Internal;
using errors::InvalidArgument;

using nearest_neighbor::HyperplaneMultiprobe;

// This class wraps the multiprobe LSH code in hyperplane_lsh_probes in a
// TensorFlow op implementation.
template <typename CoordinateType>
class HyperplaneLSHProbesOp : public OpKernel {
 public:
  using Matrix = Eigen::Matrix<CoordinateType, Eigen::Dynamic, Eigen::Dynamic,
                               Eigen::RowMajor>;
  using ConstMatrixMap = Eigen::Map<const Matrix>;
  using MatrixMap = Eigen::Map<Matrix>;

  explicit HyperplaneLSHProbesOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Get the input tensors and check their shapes.
    const Tensor& products_tensor = context->input(0);
    OP_REQUIRES(context, products_tensor.dims() == 2,
                InvalidArgument("Need a two-dimensional products tensor, got ",
                                products_tensor.dims(), " dimensions."));

    const Tensor& num_tables_tensor = context->input(1);
    OP_REQUIRES(context, num_tables_tensor.dims() == 0,
                InvalidArgument("Need a scalar num_tables tensor, got ",
                                num_tables_tensor.dims(), " dimensions."));
    int num_tables = num_tables_tensor.scalar<int32>()();
    OP_REQUIRES(context, num_tables >= 1,
                InvalidArgument("num_tables must be at least 1 but got ",
                                num_tables, "."));
    OP_REQUIRES(context, num_tables <= 1000,
                InvalidArgument("Need num_tables <= 1000, got ", num_tables,
                                ". This is mostly to protect against incorrect "
                                "use of this Op. If you really need more tables"
                                ", change the code."));

    const Tensor& num_hyperplanes_per_table_tensor = context->input(2);
    OP_REQUIRES(context, num_hyperplanes_per_table_tensor.dims() == 0,
                InvalidArgument("Need a scalar num_hyperplanes_per_table "
                                "tensor, got ",
                                num_hyperplanes_per_table_tensor.dims(),
                                " dimensions."));
    int num_hyperplanes_per_table =
        num_hyperplanes_per_table_tensor.scalar<int32>()();
    OP_REQUIRES(context, num_hyperplanes_per_table >= 1,
                InvalidArgument("num_hyperplanes_per_table must be at least 1 "
                                "but got ",
                                num_hyperplanes_per_table, "."));
    OP_REQUIRES(context, num_hyperplanes_per_table <= 30,
                InvalidArgument("Need num_hyperplanes_per_table <= 30, got ",
                                num_hyperplanes_per_table, ". "
                                "If you need more hyperplanes, change this Op"
                                " to work for larger integer types (int64)."));

    const Tensor& num_probes_tensor = context->input(3);
    OP_REQUIRES(context, num_probes_tensor.dims() == 0,
                InvalidArgument("Need a scalar num_probes tensor, got ",
                                num_probes_tensor.dims(), " dimensions."));
    int num_probes = num_probes_tensor.scalar<int32>()();
    OP_REQUIRES(context, num_probes >= 1,
                InvalidArgument("num_probes must be at least 1."));

    int expected_num_hyperplanes = num_tables * num_hyperplanes_per_table;
    OP_REQUIRES(
        context, products_tensor.dim_size(1) == expected_num_hyperplanes,
        InvalidArgument("Expected number of hyperplanes is ",
                        expected_num_hyperplanes, " but received ",
                        products_tensor.dim_size(1), " inner products per "
                        "point."));

    auto products_eigen_tensor = products_tensor.matrix<CoordinateType>();
    ConstMatrixMap products_matrix(products_eigen_tensor.data(),
                                   products_tensor.dim_size(0),
                                   products_tensor.dim_size(1));

    int batch_size = products_tensor.dim_size(0);

    Tensor* probes_tensor = nullptr;
    Tensor* tables_tensor = nullptr;
    TensorShape output_shape({batch_size, num_probes});
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, output_shape, &probes_tensor));
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, output_shape, &tables_tensor));
    auto probes_eigen_tensor = probes_tensor->matrix<int32>();
    auto tables_eigen_tensor = tables_tensor->matrix<int32>();

    // Constants (cycles per hyperplane and table) were measured on
    // lschmidt's workstation.
    int64 cost_per_unit = 21 * num_hyperplanes_per_table * num_tables;
    if (num_probes > num_tables) {
      cost_per_unit += 110 * num_hyperplanes_per_table
          * (num_probes - num_tables);
    }
    context->device()->tensorflow_cpu_worker_threads()->workers->ParallelFor(
        batch_size,
        cost_per_unit,
        [&](int64 start, int64 end) {
          HyperplaneMultiprobe<CoordinateType, int32> multiprobe(
              num_hyperplanes_per_table, num_tables);

          for (int point_index = start; point_index < end; ++point_index) {
            multiprobe.SetupProbing(products_matrix.row(point_index),
                                    num_probes);
            for (int ii = 0; ii < num_probes; ++ii) {
              int32 cur_probe;
              int_fast32_t cur_table;
              OP_REQUIRES(context,
                          multiprobe.GetNextProbe(&cur_probe, &cur_table),
                          Internal("Failed to get probe number ", ii,
                                   " for point number ", point_index, "."));
              probes_eigen_tensor(point_index, ii) = cur_probe;
              tables_eigen_tensor(point_index, ii) = cur_table;
            }
          }
        });
  }
};

REGISTER_KERNEL_BUILDER(Name("HyperplaneLSHProbes")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("CoordinateType"),
                        HyperplaneLSHProbesOp<float>);

REGISTER_KERNEL_BUILDER(Name("HyperplaneLSHProbes")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<double>("CoordinateType"),
                        HyperplaneLSHProbesOp<double>);

}  // namespace tensorflow
