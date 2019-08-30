/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_KERNELS_DATA_RANGE_DATASET_OP_TEST_H
#define TENSORFLOW_CORE_KERNELS_DATA_RANGE_DATASET_OP_TEST_H

#include "tensorflow/core/kernels/data/dataset_test_params.h"
#include "tensorflow/core/kernels/data/range_dataset_op.h"

namespace tensorflow {
namespace data {

class RangeDatasetParams : public DatasetParams {
 public:
  RangeDatasetParams(int64 start, int64 stop, int64 step,
                     DataTypeVector output_dtypes,
                     std::vector<PartialTensorShape> output_shapes,
                     string node_name)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      std::move(node_name), DatasetParamsType::Range),
        start_(CreateTensor<int64>(TensorShape({}), {start})),
        stop_(CreateTensor<int64>(TensorShape({}), {stop})),
        step_(CreateTensor<int64>(TensorShape({}), {step})) {}

  RangeDatasetParams(int64 start, int64 stop, int64 step)
      : DatasetParams({DT_INT64}, {PartialTensorShape({})}, "range_dataset",
                      DatasetParamsType::Range),
        start_(CreateTensor<int64>(TensorShape({}), {start})),
        stop_(CreateTensor<int64>(TensorShape({}), {stop})),
        step_(CreateTensor<int64>(TensorShape({}), {step})) {}

  Status MakeInputs(gtl::InlinedVector<TensorValue, 4>* inputs) override {
    *inputs = {TensorValue(&start_), TensorValue(&stop_), TensorValue(&step_)};
    return Status::OK();
  }

  Status MakeInputPlaceholder(
      std::vector<string>* input_placeholder) const override {
    *input_placeholder = {RangeDatasetOp::kStart, RangeDatasetOp::kStop,
                          RangeDatasetOp::kStep};
    return Status::OK();
  }

  Status MakeAttributes(AttributeVector* attr_vector) const override {
    *attr_vector = {{RangeDatasetOp::kOutputTypes, output_dtypes_},
                    {RangeDatasetOp::kOutputShapes, output_shapes_}};
    return Status::OK();
  }

 private:
  Tensor start_;
  Tensor stop_;
  Tensor step_;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DATA_RANGE_DATASET_OP_TEST_H
