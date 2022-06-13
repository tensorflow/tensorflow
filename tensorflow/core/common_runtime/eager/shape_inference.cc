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

#include "tensorflow/core/common_runtime/eager/shape_inference.h"

#include <vector>

#include "tensorflow/core/common_runtime/eager/tensor_handle.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {
namespace eager {

Status RunShapeInference(const NodeDef& ndef,
                         const FunctionLibraryDefinition& lib_def,
                         const gtl::InlinedVector<TensorHandle*, 4>& inputs,
                         const gtl::InlinedVector<TensorHandle*, 2>& retvals) {
  const tensorflow::OpRegistrationData* op_reg_data;
  // TODO(b/141209983): Consider adding a shape inference cache.
  // FunctionLibraryDefinition::LookUp delegates to global OpRegistry
  // if op is not a function.
  TF_RETURN_IF_ERROR(lib_def.LookUp(ndef.op(), &op_reg_data));
  if (op_reg_data->shape_inference_fn == nullptr) return OkStatus();

  shape_inference::InferenceContext ic(
      TF_GRAPH_DEF_VERSION, ndef, op_reg_data->op_def,
      std::vector<shape_inference::ShapeHandle>(inputs.size()), {}, {}, {});
  for (size_t i = 0; i < inputs.size(); i++) {
    shape_inference::ShapeHandle shape;
    TF_RETURN_IF_ERROR(inputs[i]->InferenceShape(&ic, &shape));
    ic.SetInput(i, shape);
  }

  TF_RETURN_IF_ERROR(ic.Run(op_reg_data->shape_inference_fn));
  CHECK_EQ(ic.num_outputs(), retvals.size());
  for (int i = 0; i < ic.num_outputs(); i++) {
    shape_inference::ShapeHandle shape_handle = ic.output(i);
    retvals[i]->SetInferenceShape(&ic, shape_handle);
  }
  // TODO(slebedev): populate TensorHandle::handle_dtypes_and_shapes.
  return OkStatus();
}

}  // namespace eager
}  // namespace tensorflow
