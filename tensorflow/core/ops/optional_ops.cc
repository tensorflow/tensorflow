/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/full_type.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

// --------------------------------------------------------------------------

REGISTER_OP("OptionalFromValue")
    .Input("components: Toutput_types")
    .Output("optional: variant")
    .Attr("Toutput_types: list(type) >= 1")
    .SetTypeConstructor(full_type::VariadicTensorContainer(TFT_OPTIONAL,
                                                           "Toutput_types"))
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      std::vector<DataType> dtypes;
      TF_RETURN_IF_ERROR(c->GetAttr("Toutput_types", &dtypes));
      c->set_output(0, c->Scalar());
      std::vector<shape_inference::ShapeAndType> shapes_and_types;
      shapes_and_types.reserve(c->num_inputs());
      const FullTypeDef& ret_types = c->ret_types();
      for (int i = 0; i < c->num_inputs(); ++i) {
        // TODO(mdan): output_type(i) == optional is incorrect.
        // "Optional" is the type of the whole container, not of individual
        // elements.
        //
        // Why ret_types.args(0) and not args(i) --
        // For example if Toutput_types is (int32, float32), then
        // ret_types.args[0] (i.e. the 0th output) is
        // Optional[Record[Tensor[int32, s1], Tensor[float32, s2]]]
        // set_output_handle_shapes_and_types tracks the same thing, but in
        // a transposed way:
        // {ShapeAndType(in32, s1, Optional), ShapeAndType(in32, s2, Optional)}
        // That should be corrected in the future (see todo above).
        shapes_and_types.emplace_back(c->input(i), dtypes[i],
                                      ret_types.args(0));
      }
      c->set_output_handle_shapes_and_types(0, shapes_and_types);
      return absl::OkStatus();
    });

REGISTER_OP("OptionalNone")
    .Output("optional: variant")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("OptionalHasValue")
    .Input("optional: variant")
    .Output("has_value: bool")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("OptionalGetValue")
    .Input("optional: variant")
    .Output("components: output_types")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::DatasetIteratorShape);

}  // namespace tensorflow
