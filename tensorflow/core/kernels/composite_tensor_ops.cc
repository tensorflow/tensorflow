/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/kernels/composite_tensor_variant.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/protobuf/composite_tensor_variant.pb.h"
#include "tensorflow/core/protobuf/struct.pb.h"

namespace tensorflow {

class CompositeTensorVariantFromComponents : public OpKernel {
 public:
  explicit CompositeTensorVariantFromComponents(OpKernelConstruction* context)
      : OpKernel(context) {
    string type_spec_string;
    OP_REQUIRES_OK(context, context->GetAttr("metadata", &type_spec_string));
    OP_REQUIRES(context, metadata_.ParseFromString(type_spec_string),
                errors::InvalidArgument("Error parsing metadata"));
  }

  void Compute(OpKernelContext* context) override {
    OpInputList components_in;
    OP_REQUIRES_OK(context, context->input_list("components", &components_in));

    Tensor* encoded;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({}), &encoded));

    std::vector<Tensor> components{components_in.begin(), components_in.end()};
    encoded->flat<Variant>()(0) =
        CompositeTensorVariant(metadata_, absl::MakeSpan(components));
  }

 private:
  CompositeTensorVariantMetadata metadata_;
};

class CompositeTensorVariantToComponents : public OpKernel {
 public:
  explicit CompositeTensorVariantToComponents(OpKernelConstruction* context)
      : OpKernel(context) {
    string type_spec_string;
    OP_REQUIRES_OK(context, context->GetAttr("metadata", &type_spec_string));
    OP_REQUIRES(context, metadata_.ParseFromString(type_spec_string),
                errors::InvalidArgument("Error parsing `metadata`"));

    OP_REQUIRES_OK(context,
                   context->GetAttr("Tcomponents", &component_dtypes_));
  }

  void Compute(OpKernelContext* context) override {
    Tensor encoded_t = context->input(0);
    auto* encoded = encoded_t.flat<Variant>()(0).get<CompositeTensorVariant>();

    // Check that the encoded TypeSpec is compatible with the expected TypeSpec.
    // For now, we just check that the class matches.
    //
    // TODO(b/173744905): Update this to do a generic compatibility check. This
    // would require replacing the current design, where Python subclasses of
    // TypeSpec can override is_compatible, with a design where compatibility
    // can be deterministically determined from the metadata.
    auto expected_class = metadata_.type_spec_proto().type_spec_class();
    auto actual_class = encoded->metadata().type_spec_proto().type_spec_class();
    OP_REQUIRES(
        context, expected_class == actual_class,
        errors::InvalidArgument(
            "Expected a ", TypeSpecProto::TypeSpecClass_Name(expected_class),
            " (based on `type_spec`), but `encoded` contains a ",
            TypeSpecProto::TypeSpecClass_Name(actual_class)));

    // Extract the component tensors.
    OpOutputList components;
    OP_REQUIRES_OK(context, context->output_list("components", &components));
    int num_components = encoded->flat_components().size();

    OP_REQUIRES(context, component_dtypes_.size() == num_components,
                errors::InvalidArgument("Encoded value has ", num_components,
                                        " tensor components; expected ",
                                        component_dtypes_.size(),
                                        " components based on type_spec"));

    for (int i = 0; i < component_dtypes_.size(); i++) {
      const Tensor& component = encoded->flat_components()[i];
      OP_REQUIRES(context, component_dtypes_[i] == component.dtype(),
                  errors::InvalidArgument("Tensor component ", i, " had dtype ",
                                          DataType_Name(component.dtype()),
                                          "; expected dtype ",
                                          DataType_Name(component_dtypes_[i])));
      components.set(i, component);
    }
  }

 private:
  CompositeTensorVariantMetadata metadata_;
  std::vector<DataType> component_dtypes_;
};

REGISTER_KERNEL_BUILDER(
    Name("CompositeTensorVariantToComponents").Device(DEVICE_CPU),
    CompositeTensorVariantToComponents);
REGISTER_KERNEL_BUILDER(
    Name("CompositeTensorVariantFromComponents").Device(DEVICE_CPU),
    CompositeTensorVariantFromComponents);

}  // namespace tensorflow
