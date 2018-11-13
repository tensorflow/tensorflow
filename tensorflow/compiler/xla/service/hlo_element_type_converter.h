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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_ELEMENT_TYPE_CONVERTER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_ELEMENT_TYPE_CONVERTER_H_

#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

// A pass that eliminates certain element types as the input or output of ops by
// inserting Convert ops. This allows a backend to support an element type while
// only actually implementing the Convert op for that element type. This is
// generally not the fastest approach, but it works.
class HloElementTypeConverter : public HloPassInterface {
 public:
  // eliminate_type is the type to eliminate as the input or output of ops,
  // using Convert ops to replace it with replace_with_type.
  HloElementTypeConverter(PrimitiveType eliminate_type,
                          PrimitiveType replace_with_type);

  absl::string_view name() const override { return "element_type_converter"; }

  // Returns the pass on the module and returns whether the module was modified.
  StatusOr<bool> Run(HloModule* module) override;

 private:
  PrimitiveType eliminate_type_;
  PrimitiveType replace_with_type_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_ELEMENT_TYPE_CONVERTER_H_
