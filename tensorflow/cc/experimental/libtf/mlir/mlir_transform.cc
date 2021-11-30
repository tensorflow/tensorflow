/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/cc/experimental/libtf/mlir/mlir_transform.h"

#include <string>
#include <utility>

#include "tensorflow/cc/experimental/libtf/object.h"
#include "tensorflow/cc/experimental/libtf/value.h"
#include "tensorflow/cc/saved_model/bundle_v2.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/import_model.h"

namespace tf {
namespace libtf {

// TODO(b/190837282): All return None's become errors.
Handle LoadModule(Object self, String saved_model) {
  // Parse arguments.
  // Load SavedModel into memory.
  tensorflow::SavedModelV2Bundle bundle;
  tensorflow::Status status =
      tensorflow::SavedModelV2Bundle::Load(saved_model.get(), &bundle);
  if (!status.ok()) {
    return None();
  }
  // Fetch MLIR context
  auto* context = self.Get<internal::Capsule>(String("_context"))
                      ->cast<mlir::MLIRContext*>();

  // Load the saved model into MLIR TF dialect.
  absl::Span<std::string> exported_names(nullptr, 0);
  auto module_or =
      tensorflow::ConvertSavedModelToMlir(&bundle, context, exported_names);
  if (!module_or.status().ok()) {
    return None();
  }

  // Make a module to wrap MLIR module and allow getting strings and running
  // transforms.
  // auto obj = TaggedValue::Dict();
  Object obj;
  obj.Set(String("_module"),
          Handle(impl::TaggedValue::Capsule(new mlir::OwningModuleRef(
              std::move(module_or).ConsumeValueOrDie()))));

  auto get_string = [](Object self) {
    auto ref = self.Get<internal::Capsule>(String("_module"))
                   ->cast<mlir::OwningModuleRef*>();
    return String(tensorflow::MlirModuleToString(ref->get(), false).c_str());
  };
  obj.Set(String("ToString"), Callable(TFLIB_CALLABLE_ADAPTOR(get_string)));

  return obj;
}

None SaveModule(Object self, Object module, String directory) {
  // TODO(b/190835292): Implement save.
  return None();
}

None Transform(Object self, Object module, List passes) {
  // TODO(b/190835292): Implement save.
  return None();
}

Object MLIR() {
  Object obj;
  obj.Set(String("LoadSavedModel"),
          Callable(TFLIB_CALLABLE_ADAPTOR(LoadModule)));
  obj.Set(String("SaveSavedModel"),
          Callable(TFLIB_CALLABLE_ADAPTOR(SaveModule)));
  obj.Set(String("_context"),
          Handle(impl::TaggedValue::Capsule(new mlir::MLIRContext())));
  return obj;
}

}  // namespace libtf
}  // namespace tf
