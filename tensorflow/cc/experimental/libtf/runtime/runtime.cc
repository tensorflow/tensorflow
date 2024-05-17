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
#include "tensorflow/cc/experimental/libtf/runtime/runtime.h"

#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/c/eager/abstract_context.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/eager/graph_function.h"
#include "tensorflow/c/eager/immediate_execution_context.h"
#include "tensorflow/c/eager/tfe_context_internal.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/c/tf_status_internal.h"
#include "tensorflow/cc/experimental/libexport/load.h"
#include "tensorflow/cc/experimental/libtf/function.h"
#include "tensorflow/cc/experimental/libtf/object.h"
#include "tensorflow/cc/experimental/libtf/value.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/protobuf/saved_object_graph.pb.h"
#include "tensorflow/core/protobuf/struct.pb.h"
#include "tensorflow/core/protobuf/trackable_object_graph.pb.h"

namespace tf {
namespace libtf {
namespace runtime {

using tensorflow::AbstractContext;
using tensorflow::AbstractFunctionPtr;
using tensorflow::DataType;
using tensorflow::FunctionDef;
using tensorflow::PartialTensorShape;
using tensorflow::SavedConcreteFunction;
using tensorflow::SavedObjectGraph;
using tensorflow::Status;
using tensorflow::StructuredValue;
using tensorflow::TensorSpecProto;
using tensorflow::libexport::TFPackage;
using tensorflow::protobuf::RepeatedPtrField;
using tensorflow::tracing::graph::GraphFunction;

TaggedValue MakeCallable(const std::string& fn_name, Function fn,
                         AbstractContext* ctx) {
  auto CallFn = [fn_name, fn, ctx](TaggedValue args_,
                                   TaggedValue kwargs_) -> TaggedValue {
    std::cout << "Calling " << fn_name << std::endl;
    tensorflow::StatusOr<TaggedValue> v = fn.Execute(ctx, args_);
    return v.value();
  };
  return TaggedValue(CallFn);
}

// Import a module from a saved model.
//
// Returns a TaggedValue::Dict. All functions found on the root of the module
// will be attached as callables to this TaggedValue.
//
// `name` specifies the full path to the saved model.
//
// `ctx` should outlive the lifetime of the module.
static tensorflow::StatusOr<TaggedValue> ImportModule(String name,
                                                      AbstractContext* ctx) {
  // Load the serialized model.
  tensorflow::StatusOr<TFPackage> tf_package = TFPackage::Load(name.get());
  if (!tf_package.status().ok()) {
    return tf_package.status();
  }
  TaggedValue module = TaggedValue::Dict();

  // Initialize concrete function traces.
  const RepeatedPtrField<FunctionDef> function_defs =
      tf_package->GetFunctionDefs();
  absl::flat_hash_map<std::string, AbstractFunctionPtr> traces;
  for (auto& fdef : function_defs) {
    AbstractFunctionPtr trace(new GraphFunction(fdef), /*add_ref=*/false);
    traces[fdef.signature().name()] = trace;
  }

  // Setup polymorphic functions and wrap in Callables.
  //
  // For each child of the root, check what type it is.  If it's a
  // SavedFunction, attach that function to the module as a Callable.
  const SavedObjectGraph object_graph = tf_package->GetObjectGraph();
  auto& nodes = object_graph.nodes();
  // Get a map of the concrete functions to their input / output signatures.
  auto& concrete_functions = object_graph.concrete_functions();
  auto& root = nodes.at(0);
  for (auto& child : root.children()) {
    // The child's name describes the name of the edge that connects to the
    // parent object. This name will be the name of the object stored in the
    // generated module.
    auto& child_node = nodes.at(child.node_id());
    auto child_name = child.local_name().c_str();

    if (child_node.kind_case() == tensorflow::SavedObject::kFunction) {
      Function tf_function;
      for (const std::string& fn_name :
           child_node.function().concrete_functions()) {
        // Setup input signature.
        //
        // For now, we have to do a lot of manual digging through these and
        // assume they are tensorspecs. Once TODO(b/190203981) is done, we
        // should be able to pass along the `StructuredValue`s to an API in a
        // much cleaner way.
        //
        // TODO(b/190206621): Implement API for inspecting signatures
        SavedConcreteFunction saved_concrete_function =
            concrete_functions.at(fn_name);
        TaggedValue input_signature = TaggedValue::Tuple();
        const RepeatedPtrField<StructuredValue>& args =
            saved_concrete_function.canonicalized_input_signature()
                .tuple_value()
                .values(0)
                .tuple_value()
                .values();
        for (const StructuredValue& arg : args) {
          PartialTensorShape shape = arg.tensor_spec_value().shape();
          DataType dtype = arg.tensor_spec_value().dtype();
          TaggedValue tensor_spec(shape, dtype);
          input_signature.tuple().emplace_back(tensor_spec);
        }

        // Setup output signature.
        TensorSpecProto output_tensor_spec_proto =
            saved_concrete_function.output_signature().tensor_spec_value();
        PartialTensorShape output_shape = output_tensor_spec_proto.shape();
        DataType output_dtype = output_tensor_spec_proto.dtype();
        TaggedValue output_tensor_spec(output_shape, output_dtype);

        // Register the function trace.
        //
        // This does *not* currently register the function with the runtime.
        // Instead, we're registering JIT at call time. This is likely
        // something that we'll change in TODO(b/190070277).
        auto& trace = traces[fn_name];
        Status status = tf_function.RegisterTrace(
            std::move(trace), input_signature, output_tensor_spec);
      }
      TaggedValue callable = MakeCallable(child_name, tf_function, ctx);
      module.dict()[TaggedValue(child_name)] = callable;
    }
  }
  return module;
}

// Instantiate the Runtime, creating relevant Callables for later use.
Runtime::Runtime(AbstractContext* ctx) {
  TaggedValue ctx_capsule =
      TaggedValue::Capsule(static_cast<void*>(ctx), [](void* p) {
        auto ctx = static_cast<AbstractContext*>(p);
        ctx->Release();
      });
  Set(String("ctx"), Handle(ctx_capsule));
  auto Load = [](Object self, String name) -> Object {
    auto ctx_capsule = self.Get<internal::Capsule>(String("ctx")).value();
    auto ctx = ctx_capsule.cast<AbstractContext*>();
    // TODO(b/191689645): This needs to do error handling better.
    return *Cast<Object>(Handle(*ImportModule(name, ctx)));
  };

  Set(String("Load"), Callable(TFLIB_CALLABLE_ADAPTOR(Load)));
}

tensorflow::StatusOr<Object> Runtime::Load(const String& name) {
  return Get<Callable>(String("Load"))->Call<Object>(*this, name);
}

}  // namespace runtime
}  // namespace libtf
}  // namespace tf
