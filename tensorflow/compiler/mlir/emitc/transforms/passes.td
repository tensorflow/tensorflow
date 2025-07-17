/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

include "mlir/Pass/PassBase.td"

// EmitC dialect Passes

def AddReflectionMapPass : Pass<"add-reflection-map", "mlir::emitc::ClassOp"> {
  let summary =
      "Add a reflection map function to EmitC classes for runtime field lookup";
  let description =
      [{
        This pass adds a `getBufferForName` function to EmitC classes that enables 
        runtime lookup of field buffers by their string names. 
        This enables runtime introspection and dynamic access to class fields by name,
        which is useful for interfacing with external systems that need to access 
        tensors/buffers by their semantic names.

        Example transformation:
        ```mlir
        emitc.class @MyClass {
          emitc.field @fieldName0 : !emitc.array<1xf32> {tf_saved_model.index_path = ["another_feature"]}
          emitc.field @fieldName1 : !emitc.array<1xf32> {tf_saved_model.index_path = ["some_feature"]}
          emitc.func @execute() { ... }
        }
        ```

        Becomes:
        ```mlir
        emitc.class @MyClass {
          emitc.field @fieldName0 : !emitc.array<1xf32> {tf_saved_model.index_path = ["another_feature"]}
          emitc.field @fieldName1 : !emitc.array<1xf32> {tf_saved_model.index_path = ["some_feature"]}
          emitc.func @getBufferForName(%name : !emitc.opaque<"std::string_view">) -> !emitc.opaque<"char*"> {
            %map = "emitc.constant"(){value = #emitc.opaque<"{"another_feature", reinterpret_cast<char*>(&another_feature)}, {"some_feature", reinterpret_cast<char*>(&some_feature)}">} : () -> !emitc.opaque<"std::map<std::string, char*>">
            return %null : !emitc.opaque<"char*">
          }
          emitc.func @execute() { ... }
        }
        ```
    }];
  let constructor = "mlir::emitc::CreateAddReflectionMapPass()";
  let dependentDialects = ["mlir::emitc::EmitCDialect"];
}
