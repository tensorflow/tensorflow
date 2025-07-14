//===-- Passes.td - pass definition file -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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
  let constructor = "::mlir::emitc::CreateAddReflectionMapPass()";
}
