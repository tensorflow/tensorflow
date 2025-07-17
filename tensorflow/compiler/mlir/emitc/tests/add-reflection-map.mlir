// RUN: tf-emitc-opt --add-reflection-map-pipeline  %s | FileCheck %s

// CHECK-LABEL: module {
module {
  // CHECK: emitc.class @modelClass {
  emitc.class @modelClass {
    // CHECK: emitc.field @input_tensor : !emitc.array<1xf32>
    emitc.field @input_tensor : !emitc.array<1xf32>
    // CHECK: emitc.func @getBufferForName(%arg0: !emitc.opaque<"std::string_view">) -> !emitc.opaque<"char*"> {
      // CHECK: %0 = "emitc.constant"() <{value = #emitc.opaque<"{}">}> : () -> !emitc.opaque<"std::map<std::string, char*>">
      // CHECK: %1 = "emitc.constant"() <{value = #emitc.opaque<"nullptr">}> : () -> !emitc.opaque<"char*">
      // CHECK: return %1 : !emitc.opaque<"char*">
    // CHECK: emitc.func @execute() {
    emitc.func @execute() {
      // CHECK: %0 = "emitc.constant"() <{value = 0 : index}> : () -> !emitc.size_t
      %0 = "emitc.constant"() <{value = 0 : index}> : () -> !emitc.size_t
      // CHECK: %1 = get_field @input_tensor : !emitc.array<1xf32>
      %1 = get_field @input_tensor : !emitc.array<1xf32>
      // CHECK: %2 = subscript %1[%0] : (!emitc.array<1xf32>, !emitc.size_t) -> !emitc.lvalue<f32>
      %2 = subscript %1[%0] : (!emitc.array<1xf32>, !emitc.size_t) -> !emitc.lvalue<f32>
      // CHECK: return
      return
    }
  }
}


