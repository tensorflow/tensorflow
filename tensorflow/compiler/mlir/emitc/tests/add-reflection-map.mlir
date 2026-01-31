// RUN: tf-emitc-opt --add-reflection-map-pipeline  %s | FileCheck %s -dump-input=always

emitc.class @mainClass {
  emitc.field @fieldName0 : !emitc.array<1xf32>  {tf_saved_model.index_path = ["another_feature"]}
  emitc.field @fieldName1 : !emitc.array<1xf32>  {tf_saved_model.index_path = ["some_feature"]}
  emitc.field @fieldName2 : !emitc.array<1xf32>  {tf_saved_model.index_path = ["output_0"]}
  emitc.func @execute() {
  %0 = "emitc.constant"() <{value = 0 : index}> : () -> !emitc.size_t
  %1 = get_field @fieldName0 : !emitc.array<1xf32>
  %2 = get_field @fieldName1 : !emitc.array<1xf32>
  %3 = get_field @fieldName2 : !emitc.array<1xf32>
  %4 = subscript %2[%0] : (!emitc.array<1xf32>, !emitc.size_t) -> !emitc.lvalue<f32>
  %5 = load %4 : <f32>
  %6 = subscript %1[%0] : (!emitc.array<1xf32>, !emitc.size_t) -> !emitc.lvalue<f32>
  %7 = load %6 : <f32>
  %8 = add %5, %7 : (f32, f32) -> f32
  %9 = subscript %3[%0] : (!emitc.array<1xf32>, !emitc.size_t) -> !emitc.lvalue<f32>
  assign %8 : f32 to %9 : <f32>
  return
  }
}

// CHECK: module {
// CHECK: emitc.class @mainClass {
// CHECK: emitc.field @fieldName0 : !emitc.array<1xf32> {tf_saved_model.index_path = ["another_feature"]}
// CHECK: emitc.field @fieldName1 : !emitc.array<1xf32> {tf_saved_model.index_path = ["some_feature"]}
// CHECK: emitc.field @fieldName2 : !emitc.array<1xf32> {tf_saved_model.index_path = ["output_0"]}
// CHECK: emitc.func @getBufferForName(%arg0: !emitc.opaque<"std::string_view">) -> !emitc.opaque<"char"> {
// CHECK: %0 = "emitc.constant"() <{value = #emitc.opaque<"{ { \22another_feature\22, reinterpret_cast<char*>(&another_feature) }, { \22some_feature\22, reinterpret_cast<char*>(&some_feature) }, { \22output_0\22, reinterpret_cast<char*>(&output_0) } }">}> : () -> !emitc.opaque<"const std::map<std::string, char*>">
// CHECK: %1 = call_opaque "find"(%0, %arg0) : (!emitc.opaque<"const std::map<std::string, char*>">, !emitc.opaque<"std::string_view">) -> !emitc.opaque<"std::map<std::string, char*>::const_iterator">
// CHECK: %2 = call_opaque "end"(%0) : (!emitc.opaque<"const std::map<std::string, char*>">) -> !emitc.opaque<"std::map<std::string, char*>::const_iterator">
// CHECK: %3 = call_opaque "operator=="(%1, %2) : (!emitc.opaque<"std::map<std::string, char*>::const_iterator">, !emitc.opaque<"std::map<std::string, char*>::const_iterator">) -> i1
// CHECK: %4 = "emitc.constant"() <{value = #emitc.opaque<"nullptr">}> : () -> !emitc.opaque<"char">
// CHECK: %5 = call_opaque "second"(%1) : (!emitc.opaque<"std::map<std::string, char*>::const_iterator">) -> !emitc.opaque<"char">
// CHECK: %6 = conditional %3, %4, %5 : !emitc.opaque<"char">
// CHECK: return %6 : !emitc.opaque<"char">
// CHECK: }
// CHECK: emitc.func @execute() {
// CHECK: %0 = "emitc.constant"() <{value = 0 : index}> : () -> !emitc.size_t
// CHECK: %1 = get_field @fieldName0 : !emitc.array<1xf32>
// CHECK: %2 = get_field @fieldName1 : !emitc.array<1xf32>
// CHECK: %3 = get_field @fieldName2 : !emitc.array<1xf32>
// CHECK: %4 = subscript %2[%0] : (!emitc.array<1xf32>, !emitc.size_t) -> !emitc.lvalue<f32>
// CHECK: %5 = load %4 : <f32>
// CHECK: %6 = subscript %1[%0] : (!emitc.array<1xf32>, !emitc.size_t) -> !emitc.lvalue<f32>
// CHECK: %7 = load %6 : <f32>
// CHECK: %8 = add %5, %7 : (f32, f32) -> f32
// CHECK: %9 = subscript %3[%0] : (!emitc.array<1xf32>, !emitc.size_t) -> !emitc.lvalue<f32>
// CHECK: assign %8 : f32 to %9 : <f32>
// CHECK: return
// CHECK: }