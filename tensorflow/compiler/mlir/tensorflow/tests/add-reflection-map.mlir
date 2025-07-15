emitc.class @modelClass {
    emitc.field @fieldName0 : !emitc.array<1xf32>  {tf_saved_model.index_path = ["input_tensor"]}
    emitc.func @execute() {
        %0 = "emitc.constant"() <{value = 0 : index}> : () -> !emitc.size_t
        %1 = get_field @fieldName0 : !emitc.array<1xf32>
        %2 = subscript %1[%0] : (!emitc.array<1xf32>, !emitc.size_t) -> !emitc.lvalue<f32>
        return
    }
}
