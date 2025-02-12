// RUN: tfg-opt-no-passes %s | tfg-opt-no-passes | FileCheck %s

// CHECK: module attributes {tfg.type = !tf_type.qint8
module attributes {tfg.type = !tf_type.qint8} {}
// CHECK: module attributes {tfg.type = !tf_type.qint16
module attributes {tfg.type = !tf_type.qint16} {}
// CHECK: module attributes {tfg.type = !tf_type.qint32
module attributes {tfg.type = !tf_type.qint32} {}
// CHECK: module attributes {tfg.type = !tf_type.quint8
module attributes {tfg.type = !tf_type.quint8} {}
// CHECK: module attributes {tfg.type = !tf_type.quint16
module attributes {tfg.type = !tf_type.quint16} {}
// CHECK: module attributes {tfg.type = !tf_type.string
module attributes {tfg.type = !tf_type.string} {}
// CHECK: module attributes {tfg.type = !tf_type.resource
module attributes {tfg.type = !tf_type.resource} {}
// CHECK: module attributes {tfg.type = !tf_type.variant
module attributes {tfg.type = !tf_type.variant} {}
// CHECK: module attributes {tfg.type = !tf_type.f32ref
module attributes {tfg.type = !tf_type.f32ref} {}
// CHECK: module attributes {tfg.type = !tf_type.uint4ref
module attributes {tfg.type = !tf_type.uint4ref} {}
// CHECK: module attributes {tfg.type = !tf_type.int4ref
module attributes {tfg.type = !tf_type.int4ref} {}
// CHECK: module attributes {tfg.type = !tf_type.f64ref
module attributes {tfg.type = !tf_type.f64ref} {}
// CHECK: module attributes {tfg.type = !tf_type.uint8ref
module attributes {tfg.type = !tf_type.uint8ref} {}
// CHECK: module attributes {tfg.type = !tf_type.int8ref
module attributes {tfg.type = !tf_type.int8ref} {}
// CHECK: module attributes {tfg.type = !tf_type.uint16ref
module attributes {tfg.type = !tf_type.uint16ref} {}
// CHECK: module attributes {tfg.type = !tf_type.int16ref
module attributes {tfg.type = !tf_type.int16ref} {}
// CHECK: module attributes {tfg.type = !tf_type.uint32ref
module attributes {tfg.type = !tf_type.uint32ref} {}
// CHECK: module attributes {tfg.type = !tf_type.int32ref
module attributes {tfg.type = !tf_type.int32ref} {}
// CHECK: module attributes {tfg.type = !tf_type.uint64ref
module attributes {tfg.type = !tf_type.uint64ref} {}
// CHECK: module attributes {tfg.type = !tf_type.int64ref
module attributes {tfg.type = !tf_type.int64ref} {}
// CHECK: module attributes {tfg.type = !tf_type.stringref
module attributes {tfg.type = !tf_type.stringref} {}
// CHECK: module attributes {tfg.type = !tf_type.boolref
module attributes {tfg.type = !tf_type.boolref} {}
// CHECK: module attributes {tfg.type = !tf_type.quint8ref
module attributes {tfg.type = !tf_type.quint8ref} {}
// CHECK: module attributes {tfg.type = !tf_type.qint8ref
module attributes {tfg.type = !tf_type.qint8ref} {}
// CHECK: module attributes {tfg.type = !tf_type.quint16ref
module attributes {tfg.type = !tf_type.quint16ref} {}
// CHECK: module attributes {tfg.type = !tf_type.qint16ref
module attributes {tfg.type = !tf_type.qint16ref} {}
// CHECK: module attributes {tfg.type = !tf_type.qint32ref
module attributes {tfg.type = !tf_type.qint32ref} {}
// CHECK: module attributes {tfg.type = !tf_type.bfloat16ref
module attributes {tfg.type = !tf_type.bfloat16ref} {}
// CHECK: module attributes {tfg.type = !tf_type.complex64ref
module attributes {tfg.type = !tf_type.complex64ref} {}
// CHECK: module attributes {tfg.type = !tf_type.complex128ref
module attributes {tfg.type = !tf_type.complex128ref} {}
// CHECK: module attributes {tfg.type = !tf_type.halfref
module attributes {tfg.type = !tf_type.halfref} {}
// CHECK: module attributes {tfg.type = !tf_type.float8e4m3fnref
module attributes {tfg.type = !tf_type.float8e4m3fnref} {}
// CHECK: module attributes {tfg.type = !tf_type.float8e5m2ref
module attributes {tfg.type = !tf_type.float8e5m2ref} {}
// CHECK: module attributes {tfg.type = !tf_type.float8e4m3fnuzref
module attributes {tfg.type = !tf_type.float8e4m3fnuzref} {}
// CHECK: module attributes {tfg.type = !tf_type.float8e4m3b11fnuzref
module attributes {tfg.type = !tf_type.float8e4m3b11fnuzref} {}
// CHECK: module attributes {tfg.type = !tf_type.float8e5m2fnuzref
module attributes {tfg.type = !tf_type.float8e5m2fnuzref} {}
// CHECK: module attributes {tfg.type = !tf_type.control
module attributes {tfg.type = !tf_type.control} {}
// CHECK: module attributes {tfg.type = !tf_type.tensor
module attributes {tfg.type = !tf_type.tensor} {}

// CHECK: module attributes {tfg.type = #tf_type.full_type<array<for_each<product, tensor<var "T">, var "T">>>
module attributes {tfg.type = #tf_type.full_type<array<for_each<product, tensor<var "T">, var "T">>>} {}
// CHECK: module attributes {tfg.type = #tf_type.func<@symbol, {attr = "v"}>}
module attributes {tfg.type = #tf_type.func<@symbol, {attr = "v"}>} {}
// CHECK: module attributes {tfg.type = #tf_type.placeholder<"FOO">}
module attributes {tfg.type = #tf_type.placeholder<"FOO">} {}
// CHECK: module attributes {tfg.type = #tf_type.shape<10x20x30>}
module attributes {tfg.type = #tf_type.shape<10x20x30>} {}
// CHECK: module attributes {tfg.type = #tf_type.shape<10x?x30>}
module attributes {tfg.type = #tf_type.shape<10x?x30>} {}
// CHECK: module attributes {tfg.type = #tf_type.shape<*>}
module attributes {tfg.type = #tf_type.shape<*>} {}
// CHECK: module attributes {tfg.type = #tf_type.version<producer = 42, min_consumer = 33>}
module attributes {tfg.type = #tf_type.version<producer = 42, min_consumer = 33>} {}
// CHECK: module attributes {tfg.type = #tf_type.version<producer = 1, min_consumer = 1, bad_consumers = [1, 2, 5, 12]>}
module attributes {tfg.type = #tf_type.version<producer = 1, min_consumer = 1, bad_consumers = [1, 2, 5, 12]>} {}

// CHECK: !tf_type.variant
module attributes {tfg.type = !tf_type.variant} {}

// CHECK: !tf_type.variant<tensor<?xf32>>
module attributes {tfg.type = !tf_type.variant<tensor<?xf32>>} {}

// CHECK: !tf_type.variant<tensor<3xf32>, tensor<2xi32>>
module attributes {tfg.type = !tf_type.variant<tensor<3xf32>, tensor<2xi32>>} {}

// CHECK: tensor<*x!tf_type.variant<tensor<?xf32>>>
module attributes {tfg.type = tensor<*x!tf_type.variant<tensor<?xf32>>>} {}

// CHECK: tensor<!tf_type.variant<tensor<?x!tf_type.variant<tensor<?xf32>>>>>
module attributes {tfg.type = tensor<!tf_type.variant<tensor<?x!tf_type.variant<tensor<?xf32>>>>>} {}

// CHECK: !tf_type.variantref
module attributes {tfg.type = !tf_type.variantref} {}
