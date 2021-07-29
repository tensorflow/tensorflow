// RUN: tfg-opt-no-passes %s | tfg-opt-no-passes | FileCheck %s

// CHECK: func private @qint8() -> !tf_type.qint8
func private @qint8() -> !tf_type.qint8
// CHECK: func private @qint16() -> !tf_type.qint16
func private @qint16() -> !tf_type.qint16
// CHECK: func private @qint32() -> !tf_type.qint32
func private @qint32() -> !tf_type.qint32
// CHECK: func private @quint8() -> !tf_type.quint8
func private @quint8() -> !tf_type.quint8
// CHECK: func private @quint16() -> !tf_type.quint16
func private @quint16() -> !tf_type.quint16
// CHECK: func private @string() -> !tf_type.string
func private @string() -> !tf_type.string
// CHECK: func private @resource() -> !tf_type.resource
func private @resource() -> !tf_type.resource
// CHECK: func private @variant() -> !tf_type.variant
func private @variant() -> !tf_type.variant
// CHECK: func private @f32ref() -> !tf_type.f32ref
func private @f32ref() -> !tf_type.f32ref
// CHECK: func private @f64ref() -> !tf_type.f64ref
func private @f64ref() -> !tf_type.f64ref
// CHECK: func private @uint8ref() -> !tf_type.uint8ref
func private @uint8ref() -> !tf_type.uint8ref
// CHECK: func private @int8ref() -> !tf_type.int8ref
func private @int8ref() -> !tf_type.int8ref
// CHECK: func private @uint16ref() -> !tf_type.uint16ref
func private @uint16ref() -> !tf_type.uint16ref
// CHECK: func private @int16ref() -> !tf_type.int16ref
func private @int16ref() -> !tf_type.int16ref
// CHECK: func private @uint32ref() -> !tf_type.uint32ref
func private @uint32ref() -> !tf_type.uint32ref
// CHECK: func private @int32ref() -> !tf_type.int32ref
func private @int32ref() -> !tf_type.int32ref
// CHECK: func private @uint64ref() -> !tf_type.uint64ref
func private @uint64ref() -> !tf_type.uint64ref
// CHECK: func private @int64ref() -> !tf_type.int64ref
func private @int64ref() -> !tf_type.int64ref
// CHECK: func private @stringref() -> !tf_type.stringref
func private @stringref() -> !tf_type.stringref
// CHECK: func private @boolref() -> !tf_type.boolref
func private @boolref() -> !tf_type.boolref
// CHECK: func private @quint8ref() -> !tf_type.quint8ref
func private @quint8ref() -> !tf_type.quint8ref
// CHECK: func private @qint8ref() -> !tf_type.qint8ref
func private @qint8ref() -> !tf_type.qint8ref
// CHECK: func private @quint16ref() -> !tf_type.quint16ref
func private @quint16ref() -> !tf_type.quint16ref
// CHECK: func private @qint16ref() -> !tf_type.qint16ref
func private @qint16ref() -> !tf_type.qint16ref
// CHECK: func private @qint32ref() -> !tf_type.qint32ref
func private @qint32ref() -> !tf_type.qint32ref
// CHECK: func private @bfloat16ref() -> !tf_type.bfloat16ref
func private @bfloat16ref() -> !tf_type.bfloat16ref
// CHECK: func private @complex64ref() -> !tf_type.complex64ref
func private @complex64ref() -> !tf_type.complex64ref
// CHECK: func private @complex128ref() -> !tf_type.complex128ref
func private @complex128ref() -> !tf_type.complex128ref
// CHECK: func private @halfref() -> !tf_type.halfref
func private @halfref() -> !tf_type.halfref
// CHECK: func private @control() -> !tf_type.control
func private @control() -> !tf_type.control
// CHECK: func private @tensor() -> !tf_type.tensor
func private @tensor() -> !tf_type.tensor

// CHECK: func private @func_attr() attributes {f = #tf_type.func<@symbol, {attr = "v"}>}
func private @func_attr() attributes {f = #tf_type.func<@symbol, {attr = "v"}>}
// CHECK: func private @placeholder_attr() attributes {p = #tf_type.placeholder<"FOO">}
func private @placeholder_attr() attributes {p = #tf_type.placeholder<"FOO">}
// CHECK: func private @shape_attr() attributes {p = #tf_type.shape<10x20x30>}
func private @shape_attr() attributes {p = #tf_type.shape<10x20x30>}
// CHECK: func private @shape_attr_dyn() attributes {p = #tf_type.shape<10x?x30>}
func private @shape_attr_dyn() attributes {p = #tf_type.shape<10x?x30>}
// CHECK: func private @shape_attr_unranked() attributes {p = #tf_type.shape<*>}
func private @shape_attr_unranked() attributes {p = #tf_type.shape<*>}
// CHECK: func private @version_attr() attributes {p = #tf_type.version<producer = 42, min_consumer = 33>}
func private @version_attr() attributes {p = #tf_type.version<producer = 42, min_consumer = 33>}
// CHECK: func private @version_attr_bad_consumer() attributes {p = #tf_type.version<producer = 1, min_consumer = 1, bad_consumers = [1, 2, 5, 12]>}
func private @version_attr_bad_consumer() attributes {p = #tf_type.version<producer = 1, min_consumer = 1, bad_consumers = [1, 2, 5, 12]>}

// CHECK: !tf_type.variant
func private @variant_without_type(!tf_type.variant) -> ()

// CHECK: !tf_type.variant<tensor<?xf32>>
func private @variant_with_type(!tf_type.variant<tensor<?xf32>>) -> ()

// CHECK: !tf_type.variant<tensor<3xf32>, tensor<2xi32>>
func private @variant_with_multiple_types(!tf_type.variant<tensor<3xf32>, tensor<2xi32>>) -> ()

// CHECK: tensor<*x!tf_type.variant<tensor<?xf32>>>
func private @variant_element_type(tensor<*x!tf_type.variant<tensor<?xf32>>>) -> ()

// CHECK: tensor<!tf_type.variant<tensor<?x!tf_type.variant<tensor<?xf32>>>>>
func private @nested_variant(tensor<!tf_type.variant<tensor<?x!tf_type.variant<tensor<?xf32>>>>>) -> ()

// CHECK: !tf_type.variantref
func private @variantref(!tf_type.variantref) -> ()
