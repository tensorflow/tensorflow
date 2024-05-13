
// RUN: tf-opt -tfl-legalize-tensorlist -canonicalize -split-input-file %s | FileCheck %s

// CHECK-LABEL: listReserveScalarShapeI32
func.func @listReserveScalarShapeI32(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<!tf_type.variant<tensor<*xi32>>> {
  %0 = "tf.TensorListReserve"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<!tf_type.variant<tensor<*xi32>>>
  // CHECK: %0 = "tfl.custom"(%arg0, %arg1) <{custom_code = "TensorListReserve", custom_option = #tfl<const_bytes : "0x02">}> : (tensor<i32>, tensor<i32>) -> tensor<!tf_type.variant<tensor<*xi32>>>
  func.return %0 : tensor<!tf_type.variant<tensor<*xi32>>>
}

// -----

// CHECK-LABEL: listReserve1DShapeI32
func.func @listReserve1DShapeI32(%arg0: tensor<2xi32>, %arg1: tensor<i32>) -> tensor<!tf_type.variant<tensor<*xi32>>> {
  %0 = "tf.TensorListReserve"(%arg0, %arg1) : (tensor<2xi32>, tensor<i32>) -> tensor<!tf_type.variant<tensor<*xi32>>>
  // CHECK: %0 = "tfl.custom"(%arg0, %arg1) <{custom_code = "TensorListReserve", custom_option = #tfl<const_bytes : "0x02">}> : (tensor<2xi32>, tensor<i32>) -> tensor<!tf_type.variant<tensor<*xi32>>>
  func.return %0 : tensor<!tf_type.variant<tensor<*xi32>>>
}

// -----

// CHECK-LABEL: listReserveScalarShapeFloat
func.func @listReserveScalarShapeFloat(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<!tf_type.variant<tensor<*xf32>>> {
  %0 = "tf.TensorListReserve"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<!tf_type.variant<tensor<*xf32>>>
  // CHECK: %0 = "tfl.custom"(%arg0, %arg1) <{custom_code = "TensorListReserve", custom_option = #tfl<const_bytes : "0x00">}> : (tensor<i32>, tensor<i32>) -> tensor<!tf_type.variant<tensor<*xf32>>>
  func.return %0 : tensor<!tf_type.variant<tensor<*xf32>>>
}

// -----

// CHECK-LABEL: listReserveScalarShapeLong
func.func @listReserveScalarShapeLong(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<!tf_type.variant<tensor<*xi64>>> {
  %0 = "tf.TensorListReserve"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<!tf_type.variant<tensor<*xi64>>>
  // CHECK: %0 = "tfl.custom"(%arg0, %arg1) <{custom_code = "TensorListReserve", custom_option = #tfl<const_bytes : "0x04">}> : (tensor<i32>, tensor<i32>) -> tensor<!tf_type.variant<tensor<*xi64>>>
  func.return %0 : tensor<!tf_type.variant<tensor<*xi64>>>
}

// -----

// CHECK-LABEL: listReserveScalarShapeBool
func.func @listReserveScalarShapeBool(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<!tf_type.variant<tensor<*xi1>>> {
  %0 = "tf.TensorListReserve"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<!tf_type.variant<tensor<*xi1>>>
  // CHECK: %0 = "tfl.custom"(%arg0, %arg1) <{custom_code = "TensorListReserve", custom_option = #tfl<const_bytes : "0x06">}> : (tensor<i32>, tensor<i32>) -> tensor<!tf_type.variant<tensor<*xi1>>>
  func.return %0 : tensor<!tf_type.variant<tensor<*xi1>>>
}

// -----

// CHECK-LABEL: listStack
func.func @listStack(%arg0: tensor<!tf_type.variant<tensor<*xi32>>>, %arg1: tensor<i32>) -> tensor<*xi32> {
  %0 = "tf.TensorListStack"(%arg0, %arg1) : (tensor<!tf_type.variant<tensor<*xi32>>>, tensor<i32>) -> tensor<*xi32>
  // CHECK: %0 = "tfl.custom"(%arg0, %arg1) <{custom_code = "TensorListStack", custom_option = #tfl<const_bytes : "0x">}> : (tensor<!tf_type.variant<tensor<*xi32>>>, tensor<i32>) -> tensor<*xi32>
  func.return %0 : tensor<*xi32>
}

// -----

// CHECK-LABEL: listSetItem
func.func @listSetItem(%arg0: tensor<!tf_type.variant<tensor<*xi32>>>, %arg1: tensor<i32>, %arg2: tensor<*xi32>) -> tensor<!tf_type.variant<tensor<*xi32>>> {
  %0 = "tf.TensorListSetItem"(%arg0, %arg1, %arg2) : (tensor<!tf_type.variant<tensor<*xi32>>>, tensor<i32>, tensor<*xi32>) -> tensor<!tf_type.variant<tensor<*xi32>>>
  // CHECK: %0 = "tfl.custom"(%arg0, %arg1, %arg2) <{custom_code = "TensorListSetItem", custom_option = #tfl<const_bytes : "0x">}> : (tensor<!tf_type.variant<tensor<*xi32>>>, tensor<i32>, tensor<*xi32>) -> tensor<!tf_type.variant<tensor<*xi32>>>
  func.return %0 : tensor<!tf_type.variant<tensor<*xi32>>>
}

// -----

// CHECK-LABEL: listGetItem
func.func @listGetItem(%arg0: tensor<!tf_type.variant<tensor<*xi32>>>, %arg1: tensor<i32>, %arg2: tensor<2xi32>) -> tensor<2xi32> {
  %0 = "tf.TensorListGetItem"(%arg0, %arg1, %arg2) : (tensor<!tf_type.variant<tensor<*xi32>>>, tensor<i32>, tensor<2xi32>) -> tensor<2xi32>
  // CHECK: %0 = "tfl.custom"(%arg0, %arg1, %arg2) <{custom_code = "TensorListGetItem", custom_option = #tfl<const_bytes : "0x">}> : (tensor<!tf_type.variant<tensor<*xi32>>>, tensor<i32>, tensor<2xi32>) -> tensor<2xi32>
  func.return %0 : tensor<2xi32>
}

// -----

// CHECK-LABEL: listFromTensor
func.func @listFromTensor(%tensor: tensor<3xi32>, %shape : tensor<?xi32>) -> tensor<!tf_type.variant<tensor<i32>>> {
  %0 = "tf.TensorListFromTensor"(%tensor, %shape) : (tensor<3xi32>, tensor<?xi32>) -> tensor<!tf_type.variant<tensor<i32>>>
  func.return %0 : tensor<!tf_type.variant<tensor<i32>>>
  // CHECK: %0 = "tfl.custom"(%arg0, %arg1) <{custom_code = "TensorListFromTensor", custom_option = #tfl<const_bytes : "0x">}> : (tensor<3xi32>, tensor<?xi32>) -> tensor<!tf_type.variant<tensor<i32>>>
}

// -----

// CHECK-LABEL: typeNotSupportedNotLegalized
func.func @typeNotSupportedNotLegalized(%arg0: tensor<!tf_type.variant<tensor<*xf64>>>, %arg1: tensor<i32>, %arg2: tensor<*xf64>) -> tensor<!tf_type.variant<tensor<*xf64>>> {
  %0 = "tf.TensorListSetItem"(%arg0, %arg1, %arg2) : (tensor<!tf_type.variant<tensor<*xf64>>>, tensor<i32>, tensor<*xf64>) -> tensor<!tf_type.variant<tensor<*xf64>>>
  // CHECK-NOT: "tfl.custom"
  // CHECK-MSG: Tried legalizing to tfl custom tensorlist ops, but not all can be supported.
  func.return %0 : tensor<!tf_type.variant<tensor<*xf64>>>
}

// -----

// CHECK-LABEL: listLength
func.func @listLength(%arg0: tensor<!tf_type.variant<tensor<*xi32>>>) -> tensor<i32> {
  %0 = "tf.TensorListLength"(%arg0) : (tensor<!tf_type.variant<tensor<*xi32>>>) -> tensor<i32>
  // CHECK: %0 = "tfl.custom"(%arg0) <{custom_code = "TensorListLength", custom_option = #tfl<const_bytes : "0x">}> : (tensor<!tf_type.variant<tensor<*xi32>>>) -> tensor<i32>
  func.return %0 : tensor<i32>
}

// -----

// CHECK-LABEL: listEmptyToListReserve
func.func @listEmptyToListReserve(%arg0: tensor<?xi32>, %arg1: tensor<i32>) -> tensor<!tf_type.variant<tensor<*xi64>>> {
  %0 = "tf.EmptyTensorList"(%arg0, %arg1) : (tensor<?xi32>, tensor<i32>) -> tensor<!tf_type.variant<tensor<*xi64>>>
  // CHECK: %cst = arith.constant dense<0> : tensor<i32>
  // CHECK: %0 = "tfl.custom"(%arg0, %cst) <{custom_code = "TensorListReserve", custom_option = #tfl<const_bytes : "0x04">}> : (tensor<?xi32>, tensor<i32>) -> tensor<!tf_type.variant<tensor<*xi64>>>
  func.return %0 : tensor<!tf_type.variant<tensor<*xi64>>>
}

// -----

// CHECK-LABEL: listElementShape
func.func @listElementShape(%arg0: tensor<!tf_type.variant<tensor<*xi32>>>) -> tensor<*xi32> {
  %0 = "tf.TensorListElementShape"(%arg0) : (tensor<!tf_type.variant<tensor<*xi32>>>) -> tensor<*xi32>
  // CHECK: %0 = "tfl.custom"(%arg0) <{custom_code = "TensorListElementShape", custom_option = #tfl<const_bytes : "0x">}> : (tensor<!tf_type.variant<tensor<*xi32>>>) -> tensor<*xi32>
  func.return %0 : tensor<*xi32>
}

// -----

// CHECK-LABEL: listPopBack
func.func @listPopBack(%arg0: tensor<!tf_type.variant<tensor<2xi32>>>, %arg1: tensor<1xi32>) -> (tensor<!tf_type.variant<tensor<2xi32>>>, tensor<2xi32>)  {
  %0, %1 = "tf.TensorListPopBack"(%arg0, %arg1) : (tensor<!tf_type.variant<tensor<2xi32>>>, tensor<1xi32>) -> (tensor<!tf_type.variant<tensor<2xi32>>>, tensor<2xi32>)
  // CHECK: %0:2 = "tfl.custom"(%arg0, %arg1) <{custom_code = "TensorListPopBack", custom_option = #tfl<const_bytes : "0x">}> : (tensor<!tf_type.variant<tensor<2xi32>>>, tensor<1xi32>) -> (tensor<!tf_type.variant<tensor<2xi32>>>, tensor<2xi32>)
  func.return %0, %1 : tensor<!tf_type.variant<tensor<2xi32>>>, tensor<2xi32>
}

// -----

// CHECK-LABEL: listPushBack
func.func @listPushBack(%arg0: tensor<!tf_type.variant<tensor<?x1xf32>>>, %arg1: tensor<16x1xf32>) -> tensor<!tf_type.variant<tensor<?x1xf32>>>  {
  %0 = "tf.TensorListPushBack"(%arg0, %arg1) : (tensor<!tf_type.variant<tensor<?x1xf32>>>, tensor<16x1xf32>) -> tensor<!tf_type.variant<tensor<?x1xf32>>>
  // CHECK: %0 = "tfl.custom"(%arg0, %arg1) <{custom_code = "TensorListPushBack", custom_option = #tfl<const_bytes : "0x">}> : (tensor<!tf_type.variant<tensor<?x1xf32>>>, tensor<16x1xf32>) -> tensor<!tf_type.variant<tensor<?x1xf32>>>
  func.return %0: tensor<!tf_type.variant<tensor<?x1xf32>>>
}

// -----

// CHECK-LABEL: variantAddN
func.func @variantAddN(%arg0: tensor<!tf_type.variant<tensor<*xi32>>>, %arg1: tensor<!tf_type.variant<tensor<*xi32>>>) -> tensor<!tf_type.variant<tensor<*xi32>>> {
  %1 = "tf.AddN"(%arg0, %arg1) : (tensor<!tf_type.variant<tensor<*xi32>>>, tensor<!tf_type.variant<tensor<*xi32>>>) -> tensor<!tf_type.variant<tensor<*xi32>>>
  // CHECK: %0 = "tfl.custom"(%arg0, %arg1) <{custom_code = "VariantAddN", custom_option = #tfl<const_bytes : "0x">}> : (tensor<!tf_type.variant<tensor<*xi32>>>, tensor<!tf_type.variant<tensor<*xi32>>>) -> tensor<!tf_type.variant<tensor<*xi32>>>
  func.return %1 : tensor<!tf_type.variant<tensor<*xi32>>>
}

// -----

// CHECK-LABEL: variantZeroesLikeNoLegalize
func.func @variantZeroesLikeNoLegalize(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<!tf_type.variant<tensor<*xi64>>> {
  %0 = "tf.TensorListReserve"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<!tf_type.variant<tensor<*xi64>>>
  %1 = "tf.ZerosLike"(%0) : (tensor<!tf_type.variant<tensor<*xi64>>>) -> tensor<!tf_type.variant<tensor<*xi64>>>
  // CHECK-NOT: "tfl.custom"
  // CHECK-MSG: Tried legalizing to tfl custom tensorlist ops, but not all can be supported.
  func.return %1 : tensor<!tf_type.variant<tensor<*xi64>>>
}
