// RUN: tf-opt %s -canonicalize | FileCheck %s

// CHECK-LABEL: func @testShape
func.func @testShape(tensor<f32>, tensor<1x32x32x16xf32>, tensor<*xf32>) -> (tensor<0xi32>, tensor<?xi32>, tensor<?xi32>) {
^bb0(%arg0: tensor<f32>, %arg1: tensor<1x32x32x16xf32>, %arg2: tensor<*xf32>):

  // CHECK-DAG: tf.Const{{.*}} dense<> : tensor<0xi32>
  %0 = "tf.Shape"(%arg0) {T = "tfdtype$DT_FLOAT", output = "tfdtype$DT_INT32"} : (tensor<f32>) -> tensor<0xi32>

  // Result shape need not be static. Folding harness uses TensorFlow constant
  // in that case.
  // CHECK-DAG: "tf.Const"() <{value = dense<[1, 32, 32, 16]> : tensor<4xi32>}> : () -> tensor<?xi32>
  %1 = "tf.Shape"(%arg1) {T = "tfdtype$DT_FLOAT", output = "tfdtype$DT_INT32"} : (tensor<1x32x32x16xf32>) -> tensor<?xi32>

  // CHECK: "tf.Shape"(%arg2) {T = "tfdtype$DT_FLOAT", output = "tfdtype$DT_INT32"} : (tensor<*xf32>) -> tensor<?xi32>
  %2 = "tf.Shape"(%arg2) {T = "tfdtype$DT_FLOAT", output = "tfdtype$DT_INT32"} : (tensor<*xf32>) -> tensor<?xi32>

  func.return %0, %1, %2 : tensor<0xi32>, tensor<?xi32>, tensor<?xi32>
}

// CHECK-LABEL: func @testPow
// CHECK-SAME:(%[[ARG_0:.*]]: tensor<4xf32>, %[[ARG_1:.*]]: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>)
func.func @testPow(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) {

  %cst_zero = arith.constant dense<0.0> : tensor<f32>
  %cst_one = arith.constant dense<1.0> : tensor<f32>

  // CHECK-DAG: %[[RES_NO_FOLD:.*]] = "tf.Pow"(%arg0, %arg1)
  %0 = "tf.Pow"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

  // CHECK-DAG: %[[POW_ZERO:.*]] = "tf.Const"() <{value = dense<1.000000e+00> : tensor<4xf32>}> : () -> tensor<4xf32>
  %1 = "tf.Pow"(%arg0, %cst_zero) : (tensor<4xf32>, tensor<f32>) -> tensor<4xf32>

  // CHECK-NOT: "tf.Pow"
  %2 = "tf.Pow"(%arg0, %cst_one) : (tensor<4xf32>, tensor<f32>) -> tensor<4xf32>

  // CHECK: return %[[RES_NO_FOLD]], %[[POW_ZERO]], %[[ARG_0]]
  func.return %0, %1, %2 : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
}

// CHECK-LABEL: func @testEmpty32
func.func @testEmpty32() -> (tensor<5xi32>) {
  %0 = "tf.Const"() { value = dense<5> : tensor<i32> } : () -> tensor<i32>

  // CHECK: [[VAL:%.+]] = "tf.Const"() <{value = dense<0> : tensor<5xi32>}>
  // CHECK: return [[VAL]]
  %1 = "tf.Empty"(%0) : (tensor<i32>) -> (tensor<5xi32>)
  func.return %1 : tensor<5xi32>
}

// CHECK-LABEL: func @testEmpty64
func.func @testEmpty64() -> (tensor<5xi64>) {
  %0 = "tf.Const"() { value = dense<5> : tensor<i32> } : () -> tensor<i32>

  // CHECK: [[VAL:%.+]] = "tf.Const"() <{value = dense<0> : tensor<5xi64>}>
  // CHECK: return [[VAL]] : tensor<5xi64>
  %1 = "tf.Empty"(%0) : (tensor<i32>) -> (tensor<5xi64>)
  func.return %1 : tensor<5xi64>
}

// CHECK-LABEL: func @testEmptyFloat
func.func @testEmptyFloat() -> (tensor<5xf64>) {
  %0 = "tf.Const"() { value = dense<5> : tensor<i32> } : () -> tensor<i32>

  // CHECK: [[VAL:%.+]] = "tf.Const"() <{value =  dense<0.000000e+00> : tensor<5xf64>}>
  // CHECK: return [[VAL]]
  %1 = "tf.Empty"(%0) : (tensor<i32>) -> (tensor<5xf64>)
  func.return %1 : tensor<5xf64>
}

// CHECK-LABEL: func @testEmptyf16
func.func @testEmptyf16() -> (tensor<5xf16>) {
  %0 = "tf.Const"() { value = dense<5> : tensor<i32> } : () -> tensor<i32>

  // CHECK: [[VAL:%.+]] = "tf.Const"() <{value =  dense<0.000000e+00> : tensor<5xf16>}>
  // CHECK: return [[VAL]]
  %1 = "tf.Empty"(%0) : (tensor<i32>) -> (tensor<5xf16>)
  func.return %1 : tensor<5xf16>
}

// CHECK-LABEL: func @testEmptybf16
func.func @testEmptybf16() -> (tensor<5xbf16>) {
  %0 = "tf.Const"() { value = dense<5> : tensor<i32> } : () -> tensor<i32>

  // CHECK: [[VAL:%.+]] = "tf.Const"() <{value =  dense<0.000000e+00> : tensor<5xbf16>}>
  // CHECK: return [[VAL]]
  %1 = "tf.Empty"(%0) : (tensor<i32>) -> (tensor<5xbf16>)
  func.return %1 : tensor<5xbf16>
}

// CHECK-LABEL: func @testShapeN
func.func @testShapeN(%arg0: tensor<f32>, %arg1: tensor<1x32x32x16xf32>) -> (tensor<0xi64>, tensor<4xi64>) {

  // CHECK-DAG: %[[SHAPE0:.*]] = "tf.Const"() <{value = dense<> : tensor<0xi64>}>
  // CHECK-DAG: %[[SHAPE1:.*]] = "tf.Const"() <{value = dense<[1, 32, 32, 16]> : tensor<4xi64>}>
  %0:2 = "tf.ShapeN"(%arg0, %arg1) : (tensor<f32>, tensor<1x32x32x16xf32>) -> (tensor<0xi64>, tensor<4xi64>)

  // CHECK: return %[[SHAPE0]], %[[SHAPE1]]
  func.return %0#0, %0#1 : tensor<0xi64>, tensor<4xi64>
}

// CHECK-LABEL: func @testShapeNPartialStatic
func.func @testShapeNPartialStatic(%arg0: tensor<f32>, %arg1: tensor<2x?x3xf32>, %arg2: tensor<1x32x32x16xf32>, %arg3: tensor<*xf32>) -> (tensor<0xi64>, tensor<3xi64>, tensor<4xi64>, tensor<?xi64>) {
  // CHECK-DAG: %[[SHAPE0:.*]] = "tf.Const"() <{value = dense<> : tensor<0xi64>}>
  // CHECK-DAG: %[[SHAPE2:.*]] = "tf.Const"() <{value = dense<[1, 32, 32, 16]> : tensor<4xi64>}>
  // CHECK: %[[SHAPE13:.*]]:2 = "tf.ShapeN"(%arg1, %arg3) : (tensor<2x?x3xf32>, tensor<*xf32>) -> (tensor<3xi64>, tensor<?xi64>)
  %0:4 = "tf.ShapeN"(%arg0, %arg1, %arg2, %arg3) : (tensor<f32>, tensor<2x?x3xf32>, tensor<1x32x32x16xf32>, tensor<*xf32>) -> (tensor<0xi64>, tensor<3xi64>, tensor<4xi64>, tensor<?xi64>)

  // CHECK: return %[[SHAPE0]], %[[SHAPE13]]#0, %[[SHAPE2]], %[[SHAPE13]]#1
  func.return %0#0, %0#1, %0#2, %0#3 : tensor<0xi64>, tensor<3xi64>, tensor<4xi64>, tensor<?xi64>
}

// CHECK-LABEL: func @testShapeNOneDynamic
func.func @testShapeNOneDynamic(%arg0: tensor<f32>, %arg1: tensor<1x32x32x16xf32>, %arg2: tensor<*xf32>) -> (tensor<0xi64>, tensor<4xi64>, tensor<?xi64>) {
  // CHECK-DAG: %[[SHAPE0:.*]] = "tf.Const"() <{value = dense<> : tensor<0xi64>}>
  // CHECK-DAG: %[[SHAPE1:.*]] = "tf.Const"() <{value = dense<[1, 32, 32, 16]> : tensor<4xi64>}>
  // CHECK: %[[SHAPE2:.*]] = "tf.Shape"(%arg2) : (tensor<*xf32>) -> tensor<?xi64>
  %0:3 = "tf.ShapeN"(%arg0, %arg1, %arg2) : (tensor<f32>, tensor<1x32x32x16xf32>, tensor<*xf32>) -> (tensor<0xi64>, tensor<4xi64>, tensor<?xi64>)

  // CHECK: return %[[SHAPE0]], %[[SHAPE1]], %[[SHAPE2]]
  func.return %0#0, %0#1, %0#2 : tensor<0xi64>, tensor<4xi64>, tensor<?xi64>
}

// CHECK-LABEL: func @testShapeNToShape
func.func @testShapeNToShape(%arg0: tensor<*xf32>) -> tensor<?xi64> {
  // CHECK: %[[SHAPE0:.*]] = "tf.Shape"(%arg0) : (tensor<*xf32>) -> tensor<?xi64>
  %0:1 = "tf.ShapeN"(%arg0) : (tensor<*xf32>) -> tensor<?xi64>

  // CHECK: return %[[SHAPE0]]
  func.return %0#0 : tensor<?xi64>
}

// CHECK-LABEL: func @testLeakyRelu
func.func @testLeakyRelu(%arg0 : tensor<16xf32>) -> (tensor<16xf32>, tensor<f32>, tensor<f32>, tensor<16xf32>) {
  %pos = arith.constant dense<5.0> : tensor<f32>
  %neg = arith.constant dense<-5.0> : tensor<f32>
  %no = "tf.LeakyRelu"(%arg0) {alpha = 0.2 : f32} : (tensor<16xf32>) -> tensor<16xf32>
  %0 = "tf.LeakyRelu"(%pos) {alpha = 0.3 : f32} : (tensor<f32>) -> tensor<f32>
  %1 = "tf.LeakyRelu"(%neg) {alpha = 0.2 : f32} : (tensor<f32>) -> tensor<f32>
  %2 = "tf.LeakyRelu"(%arg0) {alpha = 3.0 : f32} : (tensor<16xf32>) -> tensor<16xf32>
  // CHECK-DAG: [[POS:%.*]] = "tf.Const{{.*}} dense<5.000000e+00> : tensor<f32>
  // CHECK-DAG: [[NEG:%.*]] = "tf.Const{{.*}} dense<-1.000000e+00> : tensor<f32>
  // CHECK: [[NC1:%.*]] = "tf.LeakyRelu"(%arg0) <{alpha = 2.000000e-01 : f32}> : (tensor<16xf32>) -> tensor<16xf32>
  // CHECK: [[NC2:%.*]] = "tf.LeakyRelu"(%arg0) <{alpha = 3.000000e+00 : f32}> : (tensor<16xf32>) -> tensor<16xf32>
  // CHECK: return [[NC1]], [[POS]], [[NEG]], [[NC2]]
  func.return %no, %0, %1, %2 : tensor<16xf32>, tensor<f32>, tensor<f32>, tensor<16xf32>
}

// CHECK-LABEL: func @tfConst
func.func @tfConst() -> (tensor<4xf32>, tensor<1x1x6x2xf32>) {
  %0 = "tf.Const"() {device = "", name = "Const", dtype = "tfdtype$DT_FLOAT", value = #tf_type<tensor_proto : "0x746674656E736F722464747970653A2044545F464C4F41540A74656E736F725F7368617065207B0A202064696D207B0A2020202073697A653A20340A20207D0A7D0A74656E736F725F636F6E74656E743A20225C3030305C3030305C323430405C3030305C30303020405C3030305C303030205C3330315C3030305C3030305C3230305C323737220A"> : tensor<4xf32>} : () -> tensor<4xf32>
  %21 = "tf.Const"() {device = "", name = "Const_143", dtype = "tfdtype$DT_FLOAT", value = dense<0.24288677062973696> : tensor<1x1x6x2xf32>} : () -> tensor<1x1x6x2xf32>
  // CHECK-DAG: value = #tf_type<tensor_proto
  // CHECK-DAG: tf.Const{{.*}} dense<0.242886767> : tensor<1x1x6x2xf32>
  func.return %0, %21 : tensor<4xf32>, tensor<1x1x6x2xf32>
}

// CHECK-LABEL: func @testAdd() -> tensor<2x2xi32>
func.func @testAdd() -> tensor<2x2xi32> {
^bb0:
  %0 = arith.constant dense<[[0, 1], [2, 3]]> : tensor<2x2xi32>
  %1 = arith.constant dense<1> : tensor<2xi32>
  %2 = "tf.Add"(%0, %1) {device = "", name = "add"} : (tensor<2x2xi32>, tensor<2xi32>) -> tensor<2x2xi32>
  // CHECK:         [[cst:%.*]] = "tf.Const{{.*}} dense<{{\[\[}}1, 2], {{\[}}3, 4]]> : tensor<2x2xi32>
  // CHECK-NEXT:    return [[cst]] : tensor<2x2xi32>
  func.return %2: tensor<2x2xi32>
}

// CHECK-LABEL: testSimpleConcatOffset
func.func @testSimpleConcatOffset() -> (tensor<3xi32>, tensor<3xi32>, tensor<3xi32>) {
  %concat_dim = arith.constant dense<1> : tensor<i32>
  %shape0 = arith.constant dense<[2, 2, 7]> : tensor<3xi32>
  %shape1 = arith.constant dense<[2, 3, 7]> : tensor<3xi32>
  %shape2 = arith.constant dense<[2, 5, 7]> : tensor<3xi32>

  // CHECK-DAG: [[OFFSET_0:%.*]] = "tf.Const{{.*}} dense<0> : tensor<3xi32>
  // CHECK-DAG: [[OFFSET_1:%.*]] = "tf.Const{{.*}} dense<[0, 2, 0]> : tensor<3xi32>
  // CHECK-DAG: [[OFFSET_2:%.*]] = "tf.Const{{.*}} dense<[0, 5, 0]> : tensor<3xi32>

  %offset:3 = "tf.ConcatOffset"(%concat_dim, %shape0, %shape1, %shape2) : (tensor<i32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>) -> (tensor<3xi32>, tensor<3xi32>, tensor<3xi32>)

  // CHECK: return [[OFFSET_0]], [[OFFSET_1]], [[OFFSET_2]]
  func.return %offset#0, %offset#1, %offset#2: tensor<3xi32>, tensor<3xi32>, tensor<3xi32>
}

// CHECK-LABEL: testConcatOffsetWithZeros
func.func @testConcatOffsetWithZeros() -> (tensor<3xi32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>) {
  %concat_dim = arith.constant dense<1> : tensor<i32>
  %shape0 = arith.constant dense<0> : tensor<3xi32>
  %shape1 = arith.constant dense<[0, 3, 0]> : tensor<3xi32>
  %shape2 = arith.constant dense<[0, 5, 0]> : tensor<3xi32>
  %shape3 = arith.constant dense<0> : tensor<3xi32>

  // CHECK-DAG: [[OFFSET_0:%.*]] = "tf.Const{{.*}} dense<0> : tensor<3xi32>
  // CHECK-DAG: [[OFFSET_2:%.*]] = "tf.Const{{.*}} dense<[0, 3, 0]> : tensor<3xi32>
  // CHECK-DAG: [[OFFSET_3:%.*]] = "tf.Const{{.*}} dense<[0, 8, 0]> : tensor<3xi32>

  %offset:4 = "tf.ConcatOffset"(%concat_dim, %shape0, %shape1, %shape2, %shape3) : (tensor<i32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>) -> (tensor<3xi32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>)

  // CHECK: return [[OFFSET_0]], [[OFFSET_0]], [[OFFSET_2]], [[OFFSET_3]]
  func.return %offset#0, %offset#1, %offset#2, %offset#3: tensor<3xi32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>
}

// CHECK-LABEL: testConcatOffsetNegativeConcatDim
func.func @testConcatOffsetNegativeConcatDim() -> (tensor<3xi32>, tensor<3xi32>, tensor<3xi32>) {
  %concat_dim = arith.constant dense<-1> : tensor<i32>
  %shape0 = arith.constant dense<[2, 8, 3]> : tensor<3xi32>
  %shape1 = arith.constant dense<[2, 8, 5]> : tensor<3xi32>
  %shape2 = arith.constant dense<[2, 8, 7]> : tensor<3xi32>

  // CHECK-DAG: [[OFFSET_0:%.*]] = "tf.Const{{.*}} dense<0> : tensor<3xi32>
  // CHECK-DAG: [[OFFSET_1:%.*]] = "tf.Const{{.*}} dense<[0, 0, 3]> : tensor<3xi32>
  // CHECK-DAG: [[OFFSET_2:%.*]] = "tf.Const{{.*}} dense<[0, 0, 8]> : tensor<3xi32>

  %offset:3 = "tf.ConcatOffset"(%concat_dim, %shape0, %shape1, %shape2) : (tensor<i32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>) -> (tensor<3xi32>, tensor<3xi32>, tensor<3xi32>)

  // CHECK: return [[OFFSET_0]], [[OFFSET_1]], [[OFFSET_2]]
  func.return %offset#0, %offset#1, %offset#2: tensor<3xi32>, tensor<3xi32>, tensor<3xi32>
}

// CHECK-LABEL: testConcatOffsetNonConstConcatDim
func.func @testConcatOffsetNonConstConcatDim(%concat_dim: tensor<i32>) -> (tensor<3xi32>, tensor<3xi32>) {
  %shape0 = arith.constant dense<[2, 2, 7]> : tensor<3xi32>
  %shape1 = arith.constant dense<[2, 3, 7]> : tensor<3xi32>

  // CHECK: tf.ConcatOffset
  %offset:2 = "tf.ConcatOffset"(%concat_dim, %shape0, %shape1) : (tensor<i32>, tensor<3xi32>, tensor<3xi32>) -> (tensor<3xi32>, tensor<3xi32>)

  func.return %offset#0, %offset#1: tensor<3xi32>, tensor<3xi32>
}

// CHECK-LABEL: testConcatOffsetNonConstShape
func.func @testConcatOffsetNonConstShape(%shape1: tensor<3xi32>) -> (tensor<3xi32>, tensor<3xi32>) {
  %concat_dim = arith.constant dense<1> : tensor<i32>
  %shape0 = arith.constant dense<[2, 2, 7]> : tensor<3xi32>

  // CHECK: tf.ConcatOffset
  %offset:2 = "tf.ConcatOffset"(%concat_dim, %shape0, %shape1) : (tensor<i32>, tensor<3xi32>, tensor<3xi32>) -> (tensor<3xi32>, tensor<3xi32>)

  func.return %offset#0, %offset#1: tensor<3xi32>, tensor<3xi32>
}

// CHECK-LABEL: testConcatOffsetBadNegativeConcatDim
func.func @testConcatOffsetBadNegativeConcatDim() -> (tensor<3xi32>, tensor<3xi32>) {
  %concat_dim = arith.constant dense<-4> : tensor<i32>
  %shape0 = arith.constant dense<[2, 2, 7]> : tensor<3xi32>
  %shape1 = arith.constant dense<[2, 3, 7]> : tensor<3xi32>

  // CHECK: tf.ConcatOffset
  %offset:2 = "tf.ConcatOffset"(%concat_dim, %shape0, %shape1) : (tensor<i32>, tensor<3xi32>, tensor<3xi32>) -> (tensor<3xi32>, tensor<3xi32>)

  func.return %offset#0, %offset#1: tensor<3xi32>, tensor<3xi32>
}

// CHECK-LABEL: testConcatOffsetBadPositiveConcatDim
func.func @testConcatOffsetBadPositiveConcatDim() -> (tensor<3xi32>, tensor<3xi32>) {
  %concat_dim = arith.constant dense<3> : tensor<i32>
  %shape0 = arith.constant dense<[2, 2, 7]> : tensor<3xi32>
  %shape1 = arith.constant dense<[2, 3, 7]> : tensor<3xi32>

  // CHECK: tf.ConcatOffset
  %offset:2 = "tf.ConcatOffset"(%concat_dim, %shape0, %shape1) : (tensor<i32>, tensor<3xi32>, tensor<3xi32>) -> (tensor<3xi32>, tensor<3xi32>)

  func.return %offset#0, %offset#1: tensor<3xi32>, tensor<3xi32>
}

// CHECK-LABEL: testConcatOffsetDifferentNonConcatDimElements
func.func @testConcatOffsetDifferentNonConcatDimElements() -> (tensor<3xi32>, tensor<3xi32>) {
  %concat_dim = arith.constant dense<1> : tensor<i32>
  %shape0 = arith.constant dense<[2, 2, 7]> : tensor<3xi32>
  %shape1 = arith.constant dense<[2, 3, 8]> : tensor<3xi32>

  // CHECK: tf.ConcatOffset
  %offset:2 = "tf.ConcatOffset"(%concat_dim, %shape0, %shape1) : (tensor<i32>, tensor<3xi32>, tensor<3xi32>) -> (tensor<3xi32>, tensor<3xi32>)

  func.return %offset#0, %offset#1: tensor<3xi32>, tensor<3xi32>
}

// Ops with side effects should not get constant folded.
// CHECK-LABEL: func @testSideEffectOp() -> tensor<3xf32>
func.func @testSideEffectOp() -> tensor<3xf32> {
  %0 = arith.constant dense<[3]> : tensor<1xi32>
  %1 = "tf.RandomUniform"(%0) {device = "", seed = 3 : i64, seed2 = 5 : i64} : (tensor<1xi32>) -> tensor<3xf32>
  // CHECK: %[[random:.*]] = "tf.RandomUniform"
  // CHECK: return %[[random]]
  func.return %1: tensor<3xf32>
}

// Ops with unimplemented attributes which couldn't be added to the TFE_Op.
// CHECK-LABEL: func @testUnimplementedOp() -> (tensor<i32>, tensor<i32>)
func.func @testUnimplementedOp() -> (tensor<i32>, tensor<i32>) {
  %0 = arith.constant dense<1> : tensor<i32>
  %1 = arith.constant dense<2> : tensor<i32>
  %2 = "tf.Maximum"(%0, %1) {_output_shapes = ["tfshape$"]} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %3 = "tf.Minimum"(%0, %1) {random_attr = "hello"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %2, %3: tensor<i32>, tensor<i32>

// CHECK-DAG: %[[CST:.*]] = "tf.Const"() <{value = dense<2> : tensor<i32>}> : () -> tensor<i32>
// CHECK-DAG: %[[CST1:.*]] = "tf.Const"() <{value = dense<1> : tensor<i32>}> : () -> tensor<i32>
// CHECK-NEXT: return %[[CST]], %[[CST1]]
}

// Tests ops that variable shapes are correctly evaluated on static types.
// CHECK-LABEL: func @testVariableShape
func.func @testVariableShape(%arg0: tensor<!tf_type.resource<tensor<2x4xf32>>>) -> tensor<2xi32> {
  %0 = "tf.VariableShape"(%arg0) : (tensor<!tf_type.resource<tensor<2x4xf32>>>) -> tensor<2xi32>
  // CHECK:         [[cst:%.*]] = "tf.Const{{.*}} dense<{{\[}}2, 4]> : tensor<2xi32>
  // CHECK-NEXT:    return [[cst]] : tensor<2xi32>
  func.return %0: tensor<2xi32>
}

// Tests ops that tensor list shapes are correctly evaluated on static types.
// CHECK-LABEL: func @testTensorListElementShape
func.func @testTensorListElementShape(%arg0: tensor<!tf_type.variant<tensor<2x4xf32>>>) -> tensor<2xi32> {
  %0 = "tf.TensorListElementShape"(%arg0) : (tensor<!tf_type.variant<tensor<2x4xf32>>>) -> tensor<2xi32>
  // CHECK:         [[cst:%.*]] = "tf.Const{{.*}} dense<{{\[}}2, 4]> : tensor<2xi32>
  // CHECK-NEXT:    return [[cst]] : tensor<2xi32>
  func.return %0: tensor<2xi32>
}

func.func @RemoveTrivialAdd(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %cst = arith.constant dense<0.0> : tensor<2x2xf32>
  %0 = "tf.Add"(%arg0, %cst) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>

  // CHECK-LABEL: RemoveTrivialAdd
  // CHECK-NEXT: return %arg0 : tensor<2x2xf32>
}

func.func @RemoveTrivialAddBf16RHS(%arg0: tensor<2x2xbf16>) -> tensor<2x2xbf16> {
  %cst = arith.constant dense<0.0> : tensor<2x2xbf16>
  %0 = "tf.Add"(%arg0, %cst) : (tensor<2x2xbf16>, tensor<2x2xbf16>) -> tensor<2x2xbf16>
  func.return %0 : tensor<2x2xbf16>

  // CHECK-LABEL: RemoveTrivialAdd
  // CHECK-NEXT: return %arg0 : tensor<2x2xbf16>
}

func.func @RemoveTrivialAddBf16LHS(%arg0: tensor<2x2xbf16>) -> tensor<2x2xbf16> {
  %cst = arith.constant dense<0.0> : tensor<2x2xbf16>
  %0 = "tf.Add"(%cst, %arg0) : (tensor<2x2xbf16>, tensor<2x2xbf16>) -> tensor<2x2xbf16>
  func.return %0 : tensor<2x2xbf16>

  // CHECK-LABEL: RemoveTrivialAdd
  // CHECK-NEXT: return %arg0 : tensor<2x2xbf16>
}

func.func @RemoveTrivialAddV2(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %cst = arith.constant dense<0.0> : tensor<2x2xf32>
  %0 = "tf.AddV2"(%arg0, %cst) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>

  // CHECK-LABEL: RemoveTrivialAddV2
  // CHECK-NEXT: return %arg0 : tensor<2x2xf32>
}

func.func @RemoveTrivialSub(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %cst = arith.constant dense<0.0> : tensor<2x2xf32>
  %0 = "tf.Sub"(%arg0, %cst) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>

  // CHECK-LABEL: RemoveTrivialSub
  // CHECK-NEXT: return %arg0 : tensor<2x2xf32>
}

func.func @RemoveTrivialSubInt8(%arg0: tensor<2x2xi8>) -> tensor<2x2xi8> {
  %cst = arith.constant dense<0> : tensor<2x2xi8>
  %0 = "tf.Sub"(%arg0, %cst) : (tensor<2x2xi8>, tensor<2x2xi8>) -> tensor<2x2xi8>
  func.return %0 : tensor<2x2xi8>

  // CHECK-LABEL: RemoveTrivialSubInt8
  // CHECK-NEXT: return %arg0 : tensor<2x2xi8>
}

func.func @RemoveTrivialMul(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %cst = arith.constant dense<1.0> : tensor<2x2xf32>
  %0 = "tf.Mul"(%arg0, %cst) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>

  // CHECK-LABEL: RemoveTrivialMul
  // CHECK-NEXT: return %arg0 : tensor<2x2xf32>
}

func.func @RemoveTrivialDiv(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %cst = arith.constant dense<1.0> : tensor<2x2xf32>
  %0 = "tf.Div"(%arg0, %cst) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>

  // CHECK-LABEL: RemoveTrivialDiv
  // CHECK-NEXT: return %arg0 : tensor<2x2xf32>
}

func.func @RemoveTrivialRealDiv(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %cst = arith.constant dense<1.0> : tensor<2x2xf32>
  %0 = "tf.RealDiv"(%arg0, %cst) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>

  // CHECK-LABEL: RemoveTrivialRealDiv
  // CHECK-NEXT: return %arg0 : tensor<2x2xf32>
}

func.func @RemoveTrivialDivBf16RHS(%arg0: tensor<2x2xbf16>) -> tensor<2x2xbf16> {
  %cst = arith.constant dense<1.0> : tensor<2x2xbf16>
  %0 = "tf.Div"(%arg0, %cst) : (tensor<2x2xbf16>, tensor<2x2xbf16>) -> tensor<2x2xbf16>
  func.return %0 : tensor<2x2xbf16>

  // CHECK-LABEL: RemoveTrivialDiv
  // CHECK-NEXT: return %arg0 : tensor<2x2xbf16>
}

func.func @RemoveTrivialMulInt8(%arg0: tensor<2x2xi8>) -> tensor<2x2xi8> {
  %cst = arith.constant dense<1> : tensor<2x2xi8>
  %0 = "tf.Mul"(%cst, %arg0) : (tensor<2x2xi8>, tensor<2x2xi8>) -> tensor<2x2xi8>
  func.return %0 : tensor<2x2xi8>

  // CHECK-LABEL: RemoveTrivialMulInt8
  // CHECK-NEXT: return %arg0 : tensor<2x2xi8>
}

func.func @DivBf16LHS(%arg0: tensor<2x2xbf16>) -> tensor<2x2xbf16> {
  %cst = arith.constant dense<1.0> : tensor<2x2xbf16>
  %0 = "tf.Div"(%cst, %arg0) : (tensor<2x2xbf16>, tensor<2x2xbf16>) -> tensor<2x2xbf16>
  func.return %0 : tensor<2x2xbf16>

  // CHECK-LABEL: DivBf16LHS
  // CHECK: tf.Div
}

func.func @DontRemoveTrivialAdd(%arg0: tensor<1x2xf32>) -> tensor<2x2xf32> {
  %cst = arith.constant dense<0.0> : tensor<2x2xf32>
  %0 = "tf.AddV2"(%arg0, %cst) : (tensor<1x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>

  // CHECK-LABEL: DontRemoveTrivialAdd
  // CHECK: %[[CONST:.*]] = arith.constant dense<0.000000e+00> : tensor<2x2xf32>
  // CHECK: %[[RESULT:.*]] = "tf.AddV2"(%arg0, %[[CONST]]) : (tensor<1x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: return %[[RESULT]] : tensor<2x2xf32>
}

func.func @DontRemoveTrivialAdd2(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %cst = arith.constant dense<0.0> : tensor<2x2xf32>
  %0 = "tf.AddV2"(%arg0, %cst) : (tensor<?x?xf32> , tensor<2x2xf32>) -> tensor<?x?xf32>
  func.return %0 :tensor<?x?xf32>

  // CHECK-LABEL: DontRemoveTrivialAdd2
  // CHECK: %[[CONST:.*]] = arith.constant dense<0.000000e+00> : tensor<2x2xf32>
  // CHECK: %[[RESULT:.*]] = "tf.AddV2"(%arg0, %[[CONST]]) : (tensor<?x?xf32>, tensor<2x2xf32>) -> tensor<?x?xf32>
  // CHECK: return %[[RESULT]] : tensor<?x?xf32>
}

// Test no fold because of the broadcast.
func.func @DontRemoveTrivialMul(%arg0: tensor<1x6x8x1xf32>) -> tensor<1x6x8x1xf32> {
  %0 = "tf.Const"() {value = dense<2.000000e+00> : tensor<f32>} : () -> tensor<f32>
  %1 = "tf.Mul"(%arg0, %0) : (tensor<1x6x8x1xf32>, tensor<f32>) -> tensor<1x6x8x1xf32>
  func.return %1 : tensor<1x6x8x1xf32>
  // CHECK-LABEL: DontRemoveTrivialMul
  // CHECK: %[[CONST:.*]] = "tf.Const"() <{value = dense<2.000000e+00> : tensor<f32>}> : () -> tensor<f32>
  // CHECK: %[[RESULT:.*]] = "tf.Mul"(%arg0, %[[CONST]]) : (tensor<1x6x8x1xf32>, tensor<f32>) -> tensor<1x6x8x1xf32>
  // CHECK: return %[[RESULT]] : tensor<1x6x8x1xf32>
}

// Do not fold if the op doesn't follow the constant folding policy.
// The policy doesn't fold if the total operand size is larger than 128 MiB.
// CHECK-LABEL: DontFoldTileOperandsTooLarge
func.func @DontFoldTileOperandsTooLarge() -> (tensor<3x134217728xi8>) {
  %const_128mb_operand = "tf.Const"() {value = dense<42> : tensor<1x134217728xi8>} : () -> tensor<1x134217728xi8>
  %const_tile_3x = "tf.Const"() {value = dense<[3, 1]> : tensor<2xi32>} : () -> tensor<2xi32>
  %result = "tf.Tile"(%const_128mb_operand, %const_tile_3x) : (tensor<1x134217728xi8>, tensor<2xi32>) -> tensor<3x134217728xi8>
  // CHECK: [[TILE:%.*]] = "tf.Tile"
  // CHECK: return [[TILE]]
  func.return %result : tensor<3x134217728xi8>
}
// Do not fold if the op doesn't follow the constant folding policy.
// The policy doesn't fold if the total result size is larger than 8 KiB, and
// larger than 2x the total operand size.
// CHECK-LABEL: DontFoldTileResultTooLarge
func.func @DontFoldTileResultTooLarge() -> (tensor<3x3072xi8>) {
  %const_3kb_operand = "tf.Const"() {value = dense<42> : tensor<1x3072xi8>} : () -> tensor<1x3072xi8>
  %const_tile_3x = "tf.Const"() {value = dense<[3, 1]> : tensor<2xi32>} : () -> tensor<2xi32>
  %result = "tf.Tile"(%const_3kb_operand, %const_tile_3x) : (tensor<1x3072xi8>, tensor<2xi32>) -> tensor<3x3072xi8>
  // CHECK: [[TILE:%.*]] = "tf.Tile"
  // CHECK: return [[TILE]]
  func.return %result : tensor<3x3072xi8>
}
// Fold if the op follows the constant folding policy.
// CHECK-LABEL: FoldTile
func.func @FoldTile() -> (tensor<2x3072xi8>) {
  %const_3kb_operand = "tf.Const"() {value = dense<42> : tensor<1x3072xi8>} : () -> tensor<1x3072xi8>
  %const_tile_2x = "tf.Const"() {value = dense<[2, 1]> : tensor<2xi32>} : () -> tensor<2xi32>
  %result = "tf.Tile"(%const_3kb_operand, %const_tile_2x) : (tensor<1x3072xi8>, tensor<2xi32>) -> tensor<2x3072xi8>
  func.return %result : tensor<2x3072xi8>
  // CHECK-NOT: "tf.Tile"
  // CHECK: [[FOLDED:%.*]] = "tf.Const"() <{value = dense<42> : tensor<2x3072xi8>}> : () -> tensor<2x3072xi8>
  // CHECK: return [[FOLDED]]
}

// Verifies that very large splat constants are not materialized as Tensors for
// constant folding.
// CHECK-LABEL: @giant_tensor_input
func.func @giant_tensor_input() -> (tensor<*xf32>) {
  %input = "tf.Const"() {value = dense<1.000000e+00> : tensor<1024x1024x1024x1024xf32>} : () -> tensor<1024x1024x1024x1024xf32>
  %zero = "tf.Const"() {value = dense<0> : tensor<4xi32>} : () -> tensor<4xi32>
  %one = "tf.Const"() {value = dense<1> : tensor<4xi32>} : () -> tensor<4xi32>
  // CHECK: tf.StridedSlice
  %0 = "tf.StridedSlice"(%input, %zero, %one, %one) {begin_mask = 15 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<1024x1024x1024x1024xf32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) -> tensor<*xf32>

  func.return %0 : tensor<*xf32>
}

func.func @fold_conv() -> tensor<1x520x520x1xf32> {
  %0 = "tf.Const"() {value = dense<0.111111112> : tensor<3x3x1x1xf32>} : () -> tensor<3x3x1x1xf32>
  %1 = "tf.Const"() {value = dense<1.000000e+00> : tensor<1x520x520x1xf32>} : () -> tensor<1x520x520x1xf32>
  %2 = "tf.DepthwiseConv2dNative"(%1, %0) {data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<1x520x520x1xf32>, tensor<3x3x1x1xf32>) -> tensor<1x520x520x1xf32>
  func.return %2 : tensor<1x520x520x1xf32>

  // CHECK: tf.Const
  // CHECK-NOT: tf.DepthwiseConv2dNative
}

// CHECK-LABEL: DontFoldNoConstantFold
func.func @DontFoldNoConstantFold() -> tensor<8xf32> {
  %0 = "tf.Const"() {value = dense<[8]> : tensor<1xi32>} : () -> tensor<1xi32>
  %1 = "tf.Const"() {value = dense<[2, 1]> : tensor<2xi32>} : () -> tensor<2xi32>
  // CHECK: tf.StatelessRandomUniform
  %2 = "tf.StatelessRandomUniform"(%0, %1) : (tensor<1xi32>, tensor<2xi32>) -> tensor<8xf32>
  func.return %2 : tensor<8xf32>
}

// CHECK-LABEL: func @testBroadcastGradientArgsSameShape
func.func @testBroadcastGradientArgsSameShape() -> (tensor<0xi32>, tensor<0xi32>) {
  %s0 = "tf.Const"() {value = dense<[1, 2]> : tensor<2xi32>} : () -> tensor<2xi32>
  %s1 = "tf.Const"() {value = dense<[1, 2]> : tensor<2xi32>} : () -> tensor<2xi32>
  %r0, %r1 = "tf.BroadcastGradientArgs"(%s0, %s1) {} : (tensor<2xi32>, tensor<2xi32>) -> (tensor<0xi32>, tensor<0xi32>)

  // CHECK-DAG: %[[R:.*]] = "tf.Const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
  // CHECK-NOT: tf.BroadcastGradientArgs
  // CHECK: return %[[R]], %[[R]]

  func.return %r0, %r1 : tensor<0xi32>, tensor<0xi32>
}

// CHECK-LABEL: func @testBroadcastGradientArgs1
func.func @testBroadcastGradientArgs1() -> (tensor<1xi32>, tensor<0xi32>) {
  %s0 = "tf.Const"() {value = dense<[4]> : tensor<1xi32>} : () -> tensor<1xi32>
  %s1 = "tf.Const"() {value = dense<[2, 4]> : tensor<2xi32>} : () -> tensor<2xi32>
  %r0, %r1 = "tf.BroadcastGradientArgs"(%s0, %s1) {} : (tensor<1xi32>, tensor<2xi32>) -> (tensor<1xi32>, tensor<0xi32>)
  // CHECK-DAG: %[[R0:.*]] = "tf.Const"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
  // CHECK-DAG: %[[R1:.*]] = "tf.Const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
  // CHECK-NOT: tf.BroadcastGradientArgs
  // CHECK: return %[[R0]], %[[R1]]

  func.return %r0, %r1 : tensor<1xi32>, tensor<0xi32>
}

// CHECK-LABEL: func @testBroadcastGradientArgs2
func.func @testBroadcastGradientArgs2() -> (tensor<1xi32>, tensor<3xi32>) {
  %s2 = "tf.Const"() {value = dense<[501, 1, 32, 1280]> : tensor<4xi32>} : () -> tensor<4xi32>
  %s3 = "tf.Const"() {value = dense<[  1, 1,  1, 1280]> : tensor<4xi32>} : () -> tensor<4xi32>
  %r2, %r3 = "tf.BroadcastGradientArgs"(%s2, %s3) {} : (tensor<4xi32>, tensor<4xi32>) -> (tensor<1xi32>, tensor<3xi32>)
  // CHECK-DAG: %[[R2:.*]] = "tf.Const"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
  // CHECK-DAG: %[[R3:.*]] = "tf.Const"() <{value = dense<[0, 1, 2]> : tensor<3xi32>}> : () -> tensor<3xi32>
  // CHECK-NOT: tf.BroadcastGradientArgs
  // CHECK: return %[[R2]], %[[R3]]

  func.return %r2, %r3 : tensor<1xi32>, tensor<3xi32>
}

// CHECK-LABEL: func @testBroadcastGradientArgs3
func.func @testBroadcastGradientArgs3() -> (tensor<3xi32>, tensor<3xi32>) {
  %s4 = "tf.Const"() {value = dense<[]> : tensor<0xi32>} : () -> tensor<0xi32>
  %s5 = "tf.Const"() {value = dense<[1, 1, 1]> : tensor<3xi32>} : () -> tensor<3xi32>
  %r4, %r5 = "tf.BroadcastGradientArgs"(%s4, %s5) {} : (tensor<0xi32>, tensor<3xi32>) -> (tensor<3xi32>, tensor<3xi32>)
  // CHECK: %[[R0:.*]] = "tf.Const"() <{value = dense<[0, 1, 2]> : tensor<3xi32>}> : () -> tensor<3xi32>
  // CHECK-NOT: tf.BroadcastGradientArgs
  // CHECK: return %[[R0]], %[[R0]]

  func.return %r4, %r5 : tensor<3xi32>, tensor<3xi32>
}

// CHECK-LABEL: func @testBroadcastGradientArgs4
func.func @testBroadcastGradientArgs4() -> (tensor<2xi32>, tensor<3xi32>) {
  %s4 = "tf.Const"() {value = dense<[1, 2, 1]> : tensor<3xi32>} : () -> tensor<3xi32>
  %s5 = "tf.Const"() {value = dense<[]> : tensor<0xi32>} : () -> tensor<0xi32>
  %r4, %r5 = "tf.BroadcastGradientArgs"(%s4, %s5) {} : (tensor<3xi32>, tensor<0xi32>) -> (tensor<2xi32>, tensor<3xi32>)
  // CHECK-DAG: %[[R0:.*]] = "tf.Const"() <{value = dense<[0, 2]> : tensor<2xi32>}> : () -> tensor<2xi32>
  // CHECK-DAG: %[[R1:.*]] = "tf.Const"() <{value = dense<[0, 1, 2]> : tensor<3xi32>}> : () -> tensor<3xi32>
  // CHECK-NOT: tf.BroadcastGradientArgs
  // CHECK: return %[[R0]], %[[R1]]

  func.return %r4, %r5 : tensor<2xi32>, tensor<3xi32>
}

// CHECK-LABEL: func @testBroadcastGradientArgs5
func.func @testBroadcastGradientArgs5() -> (tensor<1xi32>, tensor<1xi32>) {
  %s4 = "tf.Const"() {value = dense<[]> : tensor<0xi32>} : () -> tensor<0xi32>
  %s5 = "tf.Const"() {value = dense<[1]> : tensor<1xi32>} : () -> tensor<1xi32>
  %r4, %r5 = "tf.BroadcastGradientArgs"(%s4, %s5) {} : (tensor<0xi32>, tensor<1xi32>) -> (tensor<1xi32>, tensor<1xi32>)
  // CHECK: %[[R0:.*]] = "tf.Const"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
  // CHECK-NOT: tf.BroadcastGradientArgs
  // CHECK: return %[[R0]], %[[R0]]

  func.return %r4, %r5 : tensor<1xi32>, tensor<1xi32>
}

// CHECK-LABEL: func @testBroadcastGradientArgs6
func.func @testBroadcastGradientArgs6() -> (tensor<1xi32>, tensor<0xi32>) {
  %s4 = "tf.Const"() {value = dense<[]> : tensor<0xi32>} : () -> tensor<0xi32>
  %s5 = "tf.Const"() {value = dense<[2]> : tensor<1xi32>} : () -> tensor<1xi32>
  %r4, %r5 = "tf.BroadcastGradientArgs"(%s4, %s5) {} : (tensor<0xi32>, tensor<1xi32>) -> (tensor<1xi32>, tensor<0xi32>)
  // CHECK-DAG: %[[R0:.*]] = "tf.Const"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
  // CHECK-DAG: %[[R1:.*]] = "tf.Const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
  // CHECK-NOT: tf.BroadcastGradientArgs
  // CHECK: return %[[R0]], %[[R1]]

  func.return %r4, %r5 : tensor<1xi32>, tensor<0xi32>
}

// CHECK-LABEL: func @testBroadcastGradientArgsHigherRank
func.func @testBroadcastGradientArgsHigherRank() -> (tensor<2xi32>, tensor<2xi32>) {
  %s0 = "tf.Const"() {value = dense<[1, 4, 1]> : tensor<3xi32>} : () -> tensor<3xi32>
  %s1 = "tf.Const"() {value = dense<[1, 4]> : tensor<2xi32>} : () -> tensor<2xi32>
  %r0, %r1 = "tf.BroadcastGradientArgs"(%s0, %s1) {} : (tensor<3xi32>, tensor<2xi32>) -> (tensor<2xi32>, tensor<2xi32>)
  // CHECK-DAG: %[[R0:.*]] = "tf.Const"() <{value = dense<[0, 2]> : tensor<2xi32>}> : () -> tensor<2xi32>
  // CHECK-DAG: %[[R1:.*]] = "tf.Const"() <{value = dense<[0, 1]> : tensor<2xi32>}> : () -> tensor<2xi32>
  // CHECK-NOT: tf.BroadcastGradientArgs
  // CHECK: return %[[R0]], %[[R1]]

  func.return %r0, %r1 : tensor<2xi32>, tensor<2xi32>
}

// CHECK-LABEL: func @testBroadcastGradientArgsScalar
func.func @testBroadcastGradientArgsScalar() -> (tensor<2xi32>, tensor<0xi32>) {
  %s0 = "tf.Const"() {value = dense<> : tensor<0xi32>} : () -> tensor<0xi32>
  %s1 = "tf.Const"() {value = dense<[2, 4]> : tensor<2xi32>} : () -> tensor<2xi32>
  %r0, %r1 = "tf.BroadcastGradientArgs"(%s0, %s1) {} : (tensor<0xi32>, tensor<2xi32>) -> (tensor<2xi32>, tensor<0xi32>)
  // CHECK-DAG: %[[R0:.*]] = "tf.Const"() <{value = dense<[0, 1]> : tensor<2xi32>}> : () -> tensor<2xi32>
  // CHECK-DAG: %[[R1:.*]] = "tf.Const"() <{value = dense<> : tensor<0xi32>}> : () -> tensor<0xi32>
  // CHECK-NOT: tf.BroadcastGradientArgs
  // CHECK: return %[[R0]], %[[R1]]

  func.return %r0, %r1 : tensor<2xi32>, tensor<0xi32>
}

// CHECK-LABEL: func @testBroadcastGradientArgI64
func.func @testBroadcastGradientArgI64() -> (tensor<2xi64>, tensor<0xi64>) {
  %s0 = "tf.Const"() {value = dense<> : tensor<0xi64>} : () -> tensor<0xi64>
  %s1 = "tf.Const"() {value = dense<[2, 4]> : tensor<2xi64>} : () -> tensor<2xi64>
  %r0, %r1 = "tf.BroadcastGradientArgs"(%s0, %s1) {} : (tensor<0xi64>, tensor<2xi64>) -> (tensor<2xi64>, tensor<0xi64>)
  // CHECK-DAG: %[[R0:.*]] = "tf.Const"() <{value = dense<[0, 1]> : tensor<2xi64>}> : () -> tensor<2xi64>
  // CHECK-DAG: %[[R1:.*]] = "tf.Const"() <{value = dense<> : tensor<0xi64>}> : () -> tensor<0xi64>
  // CHECK-NOT: tf.BroadcastGradientArgs
  // CHECK: return %[[R0]], %[[R1]]

  func.return %r0, %r1 : tensor<2xi64>, tensor<0xi64>
}

// CHECK-LABEL: func @testEmptyResults
func.func @testEmptyResults(%arg0: tensor<0x2xf32>) -> tensor<0x2xf32> {
  %indices = "tf.Const"() {value = dense<> : tensor<0xi32>} : () -> tensor<0xi32>

  // CHECK: "tf.Const"() <{value = dense<> : tensor<0x2xf32>}> : () -> tensor<0x2xf32>
  %0 = "tf.DynamicStitch"(%indices, %arg0) : (tensor<0xi32>, tensor<0x2xf32>) -> tensor<0x2xf32>
  func.return %0 : tensor<0x2xf32>
}

// Verifies that tf.Yield op which has no result and is not side effecting is
// preserved.
//
// CHECK-LABEL: func @yieldOp
func.func @yieldOp(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<i1>) -> (tensor<f32>) {
  // CHECK-COUNT-2: tf.Yield
  %0 = "tf.IfRegion"(%arg2) ({
      "tf.Yield"(%arg0) : (tensor<f32>) -> ()
    }, {
      "tf.Yield"(%arg1) : (tensor<f32>) -> ()
    }) { is_stateless = true}: (tensor<i1>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// CHECK-LABEL: @range_int
func.func @range_int() -> tensor<?xi32> {
  %cst = arith.constant dense<0> : tensor<i32>
  %cst_1 = arith.constant dense<4> : tensor<i32>
  %cst_2 = arith.constant dense<1> : tensor<i32>

  // CHECK: %[[CST:.*]] = "tf.Const"() <{value = dense<[0, 1, 2, 3]> : tensor<4xi32>}> : () -> tensor<?xi32>
  // CHECK: return %[[CST]]
  %0 = "tf.Range"(%cst, %cst_1, %cst_2) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<?xi32>
  func.return %0 : tensor<?xi32>
}

// CHECK-LABEL: @range_uint
func.func @range_uint() -> tensor<?xui32> {
  %cst = arith.constant dense<0> : tensor<ui32>
  %cst_1 = arith.constant dense<4> : tensor<ui32>
  %cst_2 = arith.constant dense<1> : tensor<ui32>

  // CHECK: %[[CST:.*]] = "tf.Const"() <{value = dense<[0, 1, 2, 3]> : tensor<4xui32>}> : () -> tensor<?xui32>
  // CHECK: return %[[CST]]
  %0 = "tf.Range"(%cst, %cst_1, %cst_2) : (tensor<ui32>, tensor<ui32>, tensor<ui32>) -> tensor<?xui32>
  func.return %0 : tensor<?xui32>
}

// CHECK-LABEL: @range_float
func.func @range_float() -> tensor<?xf32> {
  %cst = arith.constant dense<0.0> : tensor<f32>
  %cst_1 = arith.constant dense<4.0> : tensor<f32>
  %cst_2 = arith.constant dense<1.0> : tensor<f32>

  // CHECK: %[[CST:.*]] = "tf.Const"() <{value = dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<4xf32>}> : () -> tensor<?xf32>
  // CHECK: return %[[CST]]
  %0 = "tf.Range"(%cst, %cst_1, %cst_2) : (tensor<f32>, tensor<f32>, tensor<f32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// CHECK-LABEL: func @testLogicalAndFoldsWithConstantFalse
func.func @testLogicalAndFoldsWithConstantFalse(%arg0: tensor<i1>) -> (tensor<i1>) {
  // CHECK: [[CST:%.+]] = "tf.Const"() <{value = dense<false> : tensor<i1>}> : () -> tensor<i1>
  %cst = arith.constant dense<false> : tensor<i1>

  %0 = "tf.LogicalAnd"(%cst, %arg0) : (tensor<i1>, tensor<i1>) -> tensor<i1>

  // CHECK: return [[CST]]
  func.return %0: tensor<i1>
}

// CHECK-LABEL: func @testLogicalAndFoldsWithConstantFalseSecondArg
func.func @testLogicalAndFoldsWithConstantFalseSecondArg(%arg0: tensor<i1>) -> (tensor<i1>) {
  // CHECK: [[CST:%.+]] = "tf.Const"() <{value = dense<false> : tensor<i1>}> : () -> tensor<i1>
  %cst = arith.constant dense<false> : tensor<i1>

  %0 = "tf.LogicalAnd"(%arg0, %cst) : (tensor<i1>, tensor<i1>) -> tensor<i1>

  // CHECK: return [[CST]]
  func.return %0: tensor<i1>
}

// CHECK-LABEL: func @testLogicalAndNoFoldWithConstTrue
func.func @testLogicalAndNoFoldWithConstTrue(%arg0: tensor<i1>) -> (tensor<i1>) {
  %cst = arith.constant dense<true> : tensor<i1>

  // CHECK: %[[LOGICAL_AND:.*]] = "tf.LogicalAnd"
  %0 = "tf.LogicalAnd"(%cst, %arg0) : (tensor<i1>, tensor<i1>) -> tensor<i1>

  // CHECK: return %[[LOGICAL_AND]]
  func.return %0 : tensor<i1>
}

// CHECK-LABEL: func @testLogicalAndDoesntFoldWithConstantFalseBroadcast
func.func @testLogicalAndDoesntFoldWithConstantFalseBroadcast(%arg0: tensor<2xi1>) -> (tensor<2xi1>) {
  %cst = arith.constant dense<false> : tensor<i1>

  // CHECK: %[[LOGICAL_AND:.*]] = "tf.LogicalAnd"
  %0 = "tf.LogicalAnd"(%cst, %arg0) : (tensor<i1>, tensor<2xi1>) -> tensor<2xi1>

  // CHECK: return %[[LOGICAL_AND]]
  func.return %0: tensor<2xi1>
}

// -----

// GlobalIterId should not be constant folded
// CHECK-LABEL: func @testGlobalIterIdNotFolded
func.func @testGlobalIterIdNotFolded() -> (tensor<i64>) {
  // CHECK: %[[X:.*]] = "tf.GlobalIterId"
  %0 = "tf.GlobalIterId"() : () -> tensor<i64>
  // CHECK: return %[[X]]
  func.return %0: tensor<i64>
}

// -----

// CHECK-LABEL: func @testReadFileOpNotFolded
func.func @testReadFileOpNotFolded() -> (tensor<!tf_type.string>) {
  %0 = "tf.Const"() { value = dense<"filepath"> : tensor<!tf_type.string> } : () -> tensor<!tf_type.string>
  // CHECK: %[[X:.*]] = "tf.ReadFile"
  %1 = "tf.ReadFile"(%0) : (tensor<!tf_type.string>) -> tensor<!tf_type.string>
  // CHECK: return %[[X]]
  func.return %1: tensor<!tf_type.string>
}
