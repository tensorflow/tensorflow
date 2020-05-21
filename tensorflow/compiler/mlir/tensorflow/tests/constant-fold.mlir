// RUN: tf-opt %s -canonicalize | FileCheck %s

// CHECK-LABEL: func @testShape
func @testShape(tensor<f32>, tensor<1x32x32x16xf32>, tensor<*xf32>) -> (tensor<0xi32>, tensor<?xi32>, tensor<?xi32>) {
^bb0(%arg0: tensor<f32>, %arg1: tensor<1x32x32x16xf32>, %arg2: tensor<*xf32>):

  // CHECK: tf.Const{{.*}} dense<[]> : tensor<0xi32>
  %0 = "tf.Shape"(%arg0) {T = "tfdtype$DT_FLOAT", output = "tfdtype$DT_INT32"} : (tensor<f32>) -> tensor<0xi32>

  // Result shape need not be static. Folding harness uses TensorFlow constant
  // in that case.
  // CHECK: "tf.Const"() {value = dense<[1, 32, 32, 16]> : tensor<4xi32>} : () -> tensor<?xi32>
  %1 = "tf.Shape"(%arg1) {T = "tfdtype$DT_FLOAT", output = "tfdtype$DT_INT32"} : (tensor<1x32x32x16xf32>) -> tensor<?xi32>

  // CHECK: "tf.Shape"(%arg2) {T = "tfdtype$DT_FLOAT", output = "tfdtype$DT_INT32"} : (tensor<*xf32>) -> tensor<?xi32>
  %2 = "tf.Shape"(%arg2) {T = "tfdtype$DT_FLOAT", output = "tfdtype$DT_INT32"} : (tensor<*xf32>) -> tensor<?xi32>

  return %0, %1, %2 : tensor<0xi32>, tensor<?xi32>, tensor<?xi32>
}

// CHECK-LABEL: func @testPow
// CHECK-SAME:(%[[ARG_0:.*]]: tensor<4xf32>, %[[ARG_1:.*]]: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>)
func @testPow(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) {

  %cst_zero = constant dense<0.0> : tensor<f32>
  %cst_one = constant dense<1.0> : tensor<f32>

  // CHECK-DAG: %[[RES_NO_FOLD:.*]] = "tf.Pow"(%arg0, %arg1)
  %0 = "tf.Pow"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

  // CHECK-DAG: %[[POW_ZERO:.*]] = "tf.Const"() {value = dense<1.000000e+00> : tensor<4xf32>} : () -> tensor<4xf32>
  %1 = "tf.Pow"(%arg0, %cst_zero) : (tensor<4xf32>, tensor<f32>) -> tensor<4xf32>

  // CHECK-NOT: "tf.Pow"
  %2 = "tf.Pow"(%arg0, %cst_one) : (tensor<4xf32>, tensor<f32>) -> tensor<4xf32>

  // CHECK: return %[[RES_NO_FOLD]], %[[POW_ZERO]], %[[ARG_0]]
  return %0, %1, %2 : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
}

// CHECK-LABEL: func @testEmpty32
func @testEmpty32() -> (tensor<5xi32>) {
  %0 = "tf.Const"() { value = dense<5> : tensor<i32> } : () -> tensor<i32>

  // CHECK: [[VAL:%.+]] = "tf.Const"() {value = dense<0> : tensor<5xi32>}
  // CHECK: return [[VAL]]
  %1 = "tf.Empty"(%0) : (tensor<i32>) -> (tensor<5xi32>)
  return %1 : tensor<5xi32>
}

// CHECK-LABEL: func @testEmpty64
func @testEmpty64() -> (tensor<5xi64>) {
  %0 = "tf.Const"() { value = dense<5> : tensor<i32> } : () -> tensor<i32>

  // CHECK: [[VAL:%.+]] = "tf.Const"() {value = dense<0> : tensor<5xi64>}
  // CHECK: return [[VAL]] : tensor<5xi64>
  %1 = "tf.Empty"(%0) : (tensor<i32>) -> (tensor<5xi64>)
  return %1 : tensor<5xi64>
}

// CHECK-LABEL: func @testEmptyFloat
func @testEmptyFloat() -> (tensor<5xf64>) {
  %0 = "tf.Const"() { value = dense<5> : tensor<i32> } : () -> tensor<i32>

  // CHECK: [[VAL:%.+]] = "tf.Const"() {value =  dense<0.000000e+00> : tensor<5xf64>}
  // CHECK: return [[VAL]]
  %1 = "tf.Empty"(%0) : (tensor<i32>) -> (tensor<5xf64>)
  return %1 : tensor<5xf64>
}

// CHECK-LABEL: func @testEmptyf16
func @testEmptyf16() -> (tensor<5xf16>) {
  %0 = "tf.Const"() { value = dense<5> : tensor<i32> } : () -> tensor<i32>

  // CHECK: [[VAL:%.+]] = "tf.Const"() {value =  dense<0.000000e+00> : tensor<5xf16>}
  // CHECK: return [[VAL]]
  %1 = "tf.Empty"(%0) : (tensor<i32>) -> (tensor<5xf16>)
  return %1 : tensor<5xf16>
}

// CHECK-LABEL: func @testEmptybf16
func @testEmptybf16() -> (tensor<5xbf16>) {
  %0 = "tf.Const"() { value = dense<5> : tensor<i32> } : () -> tensor<i32>

  // CHECK: [[VAL:%.+]] = "tf.Const"() {value =  dense<0.000000e+00> : tensor<5xbf16>}
  // CHECK: return [[VAL]]
  %1 = "tf.Empty"(%0) : (tensor<i32>) -> (tensor<5xbf16>)
  return %1 : tensor<5xbf16>
}

// CHECK-LABEL: func @testShapeN
func @testShapeN(%arg0: tensor<f32>, %arg1: tensor<1x32x32x16xf32>, %arg2: tensor<*xf32>) -> (tensor<0xi64>, tensor<4xi64>, tensor<4xi64>, tensor<?xi64>) {

  // CHECK: "tf.Const"() {value = dense<[]> : tensor<0xi64>
  // CHECK: "tf.Const"() {value = dense<[1, 32, 32, 16]> : tensor<4xi64>}
  %0:2 = "tf.ShapeN"(%arg0, %arg1) : (tensor<f32>, tensor<1x32x32x16xf32>) -> (tensor<0xi64>, tensor<4xi64>)

  // CHECK: tf.ShapeN
  %1:2 = "tf.ShapeN"(%arg1, %arg2) : (tensor<1x32x32x16xf32>, tensor<*xf32>) -> (tensor<4xi64>, tensor<?xi64>)

  return %0#0, %0#1, %1#0, %1#1 : tensor<0xi64>, tensor<4xi64>, tensor<4xi64>, tensor<?xi64>
}

// CHECK-LABEL: func @testLeakyRelu
func @testLeakyRelu(%arg0 : tensor<16xf32>) -> (tensor<16xf32>, tensor<f32>, tensor<f32>, tensor<16xf32>) {
  %pos = constant dense<5.0> : tensor<f32>
  %neg = constant dense<-5.0> : tensor<f32>
  %no = "tf.LeakyRelu"(%arg0) {alpha = 0.2 : f32} : (tensor<16xf32>) -> tensor<16xf32>
  %0 = "tf.LeakyRelu"(%pos) {alpha = 0.3 : f32} : (tensor<f32>) -> tensor<f32>
  %1 = "tf.LeakyRelu"(%neg) {alpha = 0.2 : f32} : (tensor<f32>) -> tensor<f32>
  %2 = "tf.LeakyRelu"(%arg0) {alpha = 3.0 : f32} : (tensor<16xf32>) -> tensor<16xf32>
  // CHECK: [[POS:%.*]] = "tf.Const{{.*}} dense<5.000000e+00> : tensor<f32>
  // CHECK: [[NEG:%.*]] = "tf.Const{{.*}} dense<-1.000000e+00> : tensor<f32>
  // CHECK: [[NC1:%.*]] = "tf.LeakyRelu"(%arg0) {alpha = 2.000000e-01 : f32} : (tensor<16xf32>) -> tensor<16xf32>
  // CHECK: [[NC2:%.*]] = "tf.LeakyRelu"(%arg0) {alpha = 3.000000e+00 : f32} : (tensor<16xf32>) -> tensor<16xf32>
  // CHECK: return [[NC1]], [[POS]], [[NEG]], [[NC2]]
  return %no, %0, %1, %2 : tensor<16xf32>, tensor<f32>, tensor<f32>, tensor<16xf32>
}

// CHECK-LABEL: func @tfConst
func @tfConst() -> (tensor<4xf32>, tensor<1x1x6x2xf32>) {
  %0 = "tf.Const"() {device = "", name = "Const", dtype = "tfdtype$DT_FLOAT", value = opaque<"tf", "0x746674656E736F722464747970653A2044545F464C4F41540A74656E736F725F7368617065207B0A202064696D207B0A2020202073697A653A20340A20207D0A7D0A74656E736F725F636F6E74656E743A20225C3030305C3030305C323430405C3030305C30303020405C3030305C303030205C3330315C3030305C3030305C3230305C323737220A"> : tensor<4xf32>} : () -> tensor<4xf32>
  %21 = "tf.Const"() {device = "", name = "Const_143", dtype = "tfdtype$DT_FLOAT", value = dense<0.24288677062973696> : tensor<1x1x6x2xf32>} : () -> tensor<1x1x6x2xf32>
  // CHECK-DAG: value = opaque<"tf"
  // CHECK-DAG: tf.Const{{.*}} dense<0.242886767> : tensor<1x1x6x2xf32>
  return %0, %21 : tensor<4xf32>, tensor<1x1x6x2xf32>
}

// CHECK-LABEL: func @testAdd() -> tensor<2x2xi32>
func @testAdd() -> tensor<2x2xi32> {
^bb0:
  %0 = constant dense<[[0, 1], [2, 3]]> : tensor<2x2xi32>
  %1 = constant dense<1> : tensor<2xi32>
  %2 = "tf.Add"(%0, %1) {device = "", name = "add"} : (tensor<2x2xi32>, tensor<2xi32>) -> tensor<2x2xi32>
  // CHECK:         [[cst:%.*]] = "tf.Const{{.*}} dense<{{\[\[}}1, 2], {{\[}}3, 4]]> : tensor<2x2xi32>
  // CHECK-NEXT:    return [[cst]] : tensor<2x2xi32>
  return %2: tensor<2x2xi32>
}

// CHECK-LABEL: testSimpleConcatOffset
func @testSimpleConcatOffset() -> (tensor<3xi32>, tensor<3xi32>, tensor<3xi32>) {
  %concat_dim = constant dense<1> : tensor<i32>
  %shape0 = constant dense<[2, 2, 7]> : tensor<3xi32>
  %shape1 = constant dense<[2, 3, 7]> : tensor<3xi32>
  %shape2 = constant dense<[2, 5, 7]> : tensor<3xi32>

  // CHECK: [[OFFSET_0:%.*]] = "tf.Const{{.*}} dense<0> : tensor<3xi32>
  // CHECK: [[OFFSET_1:%.*]] = "tf.Const{{.*}} dense<[0, 2, 0]> : tensor<3xi32>
  // CHECK: [[OFFSET_2:%.*]] = "tf.Const{{.*}} dense<[0, 5, 0]> : tensor<3xi32>

  %offset:3 = "tf.ConcatOffset"(%concat_dim, %shape0, %shape1, %shape2) : (tensor<i32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>) -> (tensor<3xi32>, tensor<3xi32>, tensor<3xi32>)

  // CHECK: return [[OFFSET_0]], [[OFFSET_1]], [[OFFSET_2]]
  return %offset#0, %offset#1, %offset#2: tensor<3xi32>, tensor<3xi32>, tensor<3xi32>
}

// CHECK-LABEL: testConcatOffsetWithZeros
func @testConcatOffsetWithZeros() -> (tensor<3xi32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>) {
  %concat_dim = constant dense<1> : tensor<i32>
  %shape0 = constant dense<0> : tensor<3xi32>
  %shape1 = constant dense<[0, 3, 0]> : tensor<3xi32>
  %shape2 = constant dense<[0, 5, 0]> : tensor<3xi32>
  %shape3 = constant dense<0> : tensor<3xi32>

  // CHECK: [[OFFSET_0:%.*]] = "tf.Const{{.*}} dense<0> : tensor<3xi32>
  // CHECK: [[OFFSET_2:%.*]] = "tf.Const{{.*}} dense<[0, 3, 0]> : tensor<3xi32>
  // CHECK: [[OFFSET_3:%.*]] = "tf.Const{{.*}} dense<[0, 8, 0]> : tensor<3xi32>

  %offset:4 = "tf.ConcatOffset"(%concat_dim, %shape0, %shape1, %shape2, %shape3) : (tensor<i32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>) -> (tensor<3xi32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>)

  // CHECK: return [[OFFSET_0]], [[OFFSET_0]], [[OFFSET_2]], [[OFFSET_3]]
  return %offset#0, %offset#1, %offset#2, %offset#3: tensor<3xi32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>
}

// CHECK-LABEL: testConcatOffsetNegativeConcatDim
func @testConcatOffsetNegativeConcatDim() -> (tensor<3xi32>, tensor<3xi32>, tensor<3xi32>) {
  %concat_dim = constant dense<-1> : tensor<i32>
  %shape0 = constant dense<[2, 8, 3]> : tensor<3xi32>
  %shape1 = constant dense<[2, 8, 5]> : tensor<3xi32>
  %shape2 = constant dense<[2, 8, 7]> : tensor<3xi32>

  // CHECK: [[OFFSET_0:%.*]] = "tf.Const{{.*}} dense<0> : tensor<3xi32>
  // CHECK: [[OFFSET_1:%.*]] = "tf.Const{{.*}} dense<[0, 0, 3]> : tensor<3xi32>
  // CHECK: [[OFFSET_2:%.*]] = "tf.Const{{.*}} dense<[0, 0, 8]> : tensor<3xi32>

  %offset:3 = "tf.ConcatOffset"(%concat_dim, %shape0, %shape1, %shape2) : (tensor<i32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>) -> (tensor<3xi32>, tensor<3xi32>, tensor<3xi32>)

  // CHECK: return [[OFFSET_0]], [[OFFSET_1]], [[OFFSET_2]]
  return %offset#0, %offset#1, %offset#2: tensor<3xi32>, tensor<3xi32>, tensor<3xi32>
}

// CHECK-LABEL: testConcatOffsetNonConstConcatDim
func @testConcatOffsetNonConstConcatDim(%concat_dim: tensor<i32>) -> (tensor<3xi32>, tensor<3xi32>) {
  %shape0 = constant dense<[2, 2, 7]> : tensor<3xi32>
  %shape1 = constant dense<[2, 3, 7]> : tensor<3xi32>

  // CHECK: tf.ConcatOffset
  %offset:2 = "tf.ConcatOffset"(%concat_dim, %shape0, %shape1) : (tensor<i32>, tensor<3xi32>, tensor<3xi32>) -> (tensor<3xi32>, tensor<3xi32>)

  return %offset#0, %offset#1: tensor<3xi32>, tensor<3xi32>
}

// CHECK-LABEL: testConcatOffsetNonConstShape
func @testConcatOffsetNonConstShape(%shape1: tensor<3xi32>) -> (tensor<3xi32>, tensor<3xi32>) {
  %concat_dim = constant dense<1> : tensor<i32>
  %shape0 = constant dense<[2, 2, 7]> : tensor<3xi32>

  // CHECK: tf.ConcatOffset
  %offset:2 = "tf.ConcatOffset"(%concat_dim, %shape0, %shape1) : (tensor<i32>, tensor<3xi32>, tensor<3xi32>) -> (tensor<3xi32>, tensor<3xi32>)

  return %offset#0, %offset#1: tensor<3xi32>, tensor<3xi32>
}

// CHECK-LABEL: testConcatOffsetBadNegativeConcatDim
func @testConcatOffsetBadNegativeConcatDim() -> (tensor<3xi32>, tensor<3xi32>) {
  %concat_dim = constant dense<-4> : tensor<i32>
  %shape0 = constant dense<[2, 2, 7]> : tensor<3xi32>
  %shape1 = constant dense<[2, 3, 7]> : tensor<3xi32>

  // CHECK: tf.ConcatOffset
  %offset:2 = "tf.ConcatOffset"(%concat_dim, %shape0, %shape1) : (tensor<i32>, tensor<3xi32>, tensor<3xi32>) -> (tensor<3xi32>, tensor<3xi32>)

  return %offset#0, %offset#1: tensor<3xi32>, tensor<3xi32>
}

// CHECK-LABEL: testConcatOffsetBadPositiveConcatDim
func @testConcatOffsetBadPositiveConcatDim() -> (tensor<3xi32>, tensor<3xi32>) {
  %concat_dim = constant dense<3> : tensor<i32>
  %shape0 = constant dense<[2, 2, 7]> : tensor<3xi32>
  %shape1 = constant dense<[2, 3, 7]> : tensor<3xi32>

  // CHECK: tf.ConcatOffset
  %offset:2 = "tf.ConcatOffset"(%concat_dim, %shape0, %shape1) : (tensor<i32>, tensor<3xi32>, tensor<3xi32>) -> (tensor<3xi32>, tensor<3xi32>)

  return %offset#0, %offset#1: tensor<3xi32>, tensor<3xi32>
}

// CHECK-LABEL: testConcatOffsetDifferentNonConcatDimElements
func @testConcatOffsetDifferentNonConcatDimElements() -> (tensor<3xi32>, tensor<3xi32>) {
  %concat_dim = constant dense<1> : tensor<i32>
  %shape0 = constant dense<[2, 2, 7]> : tensor<3xi32>
  %shape1 = constant dense<[2, 3, 8]> : tensor<3xi32>

  // CHECK: tf.ConcatOffset
  %offset:2 = "tf.ConcatOffset"(%concat_dim, %shape0, %shape1) : (tensor<i32>, tensor<3xi32>, tensor<3xi32>) -> (tensor<3xi32>, tensor<3xi32>)

  return %offset#0, %offset#1: tensor<3xi32>, tensor<3xi32>
}

// Ops with side effects should not get constant folded.
// CHECK-LABEL: func @testSideEffectOp() -> tensor<3xf32>
func @testSideEffectOp() -> tensor<3xf32> {
  %0 = constant dense<[3]> : tensor<1xi32>
  %1 = "tf.RandomUniform"(%0) {device = "", seed = 3 : i64, seed2 = 5 : i64} : (tensor<1xi32>) -> tensor<3xf32>
  // CHECK: %[[random:.*]] = "tf.RandomUniform"
  // CHECK: return %[[random]]
  return %1: tensor<3xf32>
}

// Ops with unimplemented attributes which couldn't be added to the TFE_Op.
// CHECK-LABEL: func @testUnimplementedOp() -> (tensor<i32>, tensor<i32>)
func @testUnimplementedOp() -> (tensor<i32>, tensor<i32>) {
  %0 = constant dense<1> : tensor<i32>
  %1 = constant dense<2> : tensor<i32>
  %2 = "tf.Maximum"(%0, %1) {_output_shapes = ["tfshape$"]} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %3 = "tf.Minimum"(%0, %1) {random_attr = "hello"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  return %2, %3: tensor<i32>, tensor<i32>

// CHECK-NEXT: %[[CST:.*]] = "tf.Const
// CHECK-NEXT: %[[CST1:.*]] = "tf.Const
// CHECK-NEXT: return %[[CST]], %[[CST1]]
}

// Tests ops that have non-local device assignment but with local device with
// same type (CPU) are correctly evaluated.
// CHECK-LABEL: func @testRemoteDevice() -> tensor<2x2xi32>
func @testRemoteDevice() -> tensor<2x2xi32> {
^bb0:
  %0 = constant dense<[[0, 1], [2, 3]]> : tensor<2x2xi32>
  %1 = constant dense<1> : tensor<2xi32>
  %2 = "tf.Add"(%0, %1) {device = "/job:remote_worker/replica:123/task:456/CPU:0", name = "add"} : (tensor<2x2xi32>, tensor<2xi32>) -> tensor<2x2xi32>
  // CHECK:         [[cst:%.*]] = "tf.Const{{.*}} dense<{{\[\[}}1, 2], {{\[}}3, 4]]> : tensor<2x2xi32>
  // CHECK-NEXT:    return [[cst]] : tensor<2x2xi32>
  return %2: tensor<2x2xi32>
}

// Tests ops that variable shapes are correctly evaluated on static types.
// CHECK-LABEL: func @testVariableShape
func @testVariableShape(%arg0: tensor<!tf.resource<tensor<2x4xf32>>>) -> tensor<2xi32> {
  %0 = "tf.VariableShape"(%arg0) : (tensor<!tf.resource<tensor<2x4xf32>>>) -> tensor<2xi32>
  // CHECK:         [[cst:%.*]] = "tf.Const{{.*}} dense<{{\[}}2, 4]> : tensor<2xi32>
  // CHECK-NEXT:    return [[cst]] : tensor<2xi32>
  return %0: tensor<2xi32>
}

// Tests ops that tensor list shapes are correctly evaluated on static types.
// CHECK-LABEL: func @testTensorListElementShape
func @testTensorListElementShape(%arg0: tensor<!tf.variant<tensor<2x4xf32>>>) -> tensor<2xi32> {
  %0 = "tf.TensorListElementShape"(%arg0) : (tensor<!tf.variant<tensor<2x4xf32>>>) -> tensor<2xi32>
  // CHECK:         [[cst:%.*]] = "tf.Const{{.*}} dense<{{\[}}2, 4]> : tensor<2xi32>
  // CHECK-NEXT:    return [[cst]] : tensor<2xi32>
  return %0: tensor<2xi32>
}

func @RemoveTrivialAdd(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %cst = constant dense<0.0> : tensor<2x2xf32>
  %0 = "tf.Add"(%arg0, %cst) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>

  // CHECK-LABEL: RemoveTrivialAdd
  // CHECK-NEXT: return %arg0 : tensor<2x2xf32>
}

func @RemoveTrivialAddBf16RHS(%arg0: tensor<2x2xbf16>) -> tensor<2x2xbf16> {
  %cst = constant dense<0.0> : tensor<2x2xbf16>
  %0 = "tf.Add"(%arg0, %cst) : (tensor<2x2xbf16>, tensor<2x2xbf16>) -> tensor<2x2xbf16>
  return %0 : tensor<2x2xbf16>

  // CHECK-LABEL: RemoveTrivialAdd
  // CHECK-NEXT: return %arg0 : tensor<2x2xbf16>
}

func @RemoveTrivialAddBf16LHS(%arg0: tensor<2x2xbf16>) -> tensor<2x2xbf16> {
  %cst = constant dense<0.0> : tensor<2x2xbf16>
  %0 = "tf.Add"(%cst, %arg0) : (tensor<2x2xbf16>, tensor<2x2xbf16>) -> tensor<2x2xbf16>
  return %0 : tensor<2x2xbf16>

  // CHECK-LABEL: RemoveTrivialAdd
  // CHECK-NEXT: return %arg0 : tensor<2x2xbf16>
}

func @RemoveTrivialAddV2(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %cst = constant dense<0.0> : tensor<2x2xf32>
  %0 = "tf.AddV2"(%arg0, %cst) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>

  // CHECK-LABEL: RemoveTrivialAddV2
  // CHECK-NEXT: return %arg0 : tensor<2x2xf32>
}

func @RemoveTrivialSub(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %cst = constant dense<0.0> : tensor<2x2xf32>
  %0 = "tf.Sub"(%arg0, %cst) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>

  // CHECK-LABEL: RemoveTrivialSub
  // CHECK-NEXT: return %arg0 : tensor<2x2xf32>
}

func @RemoveTrivialSubInt8(%arg0: tensor<2x2xi8>) -> tensor<2x2xi8> {
  %cst = constant dense<0> : tensor<2x2xi8>
  %0 = "tf.Sub"(%arg0, %cst) : (tensor<2x2xi8>, tensor<2x2xi8>) -> tensor<2x2xi8>
  return %0 : tensor<2x2xi8>

  // CHECK-LABEL: RemoveTrivialSubInt8
  // CHECK-NEXT: return %arg0 : tensor<2x2xi8>
}

func @RemoveTrivialMul(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %cst = constant dense<1.0> : tensor<2x2xf32>
  %0 = "tf.Mul"(%arg0, %cst) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>

  // CHECK-LABEL: RemoveTrivialMul
  // CHECK-NEXT: return %arg0 : tensor<2x2xf32>
}

func @RemoveTrivialDiv(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %cst = constant dense<1.0> : tensor<2x2xf32>
  %0 = "tf.Div"(%arg0, %cst) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>

  // CHECK-LABEL: RemoveTrivialDiv
  // CHECK-NEXT: return %arg0 : tensor<2x2xf32>
}

func @RemoveTrivialRealDiv(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %cst = constant dense<1.0> : tensor<2x2xf32>
  %0 = "tf.RealDiv"(%arg0, %cst) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>

  // CHECK-LABEL: RemoveTrivialRealDiv
  // CHECK-NEXT: return %arg0 : tensor<2x2xf32>
}

func @RemoveTrivialDivBf16RHS(%arg0: tensor<2x2xbf16>) -> tensor<2x2xbf16> {
  %cst = constant dense<1.0> : tensor<2x2xbf16>
  %0 = "tf.Div"(%arg0, %cst) : (tensor<2x2xbf16>, tensor<2x2xbf16>) -> tensor<2x2xbf16>
  return %0 : tensor<2x2xbf16>

  // CHECK-LABEL: RemoveTrivialDiv
  // CHECK-NEXT: return %arg0 : tensor<2x2xbf16>
}

func @RemoveTrivialMulInt8(%arg0: tensor<2x2xi8>) -> tensor<2x2xi8> {
  %cst = constant dense<1> : tensor<2x2xi8>
  %0 = "tf.Mul"(%cst, %arg0) : (tensor<2x2xi8>, tensor<2x2xi8>) -> tensor<2x2xi8>
  return %0 : tensor<2x2xi8>

  // CHECK-LABEL: RemoveTrivialMulInt8
  // CHECK-NEXT: return %arg0 : tensor<2x2xi8>
}

func @DivBf16LHS(%arg0: tensor<2x2xbf16>) -> tensor<2x2xbf16> {
  %cst = constant dense<1.0> : tensor<2x2xbf16>
  %0 = "tf.Div"(%cst, %arg0) : (tensor<2x2xbf16>, tensor<2x2xbf16>) -> tensor<2x2xbf16>
  return %0 : tensor<2x2xbf16>

  // CHECK-LABEL: DivBf16LHS
  // CHECK: tf.Div
}

func @DontRemoveTrivialAdd(%arg0: tensor<1x2xf32>) -> tensor<2x2xf32> {
  %cst = constant dense<0.0> : tensor<2x2xf32>
  %0 = "tf.AddV2"(%arg0, %cst) : (tensor<1x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>

  // CHECK-LABEL: DontRemoveTrivialAdd
  // CHECK: %[[CONST:.*]] = constant dense<0.000000e+00> : tensor<2x2xf32>
  // CHECK: %[[RESULT:.*]] = "tf.AddV2"(%arg0, %[[CONST]]) : (tensor<1x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: return %[[RESULT]] : tensor<2x2xf32>
}

func @DontRemoveTrivialAdd2(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %cst = constant dense<0.0> : tensor<2x2xf32>
  %0 = "tf.AddV2"(%arg0, %cst) : (tensor<?x?xf32> , tensor<2x2xf32>) -> tensor<?x?xf32>
  return %0 :tensor<?x?xf32>

  // CHECK-LABEL: DontRemoveTrivialAdd2
  // CHECK: %[[CONST:.*]] = constant dense<0.000000e+00> : tensor<2x2xf32>
  // CHECK: %[[RESULT:.*]] = "tf.AddV2"(%arg0, %[[CONST]]) : (tensor<?x?xf32>, tensor<2x2xf32>) -> tensor<?x?xf32>
  // CHECK: return %[[RESULT]] : tensor<?x?xf32>
}

// Test no fold because of the broadcast.
func @DontRemoveTrivialMul(%arg0: tensor<1x6x8x1xf32>) -> tensor<1x6x8x1xf32> {
  %0 = "tf.Const"() {value = dense<2.000000e+00> : tensor<f32>} : () -> tensor<f32>
  %1 = "tf.Mul"(%arg0, %0) : (tensor<1x6x8x1xf32>, tensor<f32>) -> tensor<1x6x8x1xf32>
  return %1 : tensor<1x6x8x1xf32>
  // CHECK-LABEL: DontRemoveTrivialMul
  // CHECK: %[[CONST:.*]] = "tf.Const"() {value = dense<2.000000e+00> : tensor<f32>} : () -> tensor<f32>
  // CHECK: %[[RESULT:.*]] = "tf.Mul"(%arg0, %[[CONST]]) : (tensor<1x6x8x1xf32>, tensor<f32>) -> tensor<1x6x8x1xf32>
  // CHECK: return %[[RESULT]] : tensor<1x6x8x1xf32>
}
