// RUN: tf-opt %s -tfl-post-quantize | FileCheck %s
// RUN: tf-opt %s -tfl-post-quantize-remove-qdq | FileCheck --check-prefix=QDQ %s

// CHECK-LABEL: RemoveUnused
// QDQ-LABEL: RemoveUnused
func.func @RemoveUnused(%arg0: tensor<4xf32>, %arg1: tensor<i32>) -> (tensor<2xf32>,tensor<2xf32>) {
  %0 = "tfl.quantize"(%arg0) {qtype = tensor<4x!quant.uniform<u8:f32, 1.0>>} : (tensor<4xf32>) -> tensor<4x!quant.uniform<u8:f32, 1.0>>
  %1:4 = "tfl.split"(%arg1, %0) {num_splits = 4 : i32} : (tensor<i32>, tensor<4x!quant.uniform<u8:f32, 1.0>>)
  -> (tensor<2x!quant.uniform<u8:f32, 1.0>>, tensor<2x!quant.uniform<u8:f32, 1.0>>,tensor<2x!quant.uniform<u8:f32, 1.0>>, tensor<2x!quant.uniform<u8:f32, 1.0>>)
  %2 = "tfl.dequantize"(%1#0) : (tensor<2x!quant.uniform<u8:f32, 1.0>>) -> tensor<2xf32>
  %3 = "tfl.dequantize"(%1#1) : (tensor<2x!quant.uniform<u8:f32, 1.0>>) -> tensor<2xf32>

  // unused quantization ops should be removed as well.
  %4 = "tfl.dequantize"(%1#2) : (tensor<2x!quant.uniform<u8:f32, 1.0>>) -> tensor<2xf32>
  %5 = "tfl.quantize"(%4) {qtype = tensor<2x!quant.uniform<u8:f32, 1.0>>} : (tensor<2xf32>) -> (tensor<2x!quant.uniform<u8:f32, 1.0>>)
  %6 = tfl.add %5, %5 {fused_activation_function = "NONE"} : tensor<2x!quant.uniform<u8:f32, 1.0>>

  func.return %2, %3 : tensor<2xf32>, tensor<2xf32>

// CHECK-NEXT: %[[split:.*]]:4 = "tfl.split"(%arg1, %arg0)
// CHECK-NEXT: return %[[split]]#0, %[[split]]#1

// QDQ-NEXT: %[[q:.*]] = "tfl.quantize"(%arg0) <{qtype = tensor<4x!quant.uniform<u8:f32, 1.000000e+00>>}> : (tensor<4xf32>) -> tensor<4x!quant.uniform<u8:f32, 1.000000e+00>>
// QDQ-NEXT: %[[split:.*]]:4 = "tfl.split"(%arg1, %[[q]]) <{num_splits = 4 : i32}> : (tensor<i32>, tensor<4x!quant.uniform<u8:f32, 1.000000e+00>>) -> (tensor<2x!quant.uniform<u8:f32, 1.000000e+00>>, tensor<2x!quant.uniform<u8:f32, 1.000000e+00>>, tensor<2x!quant.uniform<u8:f32, 1.000000e+00>>, tensor<2x!quant.uniform<u8:f32, 1.000000e+00>>)
// QDQ-NEXT: %[[out1:.*]] = "tfl.dequantize"(%[[split]]#0) : (tensor<2x!quant.uniform<u8:f32, 1.000000e+00>>) -> tensor<2xf32>
// QDQ-NEXT: %[[out2:.*]] = "tfl.dequantize"(%[[split]]#1) : (tensor<2x!quant.uniform<u8:f32, 1.000000e+00>>) -> tensor<2xf32>
// QDQ-NEXT: return %[[out1]], %[[out2]] : tensor<2xf32>, tensor<2xf32>
}

// CHECK-LABEL: RemoveTrival
// QDQ-LABEL: RemoveTrival
func.func @RemoveTrival(%arg0: tensor<384x512x!quant.uniform<i8:f32, 1.0:-128>>, %arg1: tensor<128x512x!quant.uniform<i8<-127:127>:f32, 1.0>>, %arg2: none) -> tensor<384x128x!quant.uniform<i8:f32, 2.0>> {
  %1 = "tfl.fully_connected"(%arg0, %arg1, %arg2) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<384x512x!quant.uniform<i8:f32, 1.0:-128>>, tensor<128x512x!quant.uniform<i8<-127:127>:f32, 1.0>>, none) -> tensor<384x128x!quant.uniform<i8:f32, 1.0>>
  %2 = "tfl.quantize"(%1) {qtype = tensor<384x128x!quant.uniform<i8:f32, 2.0>>} : (tensor<384x128x!quant.uniform<i8:f32, 1.0>>) -> tensor<384x128x!quant.uniform<i8:f32, 2.0>>
  func.return %2 : tensor<384x128x!quant.uniform<i8:f32, 2.0>>

// CHECK-NEXT: %[[fc:.*]] = "tfl.fully_connected"{{.*}} -> tensor<384x128x!quant.uniform<i8:f32, 2.000000e+00>>
// CHECK-NEXT: return %[[fc]]

// QDQ-NEXT: %[[fc:.*]] = "tfl.fully_connected"(%arg0, %arg1, %arg2) <{fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"}> : (tensor<384x512x!quant.uniform<i8:f32, 1.000000e+00:-128>>, tensor<128x512x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>, none) -> tensor<384x128x!quant.uniform<i8:f32, 1.000000e+00>>
// QDQ-NEXT: %[[q:.*]] = "tfl.quantize"(%[[fc]]) <{qtype = tensor<384x128x!quant.uniform<i8:f32, 2.000000e+00>>}> : (tensor<384x128x!quant.uniform<i8:f32, 1.000000e+00>>) -> tensor<384x128x!quant.uniform<i8:f32, 2.000000e+00>>
// QDQ-NEXT: return %[[q]] : tensor<384x128x!quant.uniform<i8:f32, 2.000000e+00>>
}

func.func @main(%arg0: tensor<1x224x224x3xf32>) -> tensor<1x401408xf32> {
  %cst = arith.constant dense<[1, 401408]> : tensor<2xi32>
  %0 = "tfl.quantize"(%arg0) {qtype = tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>} : (tensor<1x224x224x3xf32>) -> tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>
  %1 = "tfl.pseudo_qconst"() {qtype = tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>, value = dense<-76> : tensor<32x3x3x3xi8>} : () -> tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>
  %2 = "tfl.pseudo_qconst"() {qtype = tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>, value = dense<0> : tensor<32xi32>} : () -> tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>
  %3 = "tfl.conv_2d"(%0, %1, %2) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>, tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>, tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>) -> tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>
  %4 = "tfl.reshape"(%3, %cst) : (tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>, tensor<2xi32>) -> tensor<1x401408x!quant.uniform<u8:f32, 0.023528476789885875>>
  %5 = "tfl.softmax"(%4) {beta = 1.000000e+00 : f32} : (tensor<1x401408x!quant.uniform<u8:f32, 0.023528476789885875>>) -> tensor<1x401408x!quant.uniform<u8:f32, 3.906250e-03>>
  %6 = "tfl.dequantize"(%5) : (tensor<1x401408x!quant.uniform<u8:f32, 3.906250e-03>>) -> tensor<1x401408xf32>
  func.return %6 : tensor<1x401408xf32>
}

func.func @main2(%arg0: tensor<2x4xf32>, %arg1: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %0 = "tfl.quantize"(%arg0) {qtype = tensor<2x4x!quant.uniform<u8:f32, 0.49803921568627452>>} : (tensor<2x4xf32>) -> tensor<2x4x!quant.uniform<u8:f32, 0.49803921568627452>>
  %1 = "tfl.quantize"(%arg1) {qtype = tensor<2x4x!quant.uniform<u8:f32, 0.49803921568627452>>} : (tensor<2x4xf32>) -> tensor<2x4x!quant.uniform<u8:f32, 0.49803921568627452>>
  %2 = tfl.add %0, %1 {fused_activation_function = "NONE"} : tensor<2x4x!quant.uniform<u8:f32, 0.49803921568627452>>
  %3 = "tfl.dequantize"(%2) : (tensor<2x4x!quant.uniform<u8:f32, 0.49803921568627452>>) -> tensor<2x4xf32>
  func.return %3 : tensor<2x4xf32>
}

// CHECK: func @main(%arg0: tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>)
// CHECK-NEXT:  %[[cst:.*]] = arith.constant dense<[1, 401408]> : tensor<2xi32>
// CHECK-NEXT:  %[[q_cst_0:.*]] = "tfl.pseudo_qconst"() <{qtype = tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>, value = dense<-76> : tensor<32x3x3x3xi8>}>
// CHECK-NEXT:  %[[q_cst_1:.*]] = "tfl.pseudo_qconst"() <{qtype = tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>, value = dense<0> : tensor<32xi32>}>
// CHECK-NEXT:  %[[conv:.*]] = "tfl.conv_2d"(%arg0, %[[q_cst_0]], %[[q_cst_1]]) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32}>
// CHECK-NEXT:  %[[reshape:.*]] = "tfl.reshape"(%[[conv]], %[[cst]]) : (tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>, tensor<2xi32>)
// CHECK-NEXT:  %[[softmax:.*]] = "tfl.softmax"(%[[reshape]]) <{beta = 1.000000e+00 : f32}> : (tensor<1x401408x!quant.uniform<u8:f32, 0.023528476789885875>>)
// CHECK-NEXT:  return %[[softmax]] : tensor<1x401408x!quant.uniform<u8:f32, 3.906250e-03>>
// CHECK-NEXT:}

// CHECK: func @main2(%arg0: tensor<2x4x!quant.uniform<u8:f32, 0.49803921568627452>>, %arg1: tensor<2x4x!quant.uniform<u8:f32, 0.49803921568627452>>) -> tensor<2x4x!quant.uniform<u8:f32, 0.49803921568627452>> {
// CHECK-NEXT:  %[[add:.*]] = tfl.add %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<2x4x!quant.uniform<u8:f32, 0.49803921568627452>>
// CHECK-NEXT:  return %[[add]] : tensor<2x4x!quant.uniform<u8:f32, 0.49803921568627452>>
// CHECK-NEXT:}

// CHECK-LABEL: HandleReturnedDequantizeWithAnotherUse
func.func @HandleReturnedDequantizeWithAnotherUse(%arg0: tensor<128x16xf32>) -> (tensor<128x16xf32>, tensor<128xi32>) {
// CHECK-NEXT:  %[[cst:.*]] = arith.constant dense<1> : tensor<i32>
  %cst = arith.constant dense<1> : tensor<i32>
// CHECK-NEXT:  %[[softmax:.*]] = "tfl.softmax"(%arg0) <{beta = 1.000000e+00 : f32}> : (tensor<128x16xf32>) -> tensor<128x16xf32>
  %0 = "tfl.softmax"(%arg0) {beta = 1.000000e+00 : f32} : (tensor<128x16xf32>) -> tensor<128x16xf32>
  %1 = "tfl.quantize"(%0) {qtype = tensor<128x16x!quant.uniform<u8:f32, 3.906250e-03>>, volatile} : (tensor<128x16xf32>) -> tensor<128x16x!quant.uniform<u8:f32, 3.906250e-03>>
  %2 = "tfl.dequantize"(%1) : (tensor<128x16x!quant.uniform<u8:f32, 3.906250e-03>>) -> tensor<128x16xf32>
// CHECK-NEXT:  %[[argmax:.*]] = "tfl.arg_max"(%[[softmax]], %[[cst]]) : (tensor<128x16xf32>, tensor<i32>) -> tensor<128xi32>
  %3 = "tfl.arg_max"(%2, %cst) : (tensor<128x16xf32>, tensor<i32>) -> tensor<128xi32>
// CHECK-NEXT:  return %[[softmax]], %[[argmax]] : tensor<128x16xf32>, tensor<128xi32>
  func.return %2, %3 : tensor<128x16xf32>, tensor<128xi32>
}

// CHECK-LABEL: PruneUnusedLstm
func.func @PruneUnusedLstm(%arg0: tensor<1x28x28xf32>) -> (tensor<1x28x28xf32>) {
    %input = "tfl.quantize"(%arg0) {qtype = tensor<1x28x28x!quant.uniform<i8:f32, 0.003:-128>>} : (tensor<1x28x28xf32>) -> tensor<1x28x28x!quant.uniform<i8:f32, 0.003:-128>>
    %cst_1 = "tfl.pseudo_qconst"() {qtype = tensor<1x20x!quant.uniform<i8:f32, 0.006:-34>>, value = dense<1> : tensor<1x20xi8>} : () -> tensor<1x20x!quant.uniform<i8:f32, 0.006:-34>>
    %cst_2 = "tfl.no_value"() {value = unit} : () -> none
    %cst_3 = "tfl.pseudo_qconst"() {qtype = tensor<20x20x!quant.uniform<i8:f32, 0.006:-34>>, value = dense<1> : tensor<20x20xi8>} : () -> tensor<20x20x!quant.uniform<i8:f32, 0.006:-34>>
    %cst_7 = "tfl.pseudo_qconst"() {qtype = tensor<20x!quant.uniform<i8:f32, 0.006:-34>>, value = dense<1> : tensor<20xi8>} : () -> tensor<20x!quant.uniform<i8:f32, 0.006:-34>>
    %cst_11 = "tfl.pseudo_qconst"() {qtype = tensor<20x28x!quant.uniform<i8:f32, 0.006:-34>>, value = dense<1> : tensor<20x28xi8>} : () -> tensor<20x28x!quant.uniform<i8:f32, 0.006:-34>>
    %cell_input = "tfl.pseudo_qconst"() {qtype = tensor<1x20x!quant.uniform<i16:f32, 0.006:-34>>, value = dense<1> : tensor<1x20xi6>} : () -> tensor<1x20x!quant.uniform<i16:f32, 0.006:-34>>
    %0 = "tfl.unidirectional_sequence_lstm"(%input,
      %cst_11, %cst_11, %cst_11, %cst_11,
      %cst_3, %cst_3, %cst_3, %cst_3,
      %cst_2, %cst_2, %cst_2,
      %cst_7, %cst_7, %cst_7, %cst_7,
      %cst_2, %cst_2,
      %cst_1, %cell_input,
      %cst_2, %cst_2, %cst_2, %cst_2) {cell_clip = 1.000000e+01 : f32, fused_activation_function = "TANH", proj_clip = 0.000000e+00 : f32, time_major = false}
    :  ( tensor<1x28x28x!quant.uniform<i8:f32, 0.003:-128>>,
         tensor<20x28x!quant.uniform<i8:f32, 0.006:-34>>, tensor<20x28x!quant.uniform<i8:f32, 0.006:-34>>, tensor<20x28x!quant.uniform<i8:f32, 0.006:-34>>, tensor<20x28x!quant.uniform<i8:f32, 0.006:-34>>,
         tensor<20x20x!quant.uniform<i8:f32, 0.006:-34>>, tensor<20x20x!quant.uniform<i8:f32, 0.006:-34>>, tensor<20x20x!quant.uniform<i8:f32, 0.006:-34>>, tensor<20x20x!quant.uniform<i8:f32, 0.006:-34>>,
         none, none, none,
         tensor<20x!quant.uniform<i8:f32, 0.006:-34>>, tensor<20x!quant.uniform<i8:f32, 0.006:-34>>, tensor<20x!quant.uniform<i8:f32, 0.006:-34>>, tensor<20x!quant.uniform<i8:f32, 0.006:-34>>,
         none, none,
         tensor<1x20x!quant.uniform<i8:f32, 0.006:-34>>, tensor<1x20x!quant.uniform<i16:f32, 0.006:-34>>,
         none, none, none, none) -> tensor<1x28x20x!quant.uniform<i8:f32, 0.006:-34>>
    func.return %arg0 : tensor<1x28x28xf32>
// CHECK-NEXT: return %arg0
}

// CHECK-LABEL: HandleVolatileRequantizeOp
func.func @HandleVolatileRequantizeOp(%arg0: tensor<1x3x3xf32>) -> (tensor<1x3x3xf32>) {
  %0 = "tfl.quantize"(%arg0) {qtype = tensor<1x3x3x!quant.uniform<i8:f32, 0.003:-128>>} : (tensor<1x3x3xf32>) -> tensor<1x3x3x!quant.uniform<i8:f32, 0.003:-128>>
  %1 = "tfl.logistic"(%0) : (tensor<1x3x3x!quant.uniform<i8:f32, 0.003:-128>>) -> tensor<1x3x3x!quant.uniform<i8:f32, 3.906250e-03:-128>>
  %2 = "tfl.quantize"(%1) {qtype = tensor<1x3x3x!quant.uniform<i8:f32, 0.004:-128>>, volatile} : (tensor<1x3x3x!quant.uniform<i8:f32, 3.906250e-03:-128>>) -> tensor<1x3x3x!quant.uniform<i8:f32, 0.004:-128>>
  %3 = "tfl.dequantize"(%2) : (tensor<1x3x3x!quant.uniform<i8:f32, 0.004:-128>>) -> tensor<1x3x3xf32>
  %4 = "tfl.div"(%arg0, %3) {fused_activation_function = "NONE"} : (tensor<1x3x3xf32>, tensor<1x3x3xf32>) -> tensor<1x3x3xf32>
  func.return %4 : tensor<1x3x3xf32>
//  CHECK: %[[logistic:.*]] = "tfl.logistic"
//  CHECK: %[[dq:.*]] = "tfl.dequantize"(%[[logistic]])
//  CHECK: %[[div:.*]] = tfl.div %arg0, %[[dq]]
}

// CHECK-LABEL: RemoveLeadingQdq
// QDQ-LABEL: RemoveLeadingQdq
func.func @RemoveLeadingQdq(%arg0: tensor<4xf32>, %arg1: tensor<i32>) -> (tensor<2xf32>) {
  %0 = "tfl.quantize"(%arg0) {qtype = tensor<4x!quant.uniform<u8:f32, 1.0>>, volatile} : (tensor<4xf32>) -> tensor<4x!quant.uniform<u8:f32, 1.0>>
  %1 = "tfl.dequantize"(%0) : (tensor<4x!quant.uniform<u8:f32, 1.0>>) -> tensor<4xf32>
  %2:4 = "tfl.split"(%arg1, %1) {num_splits = 4 : i32} : (tensor<i32>, tensor<4xf32>)
  -> (tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>)
  %3 = "tfl.quantize"(%2#0) {qtype = tensor<2x!quant.uniform<u8:f32, 1.0>>, volatile} : (tensor<2xf32>) -> tensor<2x!quant.uniform<u8:f32, 1.0>>
  %4 = "tfl.dequantize"(%3) : (tensor<2x!quant.uniform<u8:f32, 1.0>>) -> tensor<2xf32>
  func.return %4 : tensor<2xf32>

// CHECK-NEXT:  %[[dequant:.*]] = "tfl.dequantize"(%arg0) : (tensor<4x!quant.uniform<u8:f32, 1.000000e+00>>) -> tensor<4xf32>
// CHECK-NEXT:  %[[split:.*]]:4 = "tfl.split"(%arg1, %[[dequant]]) <{num_splits = 4 : i32}> : (tensor<i32>, tensor<4xf32>) -> (tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>)
// CHECK-NEXT:  %[[quant:.*]] = "tfl.quantize"(%[[split]]#0) <{qtype = tensor<2x!quant.uniform<u8:f32, 1.000000e+00>>}> {volatile} : (tensor<2xf32>) -> tensor<2x!quant.uniform<u8:f32, 1.000000e+00>>
// CHECK-NEXT:  return %[[quant]] : tensor<2x!quant.uniform<u8:f32, 1.000000e+00>>

// QDQ-NEXT:  %[[split:.*]]:4 = "tfl.split"(%arg1, %arg0) <{num_splits = 4 : i32}> : (tensor<i32>, tensor<4xf32>) -> (tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>)
// QDQ-NEXT:  return %[[split]]#0 : tensor<2xf32>
}

// CHECK-LABEL: FoldTranspose
func.func @FoldTranspose(%arg0: tensor<1x10x20x3xf32>) -> tensor<1x20x40x16xf32> {
  %cst = arith.constant dense<[1, 20, 40, 16]> : tensor<4xi32>
  %cst_0 = arith.constant dense<[2, 0, 1, 3]> : tensor<4xi32>
  %0 = "tfl.pseudo_qconst"() {qtype = tensor<16x!quant.uniform<i32:f32, 1.8527095877721169E-10>>, value = dense<0> : tensor<16xi32>} : () -> tensor<16x!quant.uniform<i32:f32, 1.8527095877721169E-10>>
  %1 = "tfl.pseudo_qconst"() {qtype = tensor<3x3x16x3x!quant.uniform<i8<-127:127>:f32, 0.047244094488188976>>, value = dense<"0x0303040002010303FFFFFD0304020401FF0000FEFF0003FF01FD0203FF0202FEFE0003010201FD04FE0402030303000202FD0100FDFE0402FEFEFE01020101FD0204FEFDFC03FFFE0101FDFE02040002FDFFFE03FFFE0201FEFDFF00FFFDFEFD030201FD01FC01FF010003FF0401FCFD0101FC0000FE03FEFE010102000002FE02030100FE00FEFDFD0003FD000303000103FE01FF02000002FF0101FDFDFF02FFFF00000203FF0003030302FDFF03FFFF030001020102FD04FE0104FE030401030102FEFCFEFD03FD03FD000102FE02020001020000FE030202030103FFFC01FC000302000304FCFF03FD04FC00010400010100030303FC02FCFEFE01000303000100010003FE000303010301010102FEFC01FD020301FFFDFFFCFDFEFCFE030001FDFCFE000202FE020300FD00FD02FF0001FF0002FF01FD010102FDFE04FCFE0000FD01000101FF0402FF020103FC020301FF03010204FDFFFE0202FF0302FF02FFFF01FF01FF04FD0002FF00FC00FC0101010404FE03040300000301FD0001FE04FF040103FF01FD0301FF0002040403FF03FE04FDFD0103FCFE01FDFCFF03FC010200FDFE020200FF00FFFC03FE"> : tensor<3x3x16x3xi8>} : () -> tensor<3x3x16x3x!quant.uniform<i8<-127:127>:f32, 0.047244094488188976>>
  %2 = "tfl.quantize"(%arg0) {qtype = tensor<1x10x20x3x!quant.uniform<i8:f32, 3.9215686274509805E-9:-1>>} : (tensor<1x10x20x3xf32>) -> tensor<1x10x20x3x!quant.uniform<i8:f32, 3.9215686274509805E-9:-1>>
  %3 = "tfl.transpose"(%1, %cst_0) : (tensor<3x3x16x3x!quant.uniform<i8<-127:127>:f32, 0.047244094488188976>>, tensor<4xi32>) -> tensor<16x3x3x3x!quant.uniform<i8<-127:127>:f32, 0.047244094488188976>>
  %4 = "tfl.transpose_conv"(%cst, %3, %2, %0) {padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32, fused_activation_function = "NONE"} : (tensor<4xi32>, tensor<16x3x3x3x!quant.uniform<i8<-127:127>:f32, 0.047244094488188976>>, tensor<1x10x20x3x!quant.uniform<i8:f32, 3.9215686274509805E-9:-1>>, tensor<16x!quant.uniform<i32:f32, 1.8527095877721169E-10>>) -> tensor<1x20x40x16x!quant.uniform<i8:f32, 0.047058823529411764>>
  %5 = "tfl.dequantize"(%4) : (tensor<1x20x40x16x!quant.uniform<i8:f32, 0.047058823529411764>>) -> tensor<1x20x40x16xf32>
  return %5 : tensor<1x20x40x16xf32>

  // CHECK-NOT: "tfl.transpose"
  // CHECK: "tfl.pseudo_qconst"() <{qtype = tensor<16x3x3x3x!quant.uniform<i8<-127:127>:f32, 0.047244094488188976>>, value = dense<"0x03030402FD010302010103FE0301020001010001FD02030101FE0400020100FDFEFD01FC01FF02FEFCFE000303FCFE00FF0301FF04010303FF0402FE01FF01000002FD03FD03FC020202FE0204FD03FF01FFFD03FEFE010003FFFF010103FD00FCFEFE020300FFFE02FD03010402040201010401FCFDFDFF0102FE010003FD00FD02FF03FF000201FF00FD0204FD010102FFFF02020003000102FF0002FF0204040300FEFFFEFDFCFC000000000201020000010001FF00FFFF01FF03FE0003FF03FFFEFE03FE03FF0000FE0303FE0002FF01FF01FF04FDFD01FD020101FDFE0101030303020203030301FD010104FD000103FC03FF02FE020402000002FDFF0103FF03010102FDFE02FF00FE01FD02FEFE0002FD02FE0203FFFFFC01FC0102FE04FCFEFC00FCFCFF03000301FFFE03030100030001000302FC01FD0000FD010101FC01020201FDFFFE02FE00FE0201020003040203010100010404FE00FDFE04FE0401FEFDFDFD00FD04FEFCFF03FFFDFF01FF04030403020200020303FF00FF03FD000104FEFD04FCFCFDFE02FF02000003FF00FF030002FDFEFD030300030401000104FCFE030103FC01FD00FC03FE"> : tensor<16x3x3x3xi8>}> : () -> tensor<16x3x3x3x!quant.uniform<i8<-127:127>:f32, 0.047244094488188976>>
  // CHECK-NEXT: "tfl.transpose_conv"
}

// CHECK-LABEL: FoldReshape
func.func @FoldReshape(%arg0: tensor<4xi32>, %arg1: tensor<1x48x80x16x!quant.uniform<i8:f32, 0.047054948993757659:-128>>, %arg2: tensor<1x!quant.uniform<i32:f32, 0.0010538385465422978>>) -> tensor<1x96x160x1x!quant.uniform<i8:f32, 0.37102097156001074:-14>> {
  %cst = arith.constant dense<[1, 2, 2, 16]> : tensor<4xi32>
  %0 = "tfl.pseudo_qconst"() {qtype = tensor<2x2x1x16x!quant.uniform<i8<-127:127>:f32, 0.022395913056501255>>, value = dense<[[[[12, -60, -51, -59, -62, 33, 53, 17, -31, 50, 27, 7, -19, -34, -14, -26]], [[47, -84, -32, -36, -102, -8, -8, 35, -33, 59, 95, 40, -25, -30, -55, 25]]], [[[4, -41, -61, 12, -23, 48, 40, 15, -39, 52, 81, -62, -24, 17, -7, -52]], [[40, -70, -45, 32, -43, 2, -30, 34, -35, 58, 77, -28, -30, 37, -47, -5]]]]> : tensor<2x2x1x16xi8>} : () -> tensor<2x2x1x16x!quant.uniform<i8<-127:127>:f32, 0.022395913056501255>>
  %1 = "tfl.reshape"(%0, %cst) : (tensor<2x2x1x16x!quant.uniform<i8<-127:127>:f32, 0.022395913056501255>>, tensor<4xi32>) -> tensor<1x2x2x16x!quant.uniform<i8<-127:127>:f32, 0.022395913056501255>>
  %2 = "tfl.transpose_conv"(%arg0, %1, %arg1, %arg2) {padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32, fused_activation_function = "NONE"} : (tensor<4xi32>, tensor<1x2x2x16x!quant.uniform<i8<-127:127>:f32, 0.022395913056501255>>, tensor<1x48x80x16x!quant.uniform<i8:f32, 0.047054948993757659:-128>>, tensor<1x!quant.uniform<i32:f32, 0.0010538385465422978>>) -> tensor<1x96x160x1x!quant.uniform<i8:f32, 0.37102097156001074:-14>>
  return %2 : tensor<1x96x160x1x!quant.uniform<i8:f32, 0.37102097156001074:-14>>
  // CHECK-NOT: "tfl.reshape"
  // CHECK{LITERAL}: "tfl.pseudo_qconst"() <{qtype = tensor<1x2x2x16x!quant.uniform<i8<-127:127>:f32, 0.022395913056501255>>, value = dense<[[[[12, -60, -51, -59, -62, 33, 53, 17, -31, 50, 27, 7, -19, -34, -14, -26], [47, -84, -32, -36, -102, -8, -8, 35, -33, 59, 95, 40, -25, -30, -55, 25]], [[4, -41, -61, 12, -23, 48, 40, 15, -39, 52, 81, -62, -24, 17, -7, -52], [40, -70, -45, 32, -43, 2, -30, 34, -35, 58, 77, -28, -30, 37, -47, -5]]]]> : tensor<1x2x2x16xi8>}> : () -> tensor<1x2x2x16x!quant.uniform<i8<-127:127>:f32, 0.022395913056501255>>
  // CHECK-NEXT: "tfl.transpose_conv"
}

// CHECK-LABEL: @FoldPerAxisReshape
func.func @FoldPerAxisReshape() -> tensor<1x2x2x!quant.uniform<i8:f32:2, {0.007,0.004}>> {
  %cst = arith.constant dense<[1, 2, 2]> : tensor<3xi32>
  %0 = "tfl.pseudo_qconst"() <{qtype = tensor<2x2x!quant.uniform<i8:f32:1, {0.007,0.004}>>, value = dense<[[-127, 127], [-85, -80]]> : tensor<2x2xi8>}> : () -> tensor<2x2x!quant.uniform<i8:f32:1, {0.007,0.004}>>
  %1 = "tfl.reshape"(%0, %cst) : (tensor<2x2x!quant.uniform<i8:f32:1, {0.007,0.004}>>, tensor<3xi32>) -> tensor<1x2x2x!quant.uniform<i8:f32:2, {0.007,0.004}>>
  return %1 : tensor<1x2x2x!quant.uniform<i8:f32:2, {0.007,0.004}>>


// CHECK{LITERAL}:  %0 = "tfl.pseudo_qconst"() <{qtype = tensor<1x2x2x!quant.uniform<i8:f32:2, {7.000000e-03,4.000000e-03}>>, value = dense<[[[-127, 127], [-85, -80]]]> : tensor<1x2x2xi8>}> : () -> tensor<1x2x2x!quant.uniform<i8:f32:2, {7.000000e-03,4.000000e-03}>>
// CHECK-NOT: tfl.reshape
// CHECK:  return %0 : tensor<1x2x2x!quant.uniform<i8:f32:2, {7.000000e-03,4.000000e-03}>>
}

// CHECK-LABEL: RemoveVolatileQConstOps
func.func @RemoveVolatileQConstOps() -> tensor<640xf32> {
  %1 = "tfl.pseudo_qconst"() <{qtype = tensor<640x!quant.uniform<i32:f32, 1.0000000949949049E-6>>, value = dense<0> : tensor<640xi32>}> {volatile} : () -> tensor<640x!quant.uniform<i32:f32, 1.0000000949949049E-6>>
  %2 = "tfl.dequantize"(%1) : (tensor<640x!quant.uniform<i32:f32, 1.0000000949949049E-6>>) -> tensor<640xf32>
  func.return %2 : tensor<640xf32>
  // CHECK: %0 = "tfl.pseudo_qconst"() <{qtype = tensor<640x!quant.uniform<i32:f32, 1.0000000949949049E-6>>, value = dense<0> : tensor<640xi32>}> {volatile} : () -> tensor<640x!quant.uniform<i32:f32, 1.0000000949949049E-6>>
  // CHECK: return %0 : tensor<640x!quant.uniform<i32:f32, 1.0000000949949049E-6>>

  // QDQ-CHECK: %cst = arith.constant dense<0.000000e+00> : tensor<640xf32>
  // QDQ-CHECK: return %cst : tensor<640xf32>
}
