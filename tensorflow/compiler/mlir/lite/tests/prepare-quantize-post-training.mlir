// RUN: tf-opt %s -tfl-prepare-quantize -tfl-test-quantize-signed -tfl-test-post-training-quantize -cse | FileCheck %s
// RUN: tf-opt %s -tfl-prepare-quantize -tfl-test-quantize-signed -tfl-test-post-training-quantize -cse -tfl-test-legacy-float-scale | FileCheck --check-prefix=Legacy %s

// CHECK-LABEL: QuantizeLstmCellInput
func.func @QuantizeLstmCellInput(%arg0: tensor<1x28x28xf32>) -> tensor<1x28x20xf32> {
    %cst_2 = "tfl.no_value"() {value = unit} : () -> none
    %cst_3 = arith.constant dense<1.0> : tensor<20x20xf32>
    %cst_7 = arith.constant dense<1.0> : tensor<20xf32>
    %cst_11 = arith.constant dense<1.0> : tensor<20x28xf32>
    %recurrent_input = arith.constant dense<1.0> : tensor<1x20xf32>
    %recurrent_stats = "quant.stats"(%recurrent_input) {layerStats = dense<[-2.0, 1.0]> : tensor<2xf32>} : (tensor<1x20xf32>) -> tensor<1x20xf32>
    %cell_input = arith.constant dense<1.0> : tensor<1x20xf32>
    %cell_stats = "quant.stats"(%cell_input) {layerStats = dense<[-2.73090601, 7.94872093]> : tensor<2xf32>} : (tensor<1x20xf32>) -> tensor<1x20xf32>
    %0 = "tfl.unidirectional_sequence_lstm"(%arg0,
      %cst_11, %cst_11, %cst_11, %cst_11,
      %cst_3, %cst_3, %cst_3, %cst_3,
      %cst_2, %cst_2, %cst_2,
      %cst_7, %cst_7, %cst_7, %cst_7,
      %cst_2, %cst_2,
      %recurrent_stats, %cell_stats,
      %cst_2, %cst_2, %cst_2, %cst_2) {cell_clip = 1.000000e+01 : f32, fused_activation_function = "TANH", proj_clip = 0.000000e+00 : f32, time_major = false}
    : ( tensor<1x28x28xf32>,
        tensor<20x28xf32>, tensor<20x28xf32>, tensor<20x28xf32>, tensor<20x28xf32>,
        tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>,
        none, none, none,
        tensor<20xf32>, tensor<20xf32>, tensor<20xf32>, tensor<20xf32>,
        none, none,
        tensor<1x20xf32>, tensor<1x20xf32>,
        none, none, none, none) -> tensor<1x28x20xf32>
    %1 = "quant.stats"(%0) {layerStats = dense<[-1.0, 2.0]> : tensor<2xf32>} : (tensor<1x28x20xf32>) -> tensor<1x28x20xf32>
    func.return %1 : tensor<1x28x20xf32>
// CHECK-DAG: %[[none:.*]] = "tfl.no_value"() {value} : () -> none
// CHECK-DAG: %[[cell_input:.*]] = arith.constant dense<1.000000e+00> : tensor<1x20xf32>
// CHECK-DAG: %[[q:.*]] = "tfl.quantize"(%[[cell_input]]) {qtype = tensor<1x20x!quant.uniform<i16:f32, 2.44140625E-4>>} : (tensor<1x20xf32>) -> tensor<1x20x!quant.uniform<i16:f32, 2.44140625E-4>>
// CHECK-DAG: %[[dq:.*]] = "tfl.dequantize"(%[[q]]) : (tensor<1x20x!quant.uniform<i16:f32, 2.44140625E-4>>) -> tensor<1x20xf32>
// Checks if input 19 is correctly passed from a dequantize op.
// CHECK: %[[lstm:.*]] = "tfl.unidirectional_sequence_lstm"(%arg0, {{(%[^%,]+, )+}}%[[dq]], %[[none]], %[[none]], %[[none]], %[[none]])
}

// CHECK-LABEL: QuantizeWithoutNorm
func.func @QuantizeWithoutNorm(%arg0: tensor<1x1x5xf32>) -> tensor<*xf32> attributes {tf.entry_function = {inputs = "input0", outputs = "output24"}} {
  %none = "tfl.no_value"() {value = unit} : () -> none
  %input = "quant.stats"(%arg0) {layerStats = dense<[-1.2, 1.5]> : tensor<2xf32>} : (tensor<1x1x5xf32>) -> tensor<1x1x5xf32>
  %0 = "tfl.pseudo_const"() {value = dense<[[1.31760073, -0.78338623, 0.287265539, -0.383972764, -0.00321021513], [0.104248755, 1.07823908, 0.138089031, 0.76123321, -1.4124943]]> : tensor<2x5xf32>} : () -> tensor<2x5xf32>
  %1 = "tfl.pseudo_const"() {value = dense<[[2.32939887, -0.623641372, -0.0191893689, 0.326861918, 0.734137893], [0.499284297, 1.25277913, 0.60228157, -1.39478016, 0.115529917]]> : tensor<2x5xf32>} : () -> tensor<2x5xf32>
  %2 = "tfl.pseudo_const"() {value = dense<[[0.839470446, 0.564852297, -0.80136007, -0.0372898243, 0.57127893], [-5.516230e-01, -1.082380e+00, 1.41860521, -0.92541927, -1.13971734]]> : tensor<2x5xf32>} : () -> tensor<2x5xf32>
  %3 = "tfl.pseudo_const"() {value = dense<[[-0.440826088, -0.0863231644, -0.707756281, -0.695703208, -1.87899077], [0.16942361, 0.206325337, 1.09067786, -2.18648934, 0.273400396]]> : tensor<2x5xf32>} : () -> tensor<2x5xf32>
  %4 = "tfl.pseudo_const"() {value = dense<[[-1.65420437, 0.19633314, 0.828249216, -0.546153665], [-1.49073172, 1.6467551, 0.904948651, 1.1367631]]> : tensor<2x4xf32>} : () -> tensor<2x4xf32>
  %5 = "tfl.pseudo_const"() {value = dense<[[-0.435141891, -0.940576493, 1.30446923, -1.02953017], [0.684501767, 0.363370508, -2.29151702, 2.41928673]]> : tensor<2x4xf32>} : () -> tensor<2x4xf32>
  %6 = "tfl.pseudo_const"() {value = dense<[[0.270476967, 0.00706229592, 0.489950746, 1.05166924], [1.28193891, 0.273171216, 0.484176666, 1.11504579]]> : tensor<2x4xf32>} : () -> tensor<2x4xf32>
  %7 = "tfl.pseudo_const"() {value = dense<[[-2.36692929, -3.483900e-01, 0.322934568, -1.56939185], [-5.623850e-01, -0.083735466, 1.73820043, 0.218063414]]> : tensor<2x4xf32>} : () -> tensor<2x4xf32>
  %8 = "tfl.pseudo_const"() {value = dense<[1.43194032, -0.553496838]> : tensor<2xf32>} : () -> tensor<2xf32>
  %9 = "tfl.pseudo_const"() {value = dense<[-1.66391921, 1.14934266]> : tensor<2xf32>} : () -> tensor<2xf32>
  %10 = "tfl.pseudo_const"() {value = dense<[-1.59288621, 0.904723584]> : tensor<2xf32>} : () -> tensor<2xf32>
  %11 = "tfl.pseudo_const"() {value = dense<[-0.323118627, 1.77580559]> : tensor<2xf32>} : () -> tensor<2xf32>
  %12 = "tfl.pseudo_const"() {value = dense<[-1.0347594, -1.09994471]> : tensor<2xf32>} : () -> tensor<2xf32>
  %13 = "tfl.pseudo_const"() {value = dense<[-2.03072214, -1.63648951]> : tensor<2xf32>} : () -> tensor<2xf32>
  %14 = "tfl.pseudo_const"() {value = dense<[-1.90073407, -0.286088765]> : tensor<2xf32>} : () -> tensor<2xf32>
  %15 = "tfl.pseudo_const"() {value = dense<[[0.580187321, -1.72028887], [1.48392391, 0.859561979], [0.316514879, 0.81852132], [0.0933789983, 0.58165586]]> : tensor<4x2xf32>} : () -> tensor<4x2xf32>
  %16 = "tfl.pseudo_const"() {value = dense<[-0.0432887711, -0.431485623, -0.307492912, -0.882515907]> : tensor<4xf32>} : () -> tensor<4xf32>
  %recurrent_input = "tfl.pseudo_const"() {value = dense<0.000000e+00> : tensor<1x4xf32>} : () -> tensor<1x4xf32>
  %recurrent_stats = "quant.stats"(%recurrent_input) {layerStats = dense<[-2.0, 1.0]> : tensor<2xf32>} : (tensor<1x4xf32>) -> tensor<1x4xf32>
  %cell_input = "tfl.pseudo_const"() {value = dense<1.000000e+00> : tensor<1x2xf32>} : () -> tensor<1x2xf32>
  %cell_stats = "quant.stats"(%cell_input) {layerStats = dense<[-2.73090601, 7.94872093]> : tensor<2xf32>} : (tensor<1x2xf32>) -> tensor<1x2xf32>
  %19 = "tfl.pseudo_const"() {value = dense<[0.928654432, -0.393729329]> : tensor<2xf32>} : () -> tensor<2xf32>
  %20 = "tfl.pseudo_const"() {value = dense<[-0.76004064, -0.892570137]> : tensor<2xf32>} : () -> tensor<2xf32>
  %21 = "tfl.pseudo_const"() {value = dense<[-0.330534697, -1.68513882]> : tensor<2xf32>} : () -> tensor<2xf32>
  %22 = "tfl.pseudo_const"() {value = dense<[-0.896740913, -0.382640809]> : tensor<2xf32>} : () -> tensor<2xf32>
  %23 = "tfl.unidirectional_sequence_lstm"(%input,
    %0, %1, %2, %3,
    %4, %5, %6, %7,
    %8, %9, %10,
    %11, %12, %13, %14,
    %15, %16,
    %recurrent_stats, %cell_stats,
    %none, %none, %none, %none) {cell_clip = 5.000000e+01 : f32,
      effective_hidden_scale_intermediate = tensor<!quant.calibrated<f32<-5.000000e-01:5.000000e-01>>>,
      fused_activation_function = "TANH",
      proj_clip = 0.000000e+00 : f32, time_major = false} : (
        tensor<1x1x5xf32>,
        tensor<2x5xf32>, tensor<2x5xf32>, tensor<2x5xf32>, tensor<2x5xf32>,
        tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>,
        tensor<2xf32>, tensor<2xf32>, tensor<2xf32>,
        tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>,
        tensor<4x2xf32>, tensor<4xf32>,
        tensor<1x4xf32>, tensor<1x2xf32>,
        none, none, none, none) -> tensor<*xf32>
  %24 = "quant.stats"(%23) {layerStats = dense<[-1.0, 2.0]> : tensor<2xf32>} : (tensor<*xf32>) -> tensor<*xf32>
  func.return %24 : tensor<*xf32>

// CHECK-DAG: %[[input_0:.*]] = "tfl.dequantize"({{.*}}) : (tensor<1x1x5x!quant.uniform<i8:f32, 0.010588235481112611:-15>>) -> tensor<1x1x5xf32>
// CHECK-DAG: %[[input_1:.*]] = "tfl.dequantize"({{.*}}) : (tensor<2x5x!quant.uniform<i8<-127:127>:f32, 0.011122002376346137>>) -> tensor<2x5xf32>
// CHECK-DAG: %[[input_2:.*]] = "tfl.dequantize"({{.*}}) : (tensor<2x5x!quant.uniform<i8<-127:127>:f32, 0.018341723389512912>>) -> tensor<2x5xf32>
// CHECK-DAG: %[[input_3:.*]] = "tfl.dequantize"({{.*}}) : (tensor<2x5x!quant.uniform<i8<-127:127>:f32, 0.011170119751156785>>) -> tensor<2x5xf32>
// CHECK-DAG: %[[input_4:.*]] = "tfl.dequantize"({{.*}}) : (tensor<2x5x!quant.uniform<i8<-127:127>:f32, 0.017216451524749515>>) -> tensor<2x5xf32>
// CHECK-DAG: %[[input_5:.*]] = "tfl.dequantize"({{.*}}) : (tensor<2x4x!quant.uniform<i8<-127:127>:f32, 0.013025231248750461>>) -> tensor<2x4xf32>
// CHECK-DAG: %[[input_6:.*]] = "tfl.dequantize"({{.*}}) : (tensor<2x4x!quant.uniform<i8<-127:127>:f32, 0.019049501794529713>>) -> tensor<2x4xf32>
// CHECK-DAG: %[[input_7:.*]] = "tfl.dequantize"({{.*}}) : (tensor<2x4x!quant.uniform<i8<-127:127>:f32, 0.010094007169167826>>) -> tensor<2x4xf32>
// CHECK-DAG: %[[input_8:.*]] = "tfl.dequantize"({{.*}}) : (tensor<2x4x!quant.uniform<i8<-127:127>:f32, 0.018637238525030179>>) -> tensor<2x4xf32>
// CHECK-DAG: %[[input_9:.*]] = "tfl.dequantize"({{.*}}) : (tensor<2x!quant.uniform<i16<-32767:32767>:f32, 4.3700684138124656E-5>>) -> tensor<2xf32>
// CHECK-DAG: %[[input_10:.*]] = "tfl.dequantize"({{.*}}) : (tensor<2x!quant.uniform<i16<-32767:32767>:f32, 5.0780334190922573E-5>>) -> tensor<2xf32>
// CHECK-DAG: %[[input_11:.*]] = "tfl.dequantize"({{.*}}) : (tensor<2x!quant.uniform<i16<-32767:32767>:f32, 4.8612512878442185E-5>>) -> tensor<2xf32>
// CHECK-DAG: %[[input_12:.*]] = "tfl.dequantize"({{.*}}) : (tensor<2x!quant.uniform<i32:f32, 1.1776238018224695E-4>>) -> tensor<2xf32>
// CHECK-DAG: %[[input_13:.*]] = "tfl.dequantize"({{.*}}) : (tensor<2x!quant.uniform<i32:f32, 1.9420648637759368E-4>>) -> tensor<2xf32>
// CHECK-DAG: %[[input_14:.*]] = "tfl.dequantize"({{.*}}) : (tensor<2x!quant.uniform<i32:f32, 1.1827185827747504E-4>>) -> tensor<2xf32>
// CHECK-DAG: %[[input_15:.*]] = "tfl.dequantize"({{.*}}) : (tensor<2x!quant.uniform<i32:f32, 1.8229184289320815E-4>>) -> tensor<2xf32>
// CHECK-DAG: %[[input_16:.*]] = "tfl.dequantize"({{.*}}) : (tensor<4x2x!quant.uniform<i8<-127:127>:f32, 0.013545581674951268>>) -> tensor<4x2xf32>
// CHECK-DAG: %[[input_17:.*]] = "tfl.dequantize"({{.*}}) : (tensor<4x!quant.uniform<i32:f32, 5.3119928137063791E-5>>) -> tensor<4xf32>
// CHECK-DAG: %[[input_18:.*]] = "tfl.dequantize"({{.*}}) : (tensor<1x4x!quant.uniform<i8:f32, 0.015686274509803921:-1>>) -> tensor<1x4xf32>
// CHECK-DAG: %[[input_19:.*]] = "tfl.dequantize"({{.*}}) : (tensor<1x2x!quant.uniform<i16:f32, 2.44140625E-4>>) -> tensor<1x2xf32>

// CHECK: %[[lstm:.*]] = "tfl.unidirectional_sequence_lstm"(%[[input_0]], %[[input_1]], %[[input_2]], %[[input_3]], %[[input_4]], %[[input_5]], %[[input_6]], %[[input_7]], %[[input_8]],
// CHECK-SAME: %[[input_9]], %[[input_10]], %[[input_11]], %[[input_12]], %[[input_13]], %[[input_14]], %[[input_15]], %[[input_16]], %[[input_17]], %[[input_18]], %[[input_19]]
// CHECK-SAME: effective_hidden_scale_intermediate = tensor<!quant.uniform<i8:f32, 0.0039215686274509803:-1>>

// CHECK: "tfl.quantize"(%[[lstm]]) {qtype = tensor<*x!quant.uniform<i8:f32, 0.015686274509803921:-1>>, volatile}
}

// CHECK-LABEL: QuantizeLstmCifg
func.func @QuantizeLstmCifg(%arg0: tensor<1x5xf32>) -> tensor<*xf32> attributes {tf.entry_function = {inputs = "input0", outputs = "output24"}} {
  %none = "tfl.no_value"() {value = unit} : () -> none
  %input = "quant.stats"(%arg0) {layerStats = dense<[-1.2, 1.5]> : tensor<2xf32>} : (tensor<1x5xf32>) -> tensor<1x5xf32>
  %1 = "tfl.pseudo_const"() {value = dense<[[2.32939887, -0.623641372, -0.0191893689, 0.326861918, 0.734137893], [0.499284297, 1.25277913, 0.60228157, -1.39478016, 0.115529917]]> : tensor<2x5xf32>} : () -> tensor<2x5xf32>
  %2 = "tfl.pseudo_const"() {value = dense<[[0.839470446, 0.564852297, -0.80136007, -0.0372898243, 0.57127893], [-5.516230e-01, -1.082380e+00, 1.41860521, -0.92541927, -1.13971734]]> : tensor<2x5xf32>} : () -> tensor<2x5xf32>
  %3 = "tfl.pseudo_const"() {value = dense<[[-0.440826088, -0.0863231644, -0.707756281, -0.695703208, -1.87899077], [0.16942361, 0.206325337, 1.09067786, -2.18648934, 0.273400396]]> : tensor<2x5xf32>} : () -> tensor<2x5xf32>
  %5 = "tfl.pseudo_const"() {value = dense<[[-0.435141891, -0.940576493, 1.30446923, -1.02953017], [0.684501767, 0.363370508, -2.29151702, 2.41928673]]> : tensor<2x4xf32>} : () -> tensor<2x4xf32>
  %6 = "tfl.pseudo_const"() {value = dense<[[0.270476967, 0.00706229592, 0.489950746, 1.05166924], [1.28193891, 0.273171216, 0.484176666, 1.11504579]]> : tensor<2x4xf32>} : () -> tensor<2x4xf32>
  %7 = "tfl.pseudo_const"() {value = dense<[[-2.36692929, -3.483900e-01, 0.322934568, -1.56939185], [-5.623850e-01, -0.083735466, 1.73820043, 0.218063414]]> : tensor<2x4xf32>} : () -> tensor<2x4xf32>
  %9 = "tfl.pseudo_const"() {value = dense<[-1.66391921, 1.14934266]> : tensor<2xf32>} : () -> tensor<2xf32>
  %10 = "tfl.pseudo_const"() {value = dense<[-1.59288621, 0.904723584]> : tensor<2xf32>} : () -> tensor<2xf32>
  %12 = "tfl.pseudo_const"() {value = dense<[-1.0347594, -1.09994471]> : tensor<2xf32>} : () -> tensor<2xf32>
  %13 = "tfl.pseudo_const"() {value = dense<[-2.03072214, -1.63648951]> : tensor<2xf32>} : () -> tensor<2xf32>
  %14 = "tfl.pseudo_const"() {value = dense<[-1.90073407, -0.286088765]> : tensor<2xf32>} : () -> tensor<2xf32>
  %15 = "tfl.pseudo_const"() {value = dense<[[0.580187321, -1.72028887], [1.48392391, 0.859561979], [0.316514879, 0.81852132], [0.0933789983, 0.58165586]]> : tensor<4x2xf32>} : () -> tensor<4x2xf32>
  %16 = "tfl.pseudo_const"() {value = dense<[-0.0432887711, -0.431485623, -0.307492912, -0.882515907]> : tensor<4xf32>} : () -> tensor<4xf32>
  %recurrent_input = "tfl.pseudo_const"() {value = dense<0.000000e+00> : tensor<1x4xf32>} : () -> tensor<1x4xf32>
  %recurrent_stats = "quant.stats"(%recurrent_input) {layerStats = dense<[-2.0, 1.0]> : tensor<2xf32>} : (tensor<1x4xf32>) -> tensor<1x4xf32>
  %cell_input = "tfl.pseudo_const"() {value = dense<1.000000e+00> : tensor<1x2xf32>} : () -> tensor<1x2xf32>
  %cell_stats = "quant.stats"(%cell_input) {layerStats = dense<[-2.73090601, 7.94872093]> : tensor<2xf32>} : (tensor<1x2xf32>) -> tensor<1x2xf32>
  %20 = "tfl.pseudo_const"() {value = dense<[-0.76004064, -0.892570137]> : tensor<2xf32>} : () -> tensor<2xf32>
  %21 = "tfl.pseudo_const"() {value = dense<[-0.330534697, -1.68513882]> : tensor<2xf32>} : () -> tensor<2xf32>
  %22 = "tfl.pseudo_const"() {value = dense<[-0.896740913, -0.382640809]> : tensor<2xf32>} : () -> tensor<2xf32>
  %23 = "tfl.lstm"(%input,
    %none, %1, %2, %3,
    %none, %5, %6, %7,
    %none, %9, %10,
    %none, %12, %13, %14,
    %15, %16,
    %recurrent_stats, %cell_stats,
    %none, %20, %21, %22) ({}) {
      cell_clip = 5.000000e+01 : f32,
      effective_hidden_scale_intermediate = tensor<!quant.calibrated<f32<-5.000000e-01:5.000000e-01>>>,
      fused_activation_function = "TANH",
      input_to_cell_intermediate = tensor<!quant.calibrated<f32<-4.000000e+00:4.000000e+00>>>,
      input_to_forget_intermediate = tensor<!quant.calibrated<f32<-1.600000e+01:1.600000e+01>>>,
      input_to_output_intermediate = tensor<!quant.calibrated<f32<-1.000000e+00:1.000000e+00>>>,
      proj_clip = 0.000000e+00 : f32,time_major = false} : (
        tensor<1x5xf32>,
        none, tensor<2x5xf32>, tensor<2x5xf32>, tensor<2x5xf32>,
        none, tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>,
        none, tensor<2xf32>, tensor<2xf32>,
        none, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>,
        tensor<4x2xf32>, tensor<4xf32>,
        tensor<1x4xf32>, tensor<1x2xf32>,
        none, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> tensor<*xf32>
  %24 = "quant.stats"(%23) {layerStats = dense<[-1.0, 2.0]> : tensor<2xf32>} : (tensor<*xf32>) -> tensor<*xf32>
  func.return %24 : tensor<*xf32>

// CHECK-DAG: %[[none:.*]] = "tfl.no_value"() {value} : () -> none
// CHECK-DAG: %[[input_0:.*]] = "tfl.dequantize"({{.*}}) : (tensor<1x5x!quant.uniform<i8:f32, 0.010588235481112611:-15>>) -> tensor<1x5xf32>
// CHECK-DAG: %[[input_2:.*]] = "tfl.dequantize"({{.*}}) : (tensor<2x5x!quant.uniform<i8<-127:127>:f32, 0.018341723389512912>>) -> tensor<2x5xf32>
// CHECK-DAG: %[[input_3:.*]] = "tfl.dequantize"({{.*}}) : (tensor<2x5x!quant.uniform<i8<-127:127>:f32, 0.011170119751156785>>) -> tensor<2x5xf32>
// CHECK-DAG: %[[input_4:.*]] = "tfl.dequantize"({{.*}}) : (tensor<2x5x!quant.uniform<i8<-127:127>:f32, 0.017216451524749515>>) -> tensor<2x5xf32>
// CHECK-DAG: %[[input_6:.*]] = "tfl.dequantize"({{.*}}) : (tensor<2x4x!quant.uniform<i8<-127:127>:f32, 0.019049501794529713>>) -> tensor<2x4xf32>
// CHECK-DAG: %[[input_7:.*]] = "tfl.dequantize"({{.*}}) : (tensor<2x4x!quant.uniform<i8<-127:127>:f32, 0.010094007169167826>>) -> tensor<2x4xf32>
// CHECK-DAG: %[[input_8:.*]] = "tfl.dequantize"({{.*}}) : (tensor<2x4x!quant.uniform<i8<-127:127>:f32, 0.018637238525030179>>) -> tensor<2x4xf32>
// CHECK-DAG: %[[input_10:.*]] = "tfl.dequantize"({{.*}}) : (tensor<2x!quant.uniform<i16<-32767:32767>:f32, 5.0780334190922573E-5>>) -> tensor<2xf32>
// CHECK-DAG: %[[input_11:.*]] = "tfl.dequantize"({{.*}}) : (tensor<2x!quant.uniform<i16<-32767:32767>:f32, 4.8612512878442185E-5>>) -> tensor<2xf32>
// CHECK-DAG: %[[input_13:.*]] = "tfl.dequantize"({{.*}}) : (tensor<2x!quant.uniform<i32:f32, 2.6601474818224132E-8>>) -> tensor<2xf32>
// CHECK-DAG: %[[input_14:.*]] = "tfl.dequantize"({{.*}}) : (tensor<2x!quant.uniform<i32:f32, 5.0222583101003261E-8>>) -> tensor<2xf32>
// CHECK-DAG: %[[input_15:.*]] = "tfl.dequantize"({{.*}}) : (tensor<2x!quant.uniform<i32:f32, 2.6725777405118232E-8>>) -> tensor<2xf32>
// CHECK-DAG: %[[input_16:.*]] = "tfl.dequantize"({{.*}}) : (tensor<4x2x!quant.uniform<i8<-127:127>:f32, 0.013545581674951268>>) -> tensor<4x2xf32>
// CHECK-DAG: %[[input_17:.*]] = "tfl.dequantize"({{.*}}) : (tensor<4x!quant.uniform<i32:f32, 5.3119928137063791E-5>>) -> tensor<4xf32>
// CHECK-DAG: %[[input_18:.*]] = "tfl.dequantize"({{.*}}) : (tensor<1x4x!quant.uniform<i8:f32, 0.015686274509803921:-1>>) -> tensor<1x4xf32>
// CHECK-DAG: %[[input_19:.*]] = "tfl.dequantize"({{.*}}) : (tensor<1x2x!quant.uniform<i16:f32, 2.44140625E-4>>) -> tensor<1x2xf32>
// CHECK-DAG: %[[input_21:.*]] = "tfl.dequantize"({{.*}}) : (tensor<2x!quant.uniform<i16<-32767:32767>:f32, 2.7239910213861512E-5>>) -> tensor<2xf32>
// CHECK-DAG: %[[input_22:.*]] = "tfl.dequantize"({{.*}}) : (tensor<2x!quant.uniform<i16<-32767:32767>:f32, 5.1427925095427339E-5>>) -> tensor<2xf32>
// CHECK-DAG: %[[input_23:.*]] = "tfl.dequantize"({{.*}}) : (tensor<2x!quant.uniform<i16<-32767:32767>:f32, 2.736719606284107E-5>>) -> tensor<2xf32>

// CHECK: %[[lstm:.*]] = "tfl.lstm"(%[[input_0]], %[[none]], %[[input_2]], %[[input_3]], %[[input_4]], %[[none]], %[[input_6]], %[[input_7]], %[[input_8]],
// CHECK-SAME: %[[none]], %[[input_10]], %[[input_11]], %[[none]], %[[input_13]], %[[input_14]], %[[input_15]], %[[input_16]], %[[input_17]], %[[input_18]], %[[input_19]],
// CHECK-SAME: %[[none]], %[[input_21]], %[[input_22]], %[[input_23]])
// CHECK-NEXT: effective_hidden_scale_intermediate = tensor<!quant.uniform<i8:f32, 0.0039215686274509803:-1>>
// CHECK-SAME: input_to_cell_intermediate = tensor<!quant.uniform<i16<-32767:32767>:f32, 1.2207403790398877E-4>>
// CHECK-SAME: input_to_forget_intermediate = tensor<!quant.uniform<i16<-32767:32767>:f32, 4.8829615161595508E-4>>
// CHECK-SAME: input_to_output_intermediate = tensor<!quant.uniform<i16<-32767:32767>:f32, 3.0518509475997192E-5>>

// CHECK: "tfl.quantize"(%[[lstm]]) {qtype = tensor<*x!quant.uniform<i8:f32, 0.015686274509803921:-1>>, volatile}
}

// CHECK-LABEL: QuantizeUnidirectionalLstmFull
func.func @QuantizeUnidirectionalLstmFull(%arg0: tensor<1x1x5xf32>) -> tensor<*xf32> attributes {tf.entry_function = {inputs = "input0", outputs = "output24"}} {
  %input = "quant.stats"(%arg0) {layerStats = dense<[-1.2, 1.5]> : tensor<2xf32>} : (tensor<1x1x5xf32>) -> tensor<1x1x5xf32>
  %0 = "tfl.pseudo_const"() {value = dense<[[1.31760073, -0.78338623, 0.287265539, -0.383972764, -0.00321021513], [0.104248755, 1.07823908, 0.138089031, 0.76123321, -1.4124943]]> : tensor<2x5xf32>} : () -> tensor<2x5xf32>
  %1 = "tfl.pseudo_const"() {value = dense<[[2.32939887, -0.623641372, -0.0191893689, 0.326861918, 0.734137893], [0.499284297, 1.25277913, 0.60228157, -1.39478016, 0.115529917]]> : tensor<2x5xf32>} : () -> tensor<2x5xf32>
  %2 = "tfl.pseudo_const"() {value = dense<[[0.839470446, 0.564852297, -0.80136007, -0.0372898243, 0.57127893], [-5.516230e-01, -1.082380e+00, 1.41860521, -0.92541927, -1.13971734]]> : tensor<2x5xf32>} : () -> tensor<2x5xf32>
  %3 = "tfl.pseudo_const"() {value = dense<[[-0.440826088, -0.0863231644, -0.707756281, -0.695703208, -1.87899077], [0.16942361, 0.206325337, 1.09067786, -2.18648934, 0.273400396]]> : tensor<2x5xf32>} : () -> tensor<2x5xf32>
  %4 = "tfl.pseudo_const"() {value = dense<[[-1.65420437, 0.19633314, 0.828249216, -0.546153665], [-1.49073172, 1.6467551, 0.904948651, 1.1367631]]> : tensor<2x4xf32>} : () -> tensor<2x4xf32>
  %5 = "tfl.pseudo_const"() {value = dense<[[-0.435141891, -0.940576493, 1.30446923, -1.02953017], [0.684501767, 0.363370508, -2.29151702, 2.41928673]]> : tensor<2x4xf32>} : () -> tensor<2x4xf32>
  %6 = "tfl.pseudo_const"() {value = dense<[[0.270476967, 0.00706229592, 0.489950746, 1.05166924], [1.28193891, 0.273171216, 0.484176666, 1.11504579]]> : tensor<2x4xf32>} : () -> tensor<2x4xf32>
  %7 = "tfl.pseudo_const"() {value = dense<[[-2.36692929, -3.483900e-01, 0.322934568, -1.56939185], [-5.623850e-01, -0.083735466, 1.73820043, 0.218063414]]> : tensor<2x4xf32>} : () -> tensor<2x4xf32>
  %8 = "tfl.pseudo_const"() {value = dense<[1.43194032, -0.553496838]> : tensor<2xf32>} : () -> tensor<2xf32>
  %9 = "tfl.pseudo_const"() {value = dense<[-1.66391921, 1.14934266]> : tensor<2xf32>} : () -> tensor<2xf32>
  %10 = "tfl.pseudo_const"() {value = dense<[-1.59288621, 0.904723584]> : tensor<2xf32>} : () -> tensor<2xf32>
  %11 = "tfl.pseudo_const"() {value = dense<[-0.323118627, 1.77580559]> : tensor<2xf32>} : () -> tensor<2xf32>
  %12 = "tfl.pseudo_const"() {value = dense<[-1.0347594, -1.09994471]> : tensor<2xf32>} : () -> tensor<2xf32>
  %13 = "tfl.pseudo_const"() {value = dense<[-2.03072214, -1.63648951]> : tensor<2xf32>} : () -> tensor<2xf32>
  %14 = "tfl.pseudo_const"() {value = dense<[-1.90073407, -0.286088765]> : tensor<2xf32>} : () -> tensor<2xf32>
  %15 = "tfl.pseudo_const"() {value = dense<[[0.580187321, -1.72028887], [1.48392391, 0.859561979], [0.316514879, 0.81852132], [0.0933789983, 0.58165586]]> : tensor<4x2xf32>} : () -> tensor<4x2xf32>
  %16 = "tfl.pseudo_const"() {value = dense<[-0.0432887711, -0.431485623, -0.307492912, -0.882515907]> : tensor<4xf32>} : () -> tensor<4xf32>
  %recurrent_input = "tfl.pseudo_const"() {value = dense<0.000000e+00> : tensor<1x4xf32>} : () -> tensor<1x4xf32>
  %recurrent_stats = "quant.stats"(%recurrent_input) {layerStats = dense<[-2.0, 1.0]> : tensor<2xf32>} : (tensor<1x4xf32>) -> tensor<1x4xf32>
  %cell_input = "tfl.pseudo_const"() {value = dense<1.000000e+00> : tensor<1x2xf32>} : () -> tensor<1x2xf32>
  %cell_stats = "quant.stats"(%cell_input) {layerStats = dense<[-2.73090601, 7.94872093]> : tensor<2xf32>} : (tensor<1x2xf32>) -> tensor<1x2xf32>
  %19 = "tfl.pseudo_const"() {value = dense<[0.928654432, -0.393729329]> : tensor<2xf32>} : () -> tensor<2xf32>
  %20 = "tfl.pseudo_const"() {value = dense<[-0.76004064, -0.892570137]> : tensor<2xf32>} : () -> tensor<2xf32>
  %21 = "tfl.pseudo_const"() {value = dense<[-0.330534697, -1.68513882]> : tensor<2xf32>} : () -> tensor<2xf32>
  %22 = "tfl.pseudo_const"() {value = dense<[-0.896740913, -0.382640809]> : tensor<2xf32>} : () -> tensor<2xf32>
  %23 = "tfl.unidirectional_sequence_lstm"(%input,
    %0, %1, %2, %3,
    %4, %5, %6, %7,
    %8, %9, %10,
    %11, %12, %13, %14,
    %15, %16,
    %recurrent_stats, %cell_stats,
    %19, %20, %21, %22) {cell_clip = 5.000000e+01 : f32,
      effective_hidden_scale_intermediate = tensor<!quant.calibrated<f32<-5.000000e-01:5.000000e-01>>>,
      fused_activation_function = "TANH",
      input_to_cell_intermediate = tensor<!quant.calibrated<f32<-4.000000e+00:4.000000e+00>>>,
      input_to_forget_intermediate = tensor<!quant.calibrated<f32<-1.600000e+01:1.600000e+01>>>,
      input_to_input_intermediate = tensor<!quant.calibrated<f32<-3.200000e+01:3.200000e+01>>>,
      input_to_output_intermediate = tensor<!quant.calibrated<f32<-1.000000e+00:1.000000e+00>>>,
      proj_clip = 0.000000e+00 : f32, time_major = false} : (
        tensor<1x1x5xf32>,
        tensor<2x5xf32>, tensor<2x5xf32>, tensor<2x5xf32>, tensor<2x5xf32>,
        tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>,
        tensor<2xf32>, tensor<2xf32>, tensor<2xf32>,
        tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>,
        tensor<4x2xf32>, tensor<4xf32>,
        tensor<1x4xf32>, tensor<1x2xf32>,
        tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> tensor<*xf32>
  %24 = "quant.stats"(%23) {layerStats = dense<[-1.0, 2.0]> : tensor<2xf32>} : (tensor<*xf32>) -> tensor<*xf32>
  func.return %24 : tensor<*xf32>

// CHECK-DAG: %[[input_0:.*]] = "tfl.dequantize"({{.*}}) : (tensor<1x1x5x!quant.uniform<i8:f32, 0.010588235481112611:-15>>) -> tensor<1x1x5xf32>
// CHECK-DAG: %[[input_1:.*]] = "tfl.dequantize"({{.*}}) : (tensor<2x5x!quant.uniform<i8<-127:127>:f32, 0.011122002376346137>>) -> tensor<2x5xf32>
// CHECK-DAG: %[[input_2:.*]] = "tfl.dequantize"({{.*}}) : (tensor<2x5x!quant.uniform<i8<-127:127>:f32, 0.018341723389512912>>) -> tensor<2x5xf32>
// CHECK-DAG: %[[input_3:.*]] = "tfl.dequantize"({{.*}}) : (tensor<2x5x!quant.uniform<i8<-127:127>:f32, 0.011170119751156785>>) -> tensor<2x5xf32>
// CHECK-DAG: %[[input_4:.*]] = "tfl.dequantize"({{.*}}) : (tensor<2x5x!quant.uniform<i8<-127:127>:f32, 0.017216451524749515>>) -> tensor<2x5xf32>
// CHECK-DAG: %[[input_5:.*]] = "tfl.dequantize"({{.*}}) : (tensor<2x4x!quant.uniform<i8<-127:127>:f32, 0.013025231248750461>>) -> tensor<2x4xf32>
// CHECK-DAG: %[[input_6:.*]] = "tfl.dequantize"({{.*}}) : (tensor<2x4x!quant.uniform<i8<-127:127>:f32, 0.019049501794529713>>) -> tensor<2x4xf32>
// CHECK-DAG: %[[input_7:.*]] = "tfl.dequantize"({{.*}}) : (tensor<2x4x!quant.uniform<i8<-127:127>:f32, 0.010094007169167826>>) -> tensor<2x4xf32>
// CHECK-DAG: %[[input_8:.*]] = "tfl.dequantize"({{.*}}) : (tensor<2x4x!quant.uniform<i8<-127:127>:f32, 0.018637238525030179>>) -> tensor<2x4xf32>
// CHECK-DAG: %[[input_9:.*]] = "tfl.dequantize"({{.*}}) : (tensor<2x!quant.uniform<i16<-32767:32767>:f32, 4.3700684138124656E-5>>) -> tensor<2xf32>
// CHECK-DAG: %[[input_10:.*]] = "tfl.dequantize"({{.*}}) : (tensor<2x!quant.uniform<i16<-32767:32767>:f32, 5.0780334190922573E-5>>) -> tensor<2xf32>
// CHECK-DAG: %[[input_11:.*]] = "tfl.dequantize"({{.*}}) : (tensor<2x!quant.uniform<i16<-32767:32767>:f32, 4.8612512878442185E-5>>) -> tensor<2xf32>
// CHECK-DAG: %[[input_12:.*]] = "tfl.dequantize"({{.*}}) : (tensor<2x!quant.uniform<i32:f32, 2.7676903410132078E-8>>) -> tensor<2xf32>
// CHECK-DAG: %[[input_13:.*]] = "tfl.dequantize"({{.*}}) : (tensor<2x!quant.uniform<i32:f32, 2.6601474818224132E-8>>) -> tensor<2xf32>
// CHECK-DAG: %[[input_14:.*]] = "tfl.dequantize"({{.*}}) : (tensor<2x!quant.uniform<i32:f32, 5.0222583101003261E-8>>) -> tensor<2xf32>
// CHECK-DAG: %[[input_15:.*]] = "tfl.dequantize"({{.*}}) : (tensor<2x!quant.uniform<i32:f32, 2.6725777405118232E-8>>) -> tensor<2xf32>
// CHECK-DAG: %[[input_16:.*]] = "tfl.dequantize"({{.*}}) : (tensor<4x2x!quant.uniform<i8<-127:127>:f32, 0.013545581674951268>>) -> tensor<4x2xf32>
// CHECK-DAG: %[[input_17:.*]] = "tfl.dequantize"({{.*}}) : (tensor<4x!quant.uniform<i32:f32, 5.3119928137063791E-5>>) -> tensor<4xf32>
// CHECK-DAG: %[[input_18:.*]] = "tfl.dequantize"({{.*}}) : (tensor<1x4x!quant.uniform<i8:f32, 0.015686274509803921:-1>>) -> tensor<1x4xf32>
// CHECK-DAG: %[[input_19:.*]] = "tfl.dequantize"({{.*}}) : (tensor<1x2x!quant.uniform<i16:f32, 2.44140625E-4>>) -> tensor<1x2xf32>
// CHECK-DAG: %[[input_20:.*]] = "tfl.dequantize"({{.*}}) : (tensor<2x!quant.uniform<i16<-32767:32767>:f32, 2.8341149091975248E-5>>) -> tensor<2xf32>
// CHECK-DAG: %[[input_21:.*]] = "tfl.dequantize"({{.*}}) : (tensor<2x!quant.uniform<i16<-32767:32767>:f32, 2.7239910213861512E-5>>) -> tensor<2xf32>
// CHECK-DAG: %[[input_22:.*]] = "tfl.dequantize"({{.*}}) : (tensor<2x!quant.uniform<i16<-32767:32767>:f32, 5.1427925095427339E-5>>) -> tensor<2xf32>
// CHECK-DAG: %[[input_23:.*]] = "tfl.dequantize"({{.*}}) : (tensor<2x!quant.uniform<i16<-32767:32767>:f32, 2.736719606284107E-5>>) -> tensor<2xf32>

// CHECK: %[[lstm:.*]] = "tfl.unidirectional_sequence_lstm"(%[[input_0]], %[[input_1]], %[[input_2]], %[[input_3]], %[[input_4]], %[[input_5]], %[[input_6]], %[[input_7]], %[[input_8]],
// CHECK-SAME: %[[input_9]], %[[input_10]], %[[input_11]], %[[input_12]], %[[input_13]], %[[input_14]], %[[input_15]], %[[input_16]], %[[input_17]], %[[input_18]], %[[input_19]],
// CHECK-SAME: %[[input_20]], %[[input_21]], %[[input_22]], %[[input_23]])
// CHECK-SAME: effective_hidden_scale_intermediate = tensor<!quant.uniform<i8:f32, 0.0039215686274509803:-1>>
// CHECK-SAME: input_to_cell_intermediate = tensor<!quant.uniform<i16<-32767:32767>:f32, 1.2207403790398877E-4>>
// CHECK-SAME: input_to_forget_intermediate = tensor<!quant.uniform<i16<-32767:32767>:f32, 4.8829615161595508E-4>>
// CHECK-SAME: input_to_input_intermediate = tensor<!quant.uniform<i16<-32767:32767>:f32, 9.7659230323191015E-4>>
// CHECK-SAME: input_to_output_intermediate = tensor<!quant.uniform<i16<-32767:32767>:f32, 3.0518509475997192E-5>>

// CHECK: "tfl.quantize"(%[[lstm]]) {qtype = tensor<*x!quant.uniform<i8:f32, 0.015686274509803921:-1>>, volatile}
}

// CHECK-LABEL: QuantizeUnidirectionalLstmWithFixedOutputRangedInput
func.func @QuantizeUnidirectionalLstmWithFixedOutputRangedInput(%arg0: tensor<1x1x5xf32>) -> tensor<*xf32> attributes {tf.entry_function = {inputs = "input0", outputs = "output24"}} {
  %input = "tfl.logistic"(%arg0) : (tensor<1x1x5xf32>) -> tensor<1x1x5xf32>
  %0 = "tfl.pseudo_const"() {value = dense<[[1.31760073, -0.78338623, 0.287265539, -0.383972764, -0.00321021513], [0.104248755, 1.07823908, 0.138089031, 0.76123321, -1.4124943]]> : tensor<2x5xf32>} : () -> tensor<2x5xf32>
  %1 = "tfl.pseudo_const"() {value = dense<[[2.32939887, -0.623641372, -0.0191893689, 0.326861918, 0.734137893], [0.499284297, 1.25277913, 0.60228157, -1.39478016, 0.115529917]]> : tensor<2x5xf32>} : () -> tensor<2x5xf32>
  %2 = "tfl.pseudo_const"() {value = dense<[[0.839470446, 0.564852297, -0.80136007, -0.0372898243, 0.57127893], [-5.516230e-01, -1.082380e+00, 1.41860521, -0.92541927, -1.13971734]]> : tensor<2x5xf32>} : () -> tensor<2x5xf32>
  %3 = "tfl.pseudo_const"() {value = dense<[[-0.440826088, -0.0863231644, -0.707756281, -0.695703208, -1.87899077], [0.16942361, 0.206325337, 1.09067786, -2.18648934, 0.273400396]]> : tensor<2x5xf32>} : () -> tensor<2x5xf32>
  %4 = "tfl.pseudo_const"() {value = dense<[[-1.65420437, 0.19633314, 0.828249216, -0.546153665], [-1.49073172, 1.6467551, 0.904948651, 1.1367631]]> : tensor<2x4xf32>} : () -> tensor<2x4xf32>
  %5 = "tfl.pseudo_const"() {value = dense<[[-0.435141891, -0.940576493, 1.30446923, -1.02953017], [0.684501767, 0.363370508, -2.29151702, 2.41928673]]> : tensor<2x4xf32>} : () -> tensor<2x4xf32>
  %6 = "tfl.pseudo_const"() {value = dense<[[0.270476967, 0.00706229592, 0.489950746, 1.05166924], [1.28193891, 0.273171216, 0.484176666, 1.11504579]]> : tensor<2x4xf32>} : () -> tensor<2x4xf32>
  %7 = "tfl.pseudo_const"() {value = dense<[[-2.36692929, -3.483900e-01, 0.322934568, -1.56939185], [-5.623850e-01, -0.083735466, 1.73820043, 0.218063414]]> : tensor<2x4xf32>} : () -> tensor<2x4xf32>
  %8 = "tfl.pseudo_const"() {value = dense<[1.43194032, -0.553496838]> : tensor<2xf32>} : () -> tensor<2xf32>
  %9 = "tfl.pseudo_const"() {value = dense<[-1.66391921, 1.14934266]> : tensor<2xf32>} : () -> tensor<2xf32>
  %10 = "tfl.pseudo_const"() {value = dense<[-1.59288621, 0.904723584]> : tensor<2xf32>} : () -> tensor<2xf32>
  %11 = "tfl.pseudo_const"() {value = dense<[-0.323118627, 1.77580559]> : tensor<2xf32>} : () -> tensor<2xf32>
  %12 = "tfl.pseudo_const"() {value = dense<[-1.0347594, -1.09994471]> : tensor<2xf32>} : () -> tensor<2xf32>
  %13 = "tfl.pseudo_const"() {value = dense<[-2.03072214, -1.63648951]> : tensor<2xf32>} : () -> tensor<2xf32>
  %14 = "tfl.pseudo_const"() {value = dense<[-1.90073407, -0.286088765]> : tensor<2xf32>} : () -> tensor<2xf32>
  %15 = "tfl.pseudo_const"() {value = dense<[[0.580187321, -1.72028887], [1.48392391, 0.859561979], [0.316514879, 0.81852132], [0.0933789983, 0.58165586]]> : tensor<4x2xf32>} : () -> tensor<4x2xf32>
  %16 = "tfl.pseudo_const"() {value = dense<[-0.0432887711, -0.431485623, -0.307492912, -0.882515907]> : tensor<4xf32>} : () -> tensor<4xf32>
  %recurrent_input = "tfl.pseudo_const"() {value = dense<0.000000e+00> : tensor<1x4xf32>} : () -> tensor<1x4xf32>
  %recurrent_stats = "quant.stats"(%recurrent_input) {layerStats = dense<[-2.0, 1.0]> : tensor<2xf32>} : (tensor<1x4xf32>) -> tensor<1x4xf32>
  %cell_input = "tfl.pseudo_const"() {value = dense<1.000000e+00> : tensor<1x2xf32>} : () -> tensor<1x2xf32>
  %cell_stats = "quant.stats"(%cell_input) {layerStats = dense<[-2.73090601, 7.94872093]> : tensor<2xf32>} : (tensor<1x2xf32>) -> tensor<1x2xf32>
  %19 = "tfl.pseudo_const"() {value = dense<[0.928654432, -0.393729329]> : tensor<2xf32>} : () -> tensor<2xf32>
  %20 = "tfl.pseudo_const"() {value = dense<[-0.76004064, -0.892570137]> : tensor<2xf32>} : () -> tensor<2xf32>
  %21 = "tfl.pseudo_const"() {value = dense<[-0.330534697, -1.68513882]> : tensor<2xf32>} : () -> tensor<2xf32>
  %22 = "tfl.pseudo_const"() {value = dense<[-0.896740913, -0.382640809]> : tensor<2xf32>} : () -> tensor<2xf32>
  %23 = "tfl.unidirectional_sequence_lstm"(%input,
    %0, %1, %2, %3,
    %4, %5, %6, %7,
    %8, %9, %10,
    %11, %12, %13, %14,
    %15, %16,
    %recurrent_stats, %cell_stats,
    %19, %20, %21, %22) {cell_clip = 5.000000e+01 : f32,
      effective_hidden_scale_intermediate = tensor<!quant.calibrated<f32<-5.000000e-01:5.000000e-01>>>,
      fused_activation_function = "TANH",
      input_to_cell_intermediate = tensor<!quant.calibrated<f32<-4.000000e+00:4.000000e+00>>>,
      input_to_forget_intermediate = tensor<!quant.calibrated<f32<-1.600000e+01:1.600000e+01>>>,
      input_to_input_intermediate = tensor<!quant.calibrated<f32<-3.200000e+01:3.200000e+01>>>,
      input_to_output_intermediate = tensor<!quant.calibrated<f32<-1.000000e+00:1.000000e+00>>>,
      proj_clip = 0.000000e+00 : f32, time_major = false} : (
        tensor<1x1x5xf32>,
        tensor<2x5xf32>, tensor<2x5xf32>, tensor<2x5xf32>, tensor<2x5xf32>,
        tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>,
        tensor<2xf32>, tensor<2xf32>, tensor<2xf32>,
        tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>,
        tensor<4x2xf32>, tensor<4xf32>,
        tensor<1x4xf32>, tensor<1x2xf32>,
        tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> tensor<*xf32>
  %24 = "quant.stats"(%23) {layerStats = dense<[-1.0, 2.0]> : tensor<2xf32>} : (tensor<*xf32>) -> tensor<*xf32>
  func.return %24 : tensor<*xf32>
}

// CHECK-LABEL: DoNotThrownPartialQuantizeUnidirectionalLstm
func.func @DoNotThrownPartialQuantizeUnidirectionalLstm(%arg0: tensor<1x1x5xf32>) -> tensor<*xf32> attributes {tf.entry_function = {inputs = "input0", outputs = "output24"}} {
  %0 = "tfl.pseudo_const"() {value = dense<[[1.31760073, -0.78338623, 0.287265539, -0.383972764, -0.00321021513], [0.104248755, 1.07823908, 0.138089031, 0.76123321, -1.4124943]]> : tensor<2x5xf32>} : () -> tensor<2x5xf32>
  %1 = "tfl.pseudo_const"() {value = dense<[[2.32939887, -0.623641372, -0.0191893689, 0.326861918, 0.734137893], [0.499284297, 1.25277913, 0.60228157, -1.39478016, 0.115529917]]> : tensor<2x5xf32>} : () -> tensor<2x5xf32>
  %2 = "tfl.pseudo_const"() {value = dense<[[0.839470446, 0.564852297, -0.80136007, -0.0372898243, 0.57127893], [-5.516230e-01, -1.082380e+00, 1.41860521, -0.92541927, -1.13971734]]> : tensor<2x5xf32>} : () -> tensor<2x5xf32>
  %3 = "tfl.pseudo_const"() {value = dense<[[-0.440826088, -0.0863231644, -0.707756281, -0.695703208, -1.87899077], [0.16942361, 0.206325337, 1.09067786, -2.18648934, 0.273400396]]> : tensor<2x5xf32>} : () -> tensor<2x5xf32>
  %4 = "tfl.pseudo_const"() {value = dense<[[-1.65420437, 0.19633314, 0.828249216, -0.546153665], [-1.49073172, 1.6467551, 0.904948651, 1.1367631]]> : tensor<2x4xf32>} : () -> tensor<2x4xf32>
  %5 = "tfl.pseudo_const"() {value = dense<[[-0.435141891, -0.940576493, 1.30446923, -1.02953017], [0.684501767, 0.363370508, -2.29151702, 2.41928673]]> : tensor<2x4xf32>} : () -> tensor<2x4xf32>
  %6 = "tfl.pseudo_const"() {value = dense<[[0.270476967, 0.00706229592, 0.489950746, 1.05166924], [1.28193891, 0.273171216, 0.484176666, 1.11504579]]> : tensor<2x4xf32>} : () -> tensor<2x4xf32>
  %7 = "tfl.pseudo_const"() {value = dense<[[-2.36692929, -3.483900e-01, 0.322934568, -1.56939185], [-5.623850e-01, -0.083735466, 1.73820043, 0.218063414]]> : tensor<2x4xf32>} : () -> tensor<2x4xf32>
  %8 = "tfl.pseudo_const"() {value = dense<[1.43194032, -0.553496838]> : tensor<2xf32>} : () -> tensor<2xf32>
  %9 = "tfl.pseudo_const"() {value = dense<[-1.66391921, 1.14934266]> : tensor<2xf32>} : () -> tensor<2xf32>
  %10 = "tfl.pseudo_const"() {value = dense<[-1.59288621, 0.904723584]> : tensor<2xf32>} : () -> tensor<2xf32>
  %11 = "tfl.pseudo_const"() {value = dense<[-0.323118627, 1.77580559]> : tensor<2xf32>} : () -> tensor<2xf32>
  %12 = "tfl.pseudo_const"() {value = dense<[-1.0347594, -1.09994471]> : tensor<2xf32>} : () -> tensor<2xf32>
  %13 = "tfl.pseudo_const"() {value = dense<[-2.03072214, -1.63648951]> : tensor<2xf32>} : () -> tensor<2xf32>
  %14 = "tfl.pseudo_const"() {value = dense<[-1.90073407, -0.286088765]> : tensor<2xf32>} : () -> tensor<2xf32>
  %15 = "tfl.pseudo_const"() {value = dense<[[0.580187321, -1.72028887], [1.48392391, 0.859561979], [0.316514879, 0.81852132], [0.0933789983, 0.58165586]]> : tensor<4x2xf32>} : () -> tensor<4x2xf32>
  %16 = "tfl.pseudo_const"() {value = dense<[-0.0432887711, -0.431485623, -0.307492912, -0.882515907]> : tensor<4xf32>} : () -> tensor<4xf32>
  %recurrent_input = "tfl.pseudo_const"() {value = dense<0.000000e+00> : tensor<1x4xf32>} : () -> tensor<1x4xf32>
  %recurrent_stats = "quant.stats"(%recurrent_input) {layerStats = dense<[-2.0, 1.0]> : tensor<2xf32>} : (tensor<1x4xf32>) -> tensor<1x4xf32>
  %cell_input = "tfl.pseudo_const"() {value = dense<1.000000e+00> : tensor<1x2xf32>} : () -> tensor<1x2xf32>
  %cell_stats = "quant.stats"(%cell_input) {layerStats = dense<[-2.73090601, 7.94872093]> : tensor<2xf32>} : (tensor<1x2xf32>) -> tensor<1x2xf32>
  %19 = "tfl.pseudo_const"() {value = dense<[0.928654432, -0.393729329]> : tensor<2xf32>} : () -> tensor<2xf32>
  %20 = "tfl.pseudo_const"() {value = dense<[-0.76004064, -0.892570137]> : tensor<2xf32>} : () -> tensor<2xf32>
  %21 = "tfl.pseudo_const"() {value = dense<[-0.330534697, -1.68513882]> : tensor<2xf32>} : () -> tensor<2xf32>
  %22 = "tfl.pseudo_const"() {value = dense<[-0.896740913, -0.382640809]> : tensor<2xf32>} : () -> tensor<2xf32>
  %23 = "tfl.unidirectional_sequence_lstm"(%arg0,
    %0, %1, %2, %3,
    %4, %5, %6, %7,
    %8, %9, %10,
    %11, %12, %13, %14,
    %15, %16,
    %recurrent_stats, %cell_stats,
    %19, %20, %21, %22) {cell_clip = 5.000000e+01 : f32,
      effective_hidden_scale_intermediate = tensor<!quant.calibrated<f32<-5.000000e-01:5.000000e-01>>>,
      fused_activation_function = "TANH",
      input_to_cell_intermediate = tensor<!quant.calibrated<f32<-4.000000e+00:4.000000e+00>>>,
      input_to_forget_intermediate = tensor<!quant.calibrated<f32<-1.600000e+01:1.600000e+01>>>,
      input_to_input_intermediate = tensor<!quant.calibrated<f32<-3.200000e+01:3.200000e+01>>>,
      input_to_output_intermediate = tensor<!quant.calibrated<f32<-1.000000e+00:1.000000e+00>>>,
      proj_clip = 0.000000e+00 : f32, time_major = false} : (
        tensor<1x1x5xf32>,
        tensor<2x5xf32>, tensor<2x5xf32>, tensor<2x5xf32>, tensor<2x5xf32>,
        tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>,
        tensor<2xf32>, tensor<2xf32>, tensor<2xf32>,
        tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>,
        tensor<4x2xf32>, tensor<4xf32>,
        tensor<1x4xf32>, tensor<1x2xf32>,
        tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> tensor<*xf32>
  %24 = "quant.stats"(%23) {layerStats = dense<[-1.0, 2.0]> : tensor<2xf32>} : (tensor<*xf32>) -> tensor<*xf32>
  func.return %24 : tensor<*xf32>
}

// CHECK-LABEL: QuantizeLstmFull
func.func @QuantizeLstmFull(%arg0: tensor<1x5xf32>) -> tensor<*xf32> attributes {tf.entry_function = {inputs = "input0", outputs = "output24"}} {
  %input = "quant.stats"(%arg0) {layerStats = dense<[-1.2, 1.5]> : tensor<2xf32>} : (tensor<1x5xf32>) -> tensor<1x5xf32>
  %0 = "tfl.pseudo_const"() {value = dense<[[1.31760073, -0.78338623, 0.287265539, -0.383972764, -0.00321021513], [0.104248755, 1.07823908, 0.138089031, 0.76123321, -1.4124943]]> : tensor<2x5xf32>} : () -> tensor<2x5xf32>
  %1 = "tfl.pseudo_const"() {value = dense<[[2.32939887, -0.623641372, -0.0191893689, 0.326861918, 0.734137893], [0.499284297, 1.25277913, 0.60228157, -1.39478016, 0.115529917]]> : tensor<2x5xf32>} : () -> tensor<2x5xf32>
  %2 = "tfl.pseudo_const"() {value = dense<[[0.839470446, 0.564852297, -0.80136007, -0.0372898243, 0.57127893], [-5.516230e-01, -1.082380e+00, 1.41860521, -0.92541927, -1.13971734]]> : tensor<2x5xf32>} : () -> tensor<2x5xf32>
  %3 = "tfl.pseudo_const"() {value = dense<[[-0.440826088, -0.0863231644, -0.707756281, -0.695703208, -1.87899077], [0.16942361, 0.206325337, 1.09067786, -2.18648934, 0.273400396]]> : tensor<2x5xf32>} : () -> tensor<2x5xf32>
  %4 = "tfl.pseudo_const"() {value = dense<[[-1.65420437, 0.19633314, 0.828249216, -0.546153665], [-1.49073172, 1.6467551, 0.904948651, 1.1367631]]> : tensor<2x4xf32>} : () -> tensor<2x4xf32>
  %5 = "tfl.pseudo_const"() {value = dense<[[-0.435141891, -0.940576493, 1.30446923, -1.02953017], [0.684501767, 0.363370508, -2.29151702, 2.41928673]]> : tensor<2x4xf32>} : () -> tensor<2x4xf32>
  %6 = "tfl.pseudo_const"() {value = dense<[[0.270476967, 0.00706229592, 0.489950746, 1.05166924], [1.28193891, 0.273171216, 0.484176666, 1.11504579]]> : tensor<2x4xf32>} : () -> tensor<2x4xf32>
  %7 = "tfl.pseudo_const"() {value = dense<[[-2.36692929, -3.483900e-01, 0.322934568, -1.56939185], [-5.623850e-01, -0.083735466, 1.73820043, 0.218063414]]> : tensor<2x4xf32>} : () -> tensor<2x4xf32>
  %8 = "tfl.pseudo_const"() {value = dense<[1.43194032, -0.553496838]> : tensor<2xf32>} : () -> tensor<2xf32>
  %9 = "tfl.pseudo_const"() {value = dense<[-1.66391921, 1.14934266]> : tensor<2xf32>} : () -> tensor<2xf32>
  %10 = "tfl.pseudo_const"() {value = dense<[-1.59288621, 0.904723584]> : tensor<2xf32>} : () -> tensor<2xf32>
  %11 = "tfl.pseudo_const"() {value = dense<[-0.323118627, 1.77580559]> : tensor<2xf32>} : () -> tensor<2xf32>
  %12 = "tfl.pseudo_const"() {value = dense<[-1.0347594, -1.09994471]> : tensor<2xf32>} : () -> tensor<2xf32>
  %13 = "tfl.pseudo_const"() {value = dense<[-2.03072214, -1.63648951]> : tensor<2xf32>} : () -> tensor<2xf32>
  %14 = "tfl.pseudo_const"() {value = dense<[-1.90073407, -0.286088765]> : tensor<2xf32>} : () -> tensor<2xf32>
  %15 = "tfl.pseudo_const"() {value = dense<[[0.580187321, -1.72028887], [1.48392391, 0.859561979], [0.316514879, 0.81852132], [0.0933789983, 0.58165586]]> : tensor<4x2xf32>} : () -> tensor<4x2xf32>
  %16 = "tfl.pseudo_const"() {value = dense<[-0.0432887711, -0.431485623, -0.307492912, -0.882515907]> : tensor<4xf32>} : () -> tensor<4xf32>
  %recurrent_input = "tfl.pseudo_const"() {value = dense<0.000000e+00> : tensor<1x4xf32>} : () -> tensor<1x4xf32>
  %recurrent_stats = "quant.stats"(%recurrent_input) {layerStats = dense<[-2.0, 1.0]> : tensor<2xf32>} : (tensor<1x4xf32>) -> tensor<1x4xf32>
  %cell_input = "tfl.pseudo_const"() {value = dense<1.000000e+00> : tensor<1x2xf32>} : () -> tensor<1x2xf32>
  %cell_stats = "quant.stats"(%cell_input) {layerStats = dense<[-2.73090601, 7.94872093]> : tensor<2xf32>} : (tensor<1x2xf32>) -> tensor<1x2xf32>
  %19 = "tfl.pseudo_const"() {value = dense<[0.928654432, -0.393729329]> : tensor<2xf32>} : () -> tensor<2xf32>
  %20 = "tfl.pseudo_const"() {value = dense<[-0.76004064, -0.892570137]> : tensor<2xf32>} : () -> tensor<2xf32>
  %21 = "tfl.pseudo_const"() {value = dense<[-0.330534697, -1.68513882]> : tensor<2xf32>} : () -> tensor<2xf32>
  %22 = "tfl.pseudo_const"() {value = dense<[-0.896740913, -0.382640809]> : tensor<2xf32>} : () -> tensor<2xf32>
  %23 = "tfl.lstm"(%input,
    %0, %1, %2, %3,
    %4, %5, %6, %7,
    %8, %9, %10,
    %11, %12, %13, %14,
    %15, %16,
    %recurrent_stats, %cell_stats,
    %19, %20, %21, %22) ({}) {
      cell_clip = 5.000000e+01 : f32,
      effective_hidden_scale_intermediate = tensor<!quant.calibrated<f32<-5.000000e-01:5.000000e-01>>>,
      fused_activation_function = "TANH",
      input_to_cell_intermediate = tensor<!quant.calibrated<f32<-4.000000e+00:4.000000e+00>>>,
      input_to_forget_intermediate = tensor<!quant.calibrated<f32<-1.600000e+01:1.600000e+01>>>,
      input_to_input_intermediate = tensor<!quant.calibrated<f32<-3.200000e+01:3.200000e+01>>>,
      input_to_output_intermediate = tensor<!quant.calibrated<f32<-1.000000e+00:1.000000e+00>>>,
      proj_clip = 0.000000e+00 : f32,time_major = false} : (
        tensor<1x5xf32>,
        tensor<2x5xf32>, tensor<2x5xf32>, tensor<2x5xf32>, tensor<2x5xf32>,
        tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>,
        tensor<2xf32>, tensor<2xf32>, tensor<2xf32>,
        tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>,
        tensor<4x2xf32>, tensor<4xf32>,
        tensor<1x4xf32>, tensor<1x2xf32>,
        tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> tensor<*xf32>
  %24 = "quant.stats"(%23) {layerStats = dense<[-1.0, 2.0]> : tensor<2xf32>} : (tensor<*xf32>) -> tensor<*xf32>
  func.return %24 : tensor<*xf32>

// CHECK-DAG: %[[input_0:.*]] = "tfl.dequantize"({{.*}}) : (tensor<1x5x!quant.uniform<i8:f32, 0.010588235481112611:-15>>) -> tensor<1x5xf32>
// CHECK-DAG: %[[input_1:.*]] = "tfl.dequantize"({{.*}}) : (tensor<2x5x!quant.uniform<i8<-127:127>:f32, 0.011122002376346137>>) -> tensor<2x5xf32>
// CHECK-DAG: %[[input_2:.*]] = "tfl.dequantize"({{.*}}) : (tensor<2x5x!quant.uniform<i8<-127:127>:f32, 0.018341723389512912>>) -> tensor<2x5xf32>
// CHECK-DAG: %[[input_3:.*]] = "tfl.dequantize"({{.*}}) : (tensor<2x5x!quant.uniform<i8<-127:127>:f32, 0.011170119751156785>>) -> tensor<2x5xf32>
// CHECK-DAG: %[[input_4:.*]] = "tfl.dequantize"({{.*}}) : (tensor<2x5x!quant.uniform<i8<-127:127>:f32, 0.017216451524749515>>) -> tensor<2x5xf32>
// CHECK-DAG: %[[input_5:.*]] = "tfl.dequantize"({{.*}}) : (tensor<2x4x!quant.uniform<i8<-127:127>:f32, 0.013025231248750461>>) -> tensor<2x4xf32>
// CHECK-DAG: %[[input_6:.*]] = "tfl.dequantize"({{.*}}) : (tensor<2x4x!quant.uniform<i8<-127:127>:f32, 0.019049501794529713>>) -> tensor<2x4xf32>
// CHECK-DAG: %[[input_7:.*]] = "tfl.dequantize"({{.*}}) : (tensor<2x4x!quant.uniform<i8<-127:127>:f32, 0.010094007169167826>>) -> tensor<2x4xf32>
// CHECK-DAG: %[[input_8:.*]] = "tfl.dequantize"({{.*}}) : (tensor<2x4x!quant.uniform<i8<-127:127>:f32, 0.018637238525030179>>) -> tensor<2x4xf32>
// CHECK-DAG: %[[input_9:.*]] = "tfl.dequantize"({{.*}}) : (tensor<2x!quant.uniform<i16<-32767:32767>:f32, 4.3700684138124656E-5>>) -> tensor<2xf32>
// CHECK-DAG: %[[input_10:.*]] = "tfl.dequantize"({{.*}}) : (tensor<2x!quant.uniform<i16<-32767:32767>:f32, 5.0780334190922573E-5>>) -> tensor<2xf32>
// CHECK-DAG: %[[input_11:.*]] = "tfl.dequantize"({{.*}}) : (tensor<2x!quant.uniform<i16<-32767:32767>:f32, 4.8612512878442185E-5>>) -> tensor<2xf32>
// CHECK-DAG: %[[input_12:.*]] = "tfl.dequantize"({{.*}}) : (tensor<2x!quant.uniform<i32:f32, 2.7676903410132078E-8>>) -> tensor<2xf32>
// CHECK-DAG: %[[input_13:.*]] = "tfl.dequantize"({{.*}}) : (tensor<2x!quant.uniform<i32:f32, 2.6601474818224132E-8>>) -> tensor<2xf32>
// CHECK-DAG: %[[input_14:.*]] = "tfl.dequantize"({{.*}}) : (tensor<2x!quant.uniform<i32:f32, 5.0222583101003261E-8>>) -> tensor<2xf32>
// CHECK-DAG: %[[input_15:.*]] = "tfl.dequantize"({{.*}}) : (tensor<2x!quant.uniform<i32:f32, 2.6725777405118232E-8>>) -> tensor<2xf32>
// CHECK-DAG: %[[input_16:.*]] = "tfl.dequantize"({{.*}}) : (tensor<4x2x!quant.uniform<i8<-127:127>:f32, 0.013545581674951268>>) -> tensor<4x2xf32>
// CHECK-DAG: %[[input_17:.*]] = "tfl.dequantize"({{.*}}) : (tensor<4x!quant.uniform<i32:f32, 5.3119928137063791E-5>>) -> tensor<4xf32>
// CHECK-DAG: %[[input_18:.*]] = "tfl.dequantize"({{.*}}) : (tensor<1x4x!quant.uniform<i8:f32, 0.015686274509803921:-1>>) -> tensor<1x4xf32>
// CHECK-DAG: %[[input_19:.*]] = "tfl.dequantize"({{.*}}) : (tensor<1x2x!quant.uniform<i16:f32, 2.44140625E-4>>) -> tensor<1x2xf32>
// CHECK-DAG: %[[input_20:.*]] = "tfl.dequantize"({{.*}}) : (tensor<2x!quant.uniform<i16<-32767:32767>:f32, 2.8341149091975248E-5>>) -> tensor<2xf32>
// CHECK-DAG: %[[input_21:.*]] = "tfl.dequantize"({{.*}}) : (tensor<2x!quant.uniform<i16<-32767:32767>:f32, 2.7239910213861512E-5>>) -> tensor<2xf32>
// CHECK-DAG: %[[input_22:.*]] = "tfl.dequantize"({{.*}}) : (tensor<2x!quant.uniform<i16<-32767:32767>:f32, 5.1427925095427339E-5>>) -> tensor<2xf32>
// CHECK-DAG: %[[input_23:.*]] = "tfl.dequantize"({{.*}}) : (tensor<2x!quant.uniform<i16<-32767:32767>:f32, 2.736719606284107E-5>>) -> tensor<2xf32>

// CHECK: %[[lstm:.*]] = "tfl.lstm"(%[[input_0]], %[[input_1]], %[[input_2]], %[[input_3]], %[[input_4]], %[[input_5]], %[[input_6]], %[[input_7]], %[[input_8]],
// CHECK-SAME: %[[input_9]], %[[input_10]], %[[input_11]], %[[input_12]], %[[input_13]], %[[input_14]], %[[input_15]], %[[input_16]], %[[input_17]], %[[input_18]], %[[input_19]],
// CHECK-SAME: %[[input_20]], %[[input_21]], %[[input_22]], %[[input_23]])
// CHECK-NEXT: effective_hidden_scale_intermediate = tensor<!quant.uniform<i8:f32, 0.0039215686274509803:-1>>
// CHECK-SAME: input_to_cell_intermediate = tensor<!quant.uniform<i16<-32767:32767>:f32, 1.2207403790398877E-4>>
// CHECK-SAME: input_to_forget_intermediate = tensor<!quant.uniform<i16<-32767:32767>:f32, 4.8829615161595508E-4>>
// CHECK-SAME: input_to_input_intermediate = tensor<!quant.uniform<i16<-32767:32767>:f32, 9.7659230323191015E-4>>
// CHECK-SAME: input_to_output_intermediate = tensor<!quant.uniform<i16<-32767:32767>:f32, 3.0518509475997192E-5>>

// CHECK: "tfl.quantize"(%[[lstm]]) {qtype = tensor<*x!quant.uniform<i8:f32, 0.015686274509803921:-1>>, volatile}
}

// CHECK-LABEL: QuantizeSVDF
func.func @QuantizeSVDF(%arg0: tensor<1x3xf32>) -> tensor<1x2xf32>  {
  %0 = "quant.stats"(%arg0) {layerStats = dense<[2.07937503, 1.365000e+01]> : tensor<2xf32>} : (tensor<1x3xf32>) -> tensor<1x3xf32>
  %1 = "tfl.pseudo_const"() {value = dense<[[1.125947117805481, 1.0, 1.1], [-1.164743185043335, -1.0, -1.1]]> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
  %2 = "tfl.pseudo_const"() {value = dense<[[1.8328168392181396], [-1.897219181060791]]> : tensor<2x1xf32>} : () -> tensor<2x1xf32>
  %3 = "tfl.pseudo_const"() {value = dense<[1.4014043807983398, -1.0950859785079956]> : tensor<2xf32>} : () -> tensor<2xf32>
  %4 = "tfl.pseudo_const"() {value = dense<0.000000e+00> : tensor<1x4xf32>} : () -> tensor<1x4xf32>
  %5 = "quant.stats"(%4) {layerStats = dense<[-56.2916565, 122.922478]> : tensor<2xf32>} : (tensor<1x4xf32>) -> tensor<1x4xf32>
  %6 = "tfl.svdf"(%0, %1, %2, %3, %5) {fused_activation_function = "RELU", rank = 1 : i32} : (tensor<1x3xf32>, tensor<2x3xf32>, tensor<2x1xf32>, tensor<2xf32>, tensor<1x4xf32>) -> tensor<1x2xf32>
  %7 = "quant.stats"(%6) {layerStats = dense<[0.000000e+00, 33.0349121]> : tensor<2xf32>} : (tensor<1x2xf32>) -> tensor<1x2xf32>
  func.return %7 : tensor<1x2xf32>

// CHECK-DAG: %[[input_0:.*]] = "tfl.dequantize"({{.*}}) : (tensor<1x3x!quant.uniform<i8:f32, 0.053529410268746171:-128>>)
// CHECK-DAG: %[[input_1:.*]] = "tfl.dequantize"({{.*}}) : (tensor<2x3x!quant.uniform<i8<-127:127>:f32, 0.0091712061814435818>>)
// CHECK-DAG: %[[input_2:.*]] = "tfl.dequantize"({{.*}}) : (tensor<2x1x!quant.uniform<i16<-512:512>:f32, 0.0037055062130093575>>)
// CHECK-DAG: %[[input_3:.*]] = "tfl.dequantize"({{.*}}) : (tensor<2x!quant.uniform<i32:f32, 1.3900876031311922E-5>>)
// CHECK-DAG: %[[input_4:.*]] = "tfl.dequantize"({{.*}}) : (tensor<1x4x!quant.uniform<i16<-32767:32767>:f32, 0.0037514108011770368>>)
// CHECK: %[[svdf:.*]] = "tfl.svdf"(%[[input_0]], %[[input_1]], %[[input_2]], %[[input_3]], %[[input_4]])
// CHECK: %[[q:.*]] = "tfl.quantize"(%[[svdf]]) {qtype = tensor<1x2x!quant.uniform<i8:f32, 0.12954867493872549:-128>>, volatile}
// CHECK: %[[dq:.*]] = "tfl.dequantize"(%[[q]])
// CHECK: return %[[dq]]
}

// CHECK-LABEL: ZeroPointLegacy
// Legacy-LABEL: ZeroPointLegacy
// Legacy mode re-calculates zero point when it's changed due to subtle difference in scale.
func.func @ZeroPointLegacy(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32>  {
  %0 = "quant.stats"(%arg0) {layerStats = dense<[-1.0, 1.20779215]> : tensor<2xf32>} : (tensor<1x2xf32>) -> tensor<1x2xf32>
  func.return %0 : tensor<1x2xf32>
// CHECK: %1 = "tfl.dequantize"(%0) : (tensor<1x2x!quant.uniform<i8:f32, 0.0086580084819419707:-12>>)
// Legacy: %1 = "tfl.dequantize"(%0) : (tensor<1x2x!quant.uniform<i8:f32, 0.0086580086499452591:-13>>)
}
