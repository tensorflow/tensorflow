// RUN: tf-opt %s -tfl-prepare-quantize | FileCheck %s

// CHECK-LABEL: QuantizeLstmCellInput
func @QuantizeLstmCellInput(%arg0: tensor<1x28x28xf32>) -> tensor<1x28x20xf32> {
    %cst_1 = constant dense<1.0> : tensor<1x20xf32>
    %cst_2 = constant unit
    %cst_3 = constant dense<1.0> : tensor<20x20xf32>
    %cst_7 = constant dense<1.0> : tensor<20xf32>
    %cst_11 = constant dense<1.0> : tensor<20x28xf32>
    %cell_input = constant dense<0.0> : tensor<1x20xf32>
    %cell_stats = "quant.stats"(%cell_input) {layerStats = dense<[-2.73090601, 7.94872093]> : tensor<2xf32>} : (tensor<1x20xf32>) -> tensor<1x20xf32>
    %0 = "tfl.unidirectional_sequence_lstm"(%arg0,
      %cst_11, %cst_11, %cst_11, %cst_11,
      %cst_3, %cst_3, %cst_3, %cst_3,
      %cst_2, %cst_2, %cst_2,
      %cst_7, %cst_7, %cst_7, %cst_7,
      %cst_2, %cst_2,
      %cst_1, %cell_stats,
      %cst_2, %cst_2, %cst_2, %cst_2) {cell_clip = 1.000000e+01 : f32, fused_activation_function = "TANH", proj_clip = 0.000000e+00 : f32, time_major = false}
    : ( tensor<1x28x28xf32>,
        tensor<20x28xf32>, tensor<20x28xf32>, tensor<20x28xf32>, tensor<20x28xf32>,
        tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>,
        none, none, none,
        tensor<20xf32>, tensor<20xf32>, tensor<20xf32>, tensor<20xf32>,
        none, none,
        tensor<1x20xf32>, tensor<1x20xf32>,
        none, none, none, none) -> tensor<1x28x20xf32>
    return %0 : tensor<1x28x20xf32>
// CHECK: %[[none:.*]] = constant unit
// CHECK: %[[cell_input:.*]] = constant dense<0.000000e+00> : tensor<1x20xf32>
// CHECK: %[[q:.*]] = "tfl.quantize"(%[[cell_input]]) {qtype = tensor<1x20x!quant.uniform<i16:f32, 2.44140625E-4>>} : (tensor<1x20xf32>) -> tensor<1x20x!quant.uniform<i16:f32, 2.44140625E-4>>
// CHECK: %[[dq:.*]] = "tfl.dequantize"(%[[q]]) : (tensor<1x20x!quant.uniform<i16:f32, 2.44140625E-4>>) -> tensor<1x20xf32>
// Checks if input 19 is correctly passed from a dequantize op.
// CHECK: %[[lstm:.*]] = "tfl.unidirectional_sequence_lstm"(%arg0, {{(%[^%,]+, )+}}%[[dq]], %[[none]], %[[none]], %[[none]], %[[none]])
}

// CHECK-LABEL: QuantizeIntermediates
func @QuantizeIntermediates(%arg0: tensor<1x5xf32>) -> tensor<*xf32> attributes {tf.entry_function = {inputs = "input0", outputs = "output24"}} {
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
  %17 = "tfl.pseudo_const"() {value = dense<0.000000e+00> : tensor<1x4xf32>} : () -> tensor<1x4xf32>
  %18 = "tfl.pseudo_const"() {value = dense<1.000000e+00> : tensor<1x2xf32>} : () -> tensor<1x2xf32>
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
    %17, %18,
    %19, %20, %21, %22) {cell_clip = 5.000000e+01 : f32,
      effective_hidden_scale_intermediate = tensor<!quant.calibrated<f32<-5.000000e-01:5.000000e-01>>>,
      fused_activation_function = "TANH",
      input_to_cell_intermediate = tensor<!quant.calibrated<f32<-4.000000e+00:4.000000e+00>>>,
      input_to_forget_intermediate = tensor<!quant.calibrated<f32<-1.600000e+01:1.600000e+01>>>,
      input_to_input_intermediate = tensor<!quant.calibrated<f32<-3.200000e+01:3.200000e+01>>>,
      input_to_output_intermediate = tensor<!quant.calibrated<f32<-1.000000e+00:1.000000e+00>>>,
      proj_clip = 0.000000e+00 : f32, time_major = false} : (
        tensor<1x5xf32>,
        tensor<2x5xf32>, tensor<2x5xf32>, tensor<2x5xf32>, tensor<2x5xf32>,
        tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>,
        tensor<2xf32>, tensor<2xf32>, tensor<2xf32>,
        tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>,
        tensor<4x2xf32>, tensor<4xf32>,
        tensor<1x4xf32>, tensor<1x2xf32>,
        tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> tensor<*xf32>
  return %23 : tensor<*xf32>
// CHECK: effective_hidden_scale_intermediate = tensor<!quant.uniform<u8:f32, 0.0039215686274509803:128>>
// CHECK: input_to_cell_intermediate = tensor<!quant.uniform<i16<-32767:32767>:f32, 1.2207403790398877E-4>>
// CHECK: input_to_forget_intermediate = tensor<!quant.uniform<i16<-32767:32767>:f32, 4.8829615161595508E-4>>
// CHECK: input_to_input_intermediate = tensor<!quant.uniform<i16<-32767:32767>:f32, 9.7659230323191015E-4>>
// CHECK: input_to_output_intermediate = tensor<!quant.uniform<i16<-32767:32767>:f32, 3.0518509475997192E-5>>,
}
