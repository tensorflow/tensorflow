// RUN: litert-opt %s -tfl-lift-tflite-flex-ops | FileCheck %s

// CHECK-LABEL: TfAdd
func.func @TfAdd(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>) -> tensor<4xf64> {
  %0 = "tfl.custom"(%arg0, %arg1) {
    custom_code = "FlexAdd",
    custom_option = #tfl<const_bytes : "0x03416464001412034164641A001A002A070A015412023002320000021B171414042801">
  } : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>

// CHECK: "tf.Add"(%arg0, %arg1) {T = f64}  : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
  func.return %0 : tensor<4xf64>
}



// CHECK-LABEL: TfBatchMatMulV2
func.func @TfBatchMatMulV2(%arg0: tensor<4x128x2xf32>, %arg1:  tensor<2x1xf32>) -> tensor<4x128x1xf32> {
  %0 = "tfl.custom"(%arg0, %arg1) {
    custom_code = "FlexBatchMatMulV2",
    custom_option = #tfl<const_bytes : "0x0D42617463684D61744D756C56320038120D42617463684D61744D756C56321A001A002A070A0154120230012A0B0A0561646A5F78120228002A0B0A0561646A5F791202280032000002493B1414042801">
  } : (tensor<4x128x2xf32>, tensor<2x1xf32>) -> tensor<4x128x1xf32>

// CHECK: "tf.BatchMatMulV2"(%arg0, %arg1) <{adj_x = false, adj_y = false}> {T = f32} : (tensor<4x128x2xf32>, tensor<2x1xf32>) -> tensor<4x128x1xf32>
  func.return %0 : tensor<4x128x1xf32>
}


// CHECK-LABEL: TfTensorArrayV3
func.func @TfTensorArrayV3(%arg0: tensor<i32>) -> tensor<f32> {
  %0:2 = "tfl.custom"(%arg0) {
    custom_code = "FlexTensorArrayV3",
    custom_option = #tfl<const_bytes : "0x0D54656E736F724172726179563300A8120D54656E736F72417272617956331A002A1E0A186964656E746963616C5F656C656D656E745F736861706573120228012A120A0C64796E616D69635F73697A65120228002A1D0A1174656E736F725F61727261795F6E616D651208120673616D706C652A160A10636C6561725F61667465725F72656164120228012A0B0A056474797065120230012A1B0A0D656C656D656E745F7368617065120A3A08120208081202080132000002B9AB1414042801">
  } : (tensor<i32>) -> (tensor<2xi32>, tensor<*xf32>)

// CHECK: "tf.TensorArrayV3"
// CHECK-SAME: : (tensor<i32>) -> (tensor<2x!tf_type.resource>, tensor<f32>)

  %1 = "tfl.cast"(%0#1) : (tensor<*xf32>) -> tensor<f32>
  func.return %1 : tensor<f32>
}

// CHECK-LABEL: TfParseExample
func.func @TfParseExample(%arg0: tensor<1x!tf_type.string>) -> (tensor<1x1x!tf_type.string>, tensor<1x1x!tf_type.string>) {
  %0 = "tfl.pseudo_const"() {value = dense<["image/encoded", "image/text"]> : tensor<2x!tf_type.string>} : () -> tensor<2x!tf_type.string>
  %1 = "tfl.pseudo_const"() {value = dense<"image/encoded"> : tensor<1x!tf_type.string>} : () -> tensor<1x!tf_type.string>
  %2 = "tfl.pseudo_const"() {value = dense<"image/text"> : tensor<1x!tf_type.string>} : () -> tensor<1x!tf_type.string>
  %3 = "tfl.pseudo_const"() {value = dense<""> : tensor<1x!tf_type.string>} : () -> tensor<1x!tf_type.string>
  %4:2 = "tfl.custom"(%arg0, %0, %1, %2, %3, %3) {
    custom_code = "FlexParseExample",
    custom_option = #tfl<const_bytes : "0x0C50617273654578616D706C65008D120C50617273654578616D706C651A001A001A001A001A001A002A0C0A064E64656E7365120218022A1E0A0C64656E73655F736861706573120E0A0C3A04120208013A04120208012A120A0C7370617273655F747970657312020A002A0D0A074E737061727365120218002A100A065464656E736512060A0432020707320E0A0C50617273654578616D706C6500029D901414042801">
  } : (
    tensor<1x!tf_type.string>, tensor<2x!tf_type.string>, tensor<1x!tf_type.string>,
    tensor<1x!tf_type.string>, tensor<1x!tf_type.string>, tensor<1x!tf_type.string>
  ) -> (tensor<1x1x!tf_type.string>, tensor<1x1x!tf_type.string>)
  func.return %4#0, %4#1 : tensor<1x1x!tf_type.string>, tensor<1x1x!tf_type.string>
// CHECK: "tf.ParseExample"(
// CHECK-SAME: operandSegmentSizes = array<i32: 1, 1, 0, 2, 2>, resultSegmentSizes = array<i32: 0, 0, 0, 2>
}

// CHECK-LABEL: TfMapDataset
func.func @TfMapDataset(%arg0: tensor<!tf_type.variant>) -> (tensor<!tf_type.variant>) {
  %0 = "tfl.custom"(%arg0) {
    custom_code = "FlexMapDataset",
    custom_option = #tfl<const_bytes : "0x0A4D61704461746173657400CA120A4D6170446174617365741A002A1A0A1470726573657276655F63617264696E616C697479120228012A100A0A54617267756D656E747312020A002A1E0A187573655F696E7465725F6F705F706172616C6C656C69736D120228012A0E0A086D65746164617461120212002A2C0A0166122752250A235F5F696E666572656E63655F446174617365745F6D61705F6C616D6264615F343131302A150A0C6F75747075745F747970657312050A033201072A150A0D6F75747075745F73686170657312040A023A0032000002D8CD1414042801">
  } : (tensor<!tf_type.variant>) -> tensor<!tf_type.variant>

  func.return %0 : tensor<!tf_type.variant>
// CHECK: "tf.MapDataset"(
// CHECK-SAME: <{f = @{{.*}}, metadata = "", output_shapes = [#tf_type.shape<>], output_types = [!tf_type.string], preserve_cardinality = true, use_inter_op_parallelism = true}> {Targuments = []}
}

// CHECK-LABEL: TfTakeWhileDataset
func.func @TfTakeWhileDataset(%arg0: tensor<!tf_type.variant>, %arg1: tensor<!tf_type.resource>) -> (tensor<!tf_type.variant>) {
  %0 = "tfl.custom"(%arg0, %arg1) {
    custom_code = "FlexTakeWhileDataset",
    custom_option = #tfl<const_bytes : "0x1054616B655768696C654461746173657400C0121054616B655768696C65446174617365741A001A001A001A001A001A001A001A001A002A1A0A0A54617267756D656E7473120C0A0A320814140914141414092A3E0A097072656469636174651231522F0A2D5F5F696E666572656E63655F446174617365745F74616B655F7768696C655F7072656469636174655F373738302A0E0A086D65746164617461120212002A150A0C6F75747075745F747970657312050A033201072A150A0D6F75747075745F73686170657312040A023A0032000002D4C31414042801">
  } : (tensor<!tf_type.variant>, tensor<!tf_type.resource>) -> tensor<!tf_type.variant>

  func.return %0 : tensor<!tf_type.variant>
// CHECK: "tf.TakeWhileDataset"(
// CHECK-SAME: <{metadata = "", output_shapes = [#tf_type.shape<>], output_types = [!tf_type.string], predicate = @{{.*}}}> {Targuments = [!tf_type.resource, !tf_type.resource, i64, !tf_type.resource, !tf_type.resource, !tf_type.resource, !tf_type.resource, i64]}
}

// CHECK-LABEL: FailureOnInvalidOp
func.func @FailureOnInvalidOp(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>) -> tensor<4xf64> {
  // expected-error@+1 can't find registered TF op for Nop
  %0 = "tfl.custom"(%arg0, %arg1) {
    custom_code = "FlexNop",
    custom_option = #tfl<const_bytes : "0x034E6F70001412034E6F701A001A002A070A015412023002320000021B171414042801">
  } : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
  func.return %0 : tensor<4xf64>
}
