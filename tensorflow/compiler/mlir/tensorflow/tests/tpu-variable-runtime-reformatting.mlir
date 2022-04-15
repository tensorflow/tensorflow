// RUN: tf-opt %s -split-input-file -tf-tpu-variable-runtime-reformatting| FileCheck %s

// Tests that the pass can correctly transform a training loop with 2 replicas.

!tf_res_f32 = type tensor<*x!tf_type.resource<tensor<f32>>>
!tf_res_md_f32 = type tensor<*x!tf_type.resource<tensor<3x3x1x32xf32>>> // Multi-dim f32

module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
  // CHECK-LABEL: func @main
  // CHECK-SAME: %[[ARG0:.*]]: tensor<*x!tf_type.resource<tensor<f32>>> {tf.device = "/device:TPU:0"},
  // CHECK-SAME: %[[ARG1:.*]]: tensor<*x!tf_type.resource<tensor<f32>>> {tf.device = "/device:TPU:1"},
  // CHECK-SAME: %[[ARG2:.*]]: tensor<*x!tf_type.resource<tensor<3x3x1x32xf32>>> {tf.device = "/device:TPU:0"},
  // CHECK-SAME: %[[ARG3:.*]]: tensor<*x!tf_type.resource<tensor<3x3x1x32xf32>>> {tf.device = "/device:TPU:1"})
  func.func @main(%arg0: !tf_res_f32 {tf.device = "/device:TPU:0"},
             %arg1: !tf_res_f32 {tf.device = "/device:TPU:1"},
             %arg2: !tf_res_md_f32 {tf.device = "/device:TPU:0"},
             %arg3: !tf_res_md_f32 {tf.device = "/device:TPU:1"}) {

    %0 = "tf.Const"() {value = dense<100> : tensor<i32>} : () -> tensor<i32>
    // CHECK: %[[STATE0:.*]] = "tf.VarHandleOp"()
    // CHECK-SAME: device = "/device:TPU:0"
    // CHECK: %[[STATE1:.*]] = "tf.VarHandleOp"()
    // CHECK-SAME: device = "/device:TPU:1"
    // CHECK: %[[WHILE:.*]] = "tf.WhileRegion"(
    %1 = "tf.WhileRegion"(%0) ({
       // Condition region
       // CHECK: ^bb
       // CHECK: "tf.Yield"
       ^bb0(%carg0: tensor<i32>):
          %c0 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
          %c1 = "tf.GreaterEqual"(%carg0, %0) {T = i32, device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i1>
          "tf.Yield"(%c1) : (tensor<i1>) -> ()
      }, {
       // Body region
       // CHECK: ^bb0
       ^bb0(%barg0: tensor<i32>):
          %b0 = "tf.Const"() {value = dense<-1> : tensor<i32>} : () -> tensor<i32>
          %b1 = "tf.AddV2"(%barg0, %0) {T = i32, device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
          // CHECK: %[[COMPILE:.*]]:2 = "tf_device.launch"
          // CHECK-NEXT: "tf._TPUCompileMlir"()
          %compile:2 = "tf_device.launch"() ({
            %b2:2 = "tf._TPUCompileMlir"() {
              NumDynamicShapes = 0 : i64,
              // The metadata encodes 2 parameter and 2 return values.
              metadata = "\0A\0E\08\01\18\01\22\08\08\01\1A\01\01\22\01\00\0A \08\01\12\10\12\02\08\03\12\02\08\03\12\02\08\01\12\02\08 \18\01\22\08\08\01\1A\01\01\22\01\00\12\0A\0A\08\08\01\1A\01\01\22\01\00\12\0A\0A\08\08\01\1A\01\01\22\01\00\18\02 \01",
              mlir_module = "..."} : () -> (tensor<!tf_type.string>, tensor<2x!tf_type.string>)
            tf_device.return %b2#0, %b2#1 : tensor<!tf_type.string>, tensor<2x!tf_type.string>
          }) {device = "/device:CPU:0"} : () -> (tensor<!tf_type.string>, tensor<2x!tf_type.string>)
          "tf_device.launch"() ({
            "tf.TPUCompileSucceededAssert"(%compile#0) : (tensor<!tf_type.string>) -> ()
            tf_device.return
          }) {device = "/device:CPU:0"} : () -> ()
          // CHECK: tf_device.replicate
          // CHECK-SAME: [%[[ARG0]], %[[ARG1]]] as %[[R0:.*]]: tensor<*x!tf_type.resource<tensor<f32>>>,
          // CHECK-SAME: [%[[ARG2]], %[[ARG3]]] as %[[R1:.*]]: tensor<*x!tf_type.resource<tensor<3x3x1x32xf32>>>,
          // CHECK-SAME: [%[[STATE0]], %[[STATE1]]] as %[[R_STATE:.*]]: tensor<!tf_type.resource<tensor<2x!tf_type.string>>>
          // CHECK-SAME: devices = {TPU_REPLICATED_CORE_0 = ["/device:TPU:0", "/device:TPU:1"]
          %rep:2 = tf_device.replicate([%arg0, %arg1] as %arg30: tensor<*x!tf_type.resource<tensor<f32>>>,
                              [%arg2, %arg3] as %arg31: tensor<*x!tf_type.resource<tensor<3x3x1x32xf32>>>)
                  {_mirrored_variable_indices = [0, 1], devices = {TPU_REPLICATED_CORE_0 = ["/device:TPU:0", "/device:TPU:1"]}, n = 2 : i32} {
            // CHECK: %[[ID:.*]] = "tf.Identity"(%[[R0]])
            %id = "tf.Identity"(%arg30) : (tensor<*x!tf_type.resource<tensor<f32>>>) -> tensor<*x!tf_type.resource<tensor<f32>>>
            // CHECK: "tf_device.launch"
            // CHECK-NEXT: "tf.TPUReshardVariables"(%[[ID]], %[[R1]], %[[COMPILE]]#1, %[[R_STATE]])
            // CHECK-NEXT: tf_device.return
            // CHECK-NEXT: device = "TPU_REPLICATED_CORE_0"
            // CHECK: "tf.TPUExecuteAndUpdateVariables"(%[[ID]], %[[R1]], %[[COMPILE]]#1)
            "tf_device.launch"() ({
              "tf.TPUExecuteAndUpdateVariables"(%id, %arg31, %compile#1)
                    {device_var_reads_indices = [0, 1], device_var_updates_indices = [0, 1]}
                      : (tensor<*x!tf_type.resource<tensor<f32>>>, tensor<*x!tf_type.resource<tensor<3x3x1x32xf32>>>, tensor<2x!tf_type.string>) -> ()
              tf_device.return
            }) {device = "TPU_REPLICATED_CORE_0"} : () -> ()
            %ret = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
            tf_device.return %ret : tensor<i32>
          }
          // CHECK: "tf.Yield"
          "tf.Yield"(%b1) :  (tensor<i32>) -> ()
      }) {device = "", is_stateless = false} : (tensor<i32>) -> (tensor<i32>)
    // CHECK: %[[DEFAULT:.*]] = "tf.Const"()
    // CHECK:  tf_device.replicate
    // CHECK-SAME: as %[[V0:.*]]: tensor<*x!tf_type.resource<tensor<f32>>>,
    // CHECK-SAME: as %[[V1:.*]]: tensor<*x!tf_type.resource<tensor<3x3x1x32xf32>>>,
    // CHECK-SAME: [%[[STATE0]], %[[STATE1]]] as %[[STATE:.*]]: tensor<!tf_type.resource<tensor<2x!tf_type.string>>>
    // CHECK-SAME: devices = {TPU_REPLICATED_CORE_0 = ["/device:TPU:0", "/device:TPU:1"]
    // CHECK: "tf_device.launch"
    // CHECK-NEXT: "tf.TPUReshardVariables"(%[[V0]], %[[V1]], %[[DEFAULT]], %[[STATE]])
    // CHECK-NEXT: tf_device.return
    // CHECK-NEXT: device = "TPU_REPLICATED_CORE_0"
    func.return
  }
}


// -----

// Tests that the pass does not format variables with other uses.

!tf_res_f32 = type tensor<*x!tf_type.resource<tensor<f32>>>
!tf_res_md_f32 = type tensor<*x!tf_type.resource<tensor<3x3x1x32xf32>>> // Multi-dim f32

module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
  // CHECK-LABEL: func @main
  // CHECK-NOT: TPUReshardVariables
  func.func @main(%arg0: !tf_res_f32 {tf.device = "/device:TPU:0"},
             %arg1: !tf_res_f32 {tf.device = "/device:TPU:1"},
             %arg2: !tf_res_md_f32 {tf.device = "/device:TPU:0"},
             %arg3: !tf_res_md_f32 {tf.device = "/device:TPU:1"},
             %arg4: !tf_res_f32 {tf.device = "/device:TPU:1"}) {

    %0 = "tf.Const"() {value = dense<100> : tensor<i32>} : () -> tensor<i32>
    %1 = "tf.WhileRegion"(%0) ({
       // Condition region
       ^bb0(%carg0: tensor<i32>):
          %c0 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
          %c1 = "tf.GreaterEqual"(%carg0, %0) {T = i32, device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i1>
          "tf._UnknownOp1_"(%arg1) : (!tf_res_f32) -> ()
          "tf.Yield"(%c1) : (tensor<i1>) -> ()
      }, {
       // Body region
       ^bb0(%barg0: tensor<i32>):
          %b0 = "tf.Const"() {value = dense<-1> : tensor<i32>} : () -> tensor<i32>
          %b1 = "tf.AddV2"(%barg0, %0) {T = i32, device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
          %compile:2 = "tf_device.launch"() ({
            %b2:2 = "tf._TPUCompileMlir"() {
              NumDynamicShapes = 0 : i64,
              // The metadata encodes 3 parameter and 3 return values.
              metadata = "\0A\0E\08\01\18\01\22\08\08\01\1A\01\01\22\01\00\0A \08\01\12\10\12\02\08\03\12\02\08\03\12\02\08\01\12\02\08 \18\01\22\08\08\01\1A\01\01\22\01\00\0A\0E\08\01\18\01\22\08\08\01\1A\01\01\22\01\00\12\0A\0A\08\08\01\1A\01\01\22\01\00\12\0A\0A\08\08\01\1A\01\01\22\01\00\12\0A\0A\08\08\01\1A\01\01\22\01\00\18\02 \01",
              mlir_module = "..."} : () -> (tensor<!tf_type.string>, tensor<2x!tf_type.string>)
            tf_device.return %b2#0, %b2#1 : tensor<!tf_type.string>, tensor<2x!tf_type.string>
          }) {device = "/device:CPU:0"} : () -> (tensor<!tf_type.string>, tensor<2x!tf_type.string>)
          "tf_device.launch"() ({
            "tf.TPUCompileSucceededAssert"(%compile#0) : (tensor<!tf_type.string>) -> ()
            tf_device.return
          }) {device = "/device:CPU:0"} : () -> ()
          %id0 = "tf.Identity"(%arg3) : (!tf_res_md_f32) -> !tf_res_md_f32
          "tf._Unknown_"(%id0) : (!tf_res_md_f32) -> ()
          %newvar = "tf._SomeOp"() : () -> !tf_res_f32
          %rep:2 = tf_device.replicate([%arg0, %arg1] as %arg30: !tf_res_f32,
                                       [%arg2, %arg3] as %arg31: !tf_res_md_f32,
                                       [%newvar, %arg4] as %arg32 : !tf_res_f32)
                  {_mirrored_variable_indices = [0, 1, 2], devices = {TPU_REPLICATED_CORE_0 = ["/device:TPU:0", "/device:TPU:1"]}, n = 2 : i32} {
            // %arg30 is used in the cond function, %arg31 has other uses (%id0), and
            // %arg32 is not a pass-through.
            "tf_device.launch"() ({
              "tf.TPUExecuteAndUpdateVariables"(%arg30, %arg31, %arg32, %compile#1)
                    {device_var_reads_indices = [0, 1, 2], device_var_updates_indices = [0, 1, 2]}
                      : (!tf_res_f32, !tf_res_md_f32, !tf_res_f32, tensor<2x!tf_type.string>) -> ()
              tf_device.return
            }) {device = "TPU_REPLICATED_CORE_0"} : () -> ()
            %ret = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
            tf_device.return %ret : tensor<i32>
          }
          "tf.Yield"(%b1) :  (tensor<i32>) -> ()
      }) {device = "", is_stateless = false} : (tensor<i32>) -> (tensor<i32>)
    func.return
  }
}

// -----

// Tests that the pass does not format variables when model parallelism is
// present.

!tf_res_f32 = type tensor<*x!tf_type.resource<tensor<f32>>>
!tf_res_md_f32 = type tensor<*x!tf_type.resource<tensor<3x3x1x32xf32>>> // Multi-dim f32

module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
  // CHECK-LABEL: func @main
  // CHECK-NOT: TPUReshardVariables
  func.func @main(%arg0: !tf_res_f32 {tf.device = "/device:TPU:0"},
             %arg1: !tf_res_f32 {tf.device = "/device:TPU:1"},
             %arg2: !tf_res_md_f32 {tf.device = "/device:TPU:0"},
             %arg3: !tf_res_md_f32 {tf.device = "/device:TPU:1"}) {

    %0 = "tf.Const"() {value = dense<100> : tensor<i32>} : () -> tensor<i32>
    %1 = "tf.WhileRegion"(%0) ({
       // Condition region
       ^bb0(%carg0: tensor<i32>):
          %c0 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
          %c1 = "tf.GreaterEqual"(%carg0, %0) {T = i32, device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i1>
          "tf.Yield"(%c1) : (tensor<i1>) -> ()
      }, {
       // Body region
       ^bb0(%barg0: tensor<i32>):
          %b0 = "tf.Const"() {value = dense<-1> : tensor<i32>} : () -> tensor<i32>
          %b1 = "tf.AddV2"(%barg0, %0) {T = i32, device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
          %compile:2 = "tf_device.launch"() ({
            %b2:2 = "tf._TPUCompileMlir"() {
              NumDynamicShapes = 0 : i64,
              // The metadata encodes 2 parameter and 2 return values.
              metadata = "\0A\0E\08\01\18\01\22\08\08\01\1A\01\01\22\01\00\0A \08\01\12\10\12\02\08\03\12\02\08\03\12\02\08\01\12\02\08 \18\01\22\08\08\01\1A\01\01\22\01\00\12\0A\0A\08\08\01\1A\01\01\22\01\00\12\0A\0A\08\08\01\1A\01\01\22\01\00\18\02 \01",
              mlir_module = "..."} : () -> (tensor<!tf_type.string>, tensor<2x!tf_type.string>)
            tf_device.return %b2#0, %b2#1 : tensor<!tf_type.string>, tensor<2x!tf_type.string>
          }) {device = "/device:CPU:0"} : () -> (tensor<!tf_type.string>, tensor<2x!tf_type.string>)
          "tf_device.launch"() ({
            "tf.TPUCompileSucceededAssert"(%compile#0) : (tensor<!tf_type.string>) -> ()
            tf_device.return
          }) {device = "/device:CPU:0"} : () -> ()
          %rep:2 = tf_device.replicate([%arg0, %arg1] as %arg30: tensor<*x!tf_type.resource<tensor<f32>>>,
                              [%arg2, %arg3] as %arg31: tensor<*x!tf_type.resource<tensor<3x3x1x32xf32>>>)
                  {_mirrored_variable_indices = [0, 1], devices = {TPU_REPLICATED_CORE_0 = ["/device:TPU:0", "/device:TPU:1"]}, n = 2 : i32} {
            %id = "tf.Identity"(%arg30) : (tensor<*x!tf_type.resource<tensor<f32>>>) -> tensor<*x!tf_type.resource<tensor<f32>>>
            "tf_device.parallel_execute"() ({
              "tf_device.launch"() ({
                "tf.TPUExecuteAndUpdateVariables"(%id, %arg31, %compile#1)
                      {device_var_reads_indices = [0, 1], device_var_updates_indices = [0, 1]}
                        : (tensor<*x!tf_type.resource<tensor<f32>>>, tensor<*x!tf_type.resource<tensor<3x3x1x32xf32>>>, tensor<2x!tf_type.string>) -> ()
                tf_device.return
              }) {device = "TPU_REPLICATED_CORE_0"} : () -> ()
              tf_device.return
            }, {
              tf_device.return
            }) {} : () -> ()
            %ret = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
            tf_device.return %ret : tensor<i32>
          }
          "tf.Yield"(%b1) :  (tensor<i32>) -> ()
      }) {device = "", is_stateless = false} : (tensor<i32>) -> (tensor<i32>)
    func.return
  }
}

// -----

// Tests that the pass can correctly transform a training loop with a packed
// variable.
!tf_res_f32 = type tensor<*x!tf_type.resource<tensor<f32>>>
!tf_res_md_f32 = type tensor<*x!tf_type.resource<tensor<3x3x1x32xf32>>> // Multi-dim f32

module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
  // CHECK-LABEL: func @main
  // CHECK-SAME: %[[ARG0:.*]]: tensor<*x!tf_type.resource<tensor<f32>>> {tf.device = "/device:TPU:0"},
  // CHECK-SAME: %[[ARG1:.*]]: tensor<*x!tf_type.resource<tensor<f32>>> {tf.device = "/device:TPU:1"},
  // CHECK-SAME: %[[ARG2:.*]]: tensor<*x!tf_type.resource<tensor<3x3x1x32xf32>>> {tf.device = "/device:COMPOSITE:0"})
  func.func @main(%arg0: !tf_res_f32 {tf.device = "/device:TPU:0"},
             %arg1: !tf_res_f32 {tf.device = "/device:TPU:1"},
             %arg2: !tf_res_md_f32 {tf.device = "/device:COMPOSITE:0"}) {
    %0 = "tf.Const"() {value = dense<100> : tensor<i32>} : () -> tensor<i32>
    // CHECK: %[[STATE0:.*]] = "tf.VarHandleOp"()
    // CHECK-SAME: device = "/device:TPU:0"
    // CHECK: %[[STATE1:.*]] = "tf.VarHandleOp"()
    // CHECK-SAME: device = "/device:TPU:1"
    // CHECK: %[[WHILE:.*]] = "tf.WhileRegion"(
    %1 = "tf.WhileRegion"(%0) ({
       // Condition region
       // CHECK: ^bb
       // CHECK: "tf.Yield"
       ^bb0(%carg0: tensor<i32>):
          %c0 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
          %c1 = "tf.GreaterEqual"(%carg0, %0) {T = i32, device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i1>
          "tf.Yield"(%c1) : (tensor<i1>) -> ()
      }, {
       // Body region
       // CHECK: ^bb0
       ^bb0(%barg0: tensor<i32>):
          %b0 = "tf.Const"() {value = dense<-1> : tensor<i32>} : () -> tensor<i32>
          %b1 = "tf.AddV2"(%barg0, %0) {T = i32, device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
          // CHECK: %[[COMPILE:.*]]:2 = "tf_device.launch"
          // CHECK-NEXT: "tf._TPUCompileMlir"()
          %compile:2 = "tf_device.launch"() ({
            %b2:2 = "tf._TPUCompileMlir"() {
              NumDynamicShapes = 0 : i64,
              // The metadata encodes 2 parameter and 2 return values.
              metadata = "\0A\0E\08\01\18\01\22\08\08\01\1A\01\01\22\01\00\0A \08\01\12\10\12\02\08\03\12\02\08\03\12\02\08\01\12\02\08 \18\01\22\08\08\01\1A\01\01\22\01\00\12\0A\0A\08\08\01\1A\01\01\22\01\00\12\0A\0A\08\08\01\1A\01\01\22\01\00\18\02 \01",
              mlir_module = "..."} : () -> (tensor<!tf_type.string>, tensor<2x!tf_type.string>)
            tf_device.return %b2#0, %b2#1 : tensor<!tf_type.string>, tensor<2x!tf_type.string>
          }) {device = "/device:CPU:0"} : () -> (tensor<!tf_type.string>, tensor<2x!tf_type.string>)
          "tf_device.launch"() ({
            "tf.TPUCompileSucceededAssert"(%compile#0) : (tensor<!tf_type.string>) -> ()
            tf_device.return
          }) {device = "/device:CPU:0"} : () -> ()
          // CHECK: tf_device.replicate
          // CHECK-SAME: [%[[ARG0]], %[[ARG1]]] as %[[R0:.*]]: tensor<*x!tf_type.resource<tensor<f32>>>,
          // CHECK-SAME: [%[[STATE0]], %[[STATE1]]] as %[[R_STATE:.*]]: tensor<!tf_type.resource<tensor<2x!tf_type.string>>>
          // CHECK-SAME: %[[ARG2]] as %[[R1:.*]]: tensor<*x!tf_type.resource<tensor<3x3x1x32xf32>>>
          // CHECK-SAME: devices = {TPU_REPLICATED_CORE_0 = ["/device:TPU:0", "/device:TPU:1"]
          %rep:2 = tf_device.replicate([%arg0, %arg1] as %arg30: tensor<*x!tf_type.resource<tensor<f32>>>,
                                        %arg2 as %arg31: tensor<*x!tf_type.resource<tensor<3x3x1x32xf32>>>)
                  {_mirrored_variable_indices = [0, 1], _packed_input_indices = [1], devices = {TPU_REPLICATED_CORE_0 = ["/device:TPU:0", "/device:TPU:1"]}, n = 2 : i32} {
            // CHECK: %[[ID:.*]] = "tf.Identity"(%[[R0]])
            %id = "tf.Identity"(%arg30) : (tensor<*x!tf_type.resource<tensor<f32>>>) -> tensor<*x!tf_type.resource<tensor<f32>>>
            // CHECK: "tf_device.launch"
            // CHECK-NEXT: "tf.TPUReshardVariables"(%[[ID]], %[[R1]], %[[COMPILE]]#1, %[[R_STATE]])
            // CHECK-NEXT: tf_device.return
            // CHECK-NEXT: device = "TPU_REPLICATED_CORE_0"
            // CHECK: "tf.TPUExecuteAndUpdateVariables"(%[[ID]], %[[R1]], %[[COMPILE]]#1)
            "tf_device.launch"() ({
              "tf.TPUExecuteAndUpdateVariables"(%id, %arg31, %compile#1)
                    {device_var_reads_indices = [0, 1], device_var_updates_indices = [0, 1]}
                      : (tensor<*x!tf_type.resource<tensor<f32>>>, tensor<*x!tf_type.resource<tensor<3x3x1x32xf32>>>, tensor<2x!tf_type.string>) -> ()
              tf_device.return
            }) {device = "TPU_REPLICATED_CORE_0"} : () -> ()
            %ret = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
            tf_device.return %ret : tensor<i32>
          }
          // CHECK: "tf.Yield"
          "tf.Yield"(%b1) :  (tensor<i32>) -> ()
      }) {device = "", is_stateless = false} : (tensor<i32>) -> (tensor<i32>)
    // CHECK: %[[DEFAULT:.*]] = "tf.Const"()
    // CHECK:  tf_device.replicate
    // CHECK-SAME: [%[[ARG0]], %[[ARG1]]] as %[[V0:.*]]: tensor<*x!tf_type.resource<tensor<f32>>>,
    // CHECK-SAME: [%[[STATE0]], %[[STATE1]]] as %[[STATE:.*]]: tensor<!tf_type.resource<tensor<2x!tf_type.string>>>
    // CHECK-SAME: %[[ARG2]] as %[[V1:.*]]: tensor<*x!tf_type.resource<tensor<3x3x1x32xf32>>>
    // CHECK-SAME: devices = {TPU_REPLICATED_CORE_0 = ["/device:TPU:0", "/device:TPU:1"]
    // CHECK: "tf_device.launch"
    // CHECK-NEXT: "tf.TPUReshardVariables"(%[[V0]], %[[V1]], %[[DEFAULT]], %[[STATE]])
    // CHECK-NEXT: tf_device.return
    // CHECK-NEXT: device = "TPU_REPLICATED_CORE_0"
    func.return
  }
}
