// RUN: dtensor-opt %s -split-input-file -dtensor-spmd-expansion -verify-diagnostics | FileCheck %s --dump-input=fail

// Check the SPMD expansion for FFT2D
// CHECK-LABEL: module @test_FFT2D
module @test_FFT2D {
    func.func @main(%arg0: tensor<i32>,
                    %arg1: tensor<2x4x8xcomplex<f32>> {tf._layout = "sharding_specs:b,x,unsharded, mesh:|b=1,x=1,y=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1", tf._mesh = "|b=1,x=1,y=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"}) {
        // CHECK:         "tf_device.cluster"
        // CHECK:           %[[FFT_OUT_1:.*]] = "tf.FFT"(%arg1)
        // CHECK-SAME:      _layout = ["sharding_specs:b,x,unsharded, mesh:|b=1,x=1,y=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"]
        // CHECK-SAME:      (tensor<2x4x8xcomplex<f32>>) -> tensor<2x4x8xcomplex<f32>>
        // CHECK-NEXT:      %[[CONST_OUT_1:.*]] = "tf.Const"()
        // CHECK-NEXT:      %[[TRANS_OUT_1:.*]] = "tf.Transpose"(%[[FFT_OUT_1]], %[[CONST_OUT_1]])
        // CHECK-SAME:      (tensor<2x4x8xcomplex<f32>>, tensor<3xi64>) -> tensor<2x8x4xcomplex<f32>>
        // CHECK-NEXT:      %[[IDENT_OUT:.*]] = "tf.Identity"(%[[TRANS_OUT_1]])
        // CHECK-NEXT:      %[[FFT_OUT_2:.*]] = "tf.FFT"(%[[IDENT_OUT]])
        // CHECK-SAME:      _layout = ["sharding_specs:b,x,unsharded, mesh:|b=1,x=1,y=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"]
        // CHECK-SAME:      (tensor<2x8x4xcomplex<f32>>) -> tensor<2x8x4xcomplex<f32>>
        // CHECK-NEXT:      %[[CONST_OUT_2:.*]] = "tf.Const"()
        // CHECK-NEXT:      %[[TRANS_OUT_2:.*]] = "tf.Transpose"(%[[FFT_OUT_2]], %[[CONST_OUT_2]])
        // CHECK-SAME:      (tensor<2x8x4xcomplex<f32>>, tensor<3xi64>) -> tensor<2x4x8xcomplex<f32>>
        // CHECK-NEXT:      tf_device.return
        %0 = "tf_device.cluster"() ({
            %1 = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<2x4x8>, layout = #dtensor.layout<sharding_specs:b,x,unsharded, mesh:|b=1,x=1,y=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1>} : (tensor<2x4x8xcomplex<f32>>) -> tensor<2x4x8xcomplex<f32>>
            %2 = "tf.FFT2D"(%1) : (tensor<2x4x8xcomplex<f32>>) -> tensor<2x4x8xcomplex<f32>>
            %3 = "tf.DTensorLayout"(%2) {global_shape = #tf_type.shape<2x4x8>, layout = #dtensor.layout<sharding_specs:b,unsharded,x, mesh:|b=1,x=1,y=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1>} : (tensor<2x4x8xcomplex<f32>>) -> tensor<2x4x8xcomplex<f32>>
            tf_device.return %3 : tensor<2x4x8xcomplex<f32>>
        }) {_mesh = "|b=1,x=1,y=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"} : () -> tensor<2x4x8xcomplex<f32>>
        func.return
    }
}

// -----

// Check the SPMD expansion for IFFT2D
// CHECK-LABEL: module @test_IFFT2D
module @test_IFFT2D {
    func.func @main(%arg0: tensor<i32>,
                    %arg1: tensor<2x4x8xcomplex<f64>> {tf._layout = "sharding_specs:b,x,y, mesh:|b=1,x=1,y=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1", tf._mesh = "|b=1,x=1,y=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"}) {
        // CHECK:         "tf_device.cluster"
        // CHECK:           %[[CONST_OUT_1:.*]] = "tf.Const"()
        // CHECK-NEXT:      %[[TRANS_OUT_1:.*]] = "tf.Transpose"(%arg1, %[[CONST_OUT_1]])
        // CHECK-SAME:      (tensor<2x4x4xcomplex<f64>>, tensor<3xi64>) -> tensor<2x4x4xcomplex<f64>>
        // CHECK-NEXT:      %[[IDENT_OUT:.*]] = "tf.Identity"(%[[TRANS_OUT_1]])
        // CHECK-NEXT:      %[[IFFT_OUT_1:.*]] = "tf.IFFT"(%[[IDENT_OUT]])
        // CHECK-SAME:      _layout = ["sharding_specs:b,y,unsharded, mesh:|b=1,x=1,y=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"]
        // CHECK-SAME:      (tensor<2x4x4xcomplex<f64>>) -> tensor<2x4x4xcomplex<f64>>
        // CHECK-NEXT:      %[[CONST_OUT_2:.*]] = "tf.Const"()
        // CHECK-NEXT:      %[[TRANS_OUT_2:.*]] = "tf.Transpose"(%[[IFFT_OUT_1]], %[[CONST_OUT_2]])
        // CHECK-SAME:      (tensor<2x4x4xcomplex<f64>>, tensor<3xi64>) -> tensor<2x4x4xcomplex<f64>>
        // CHECK-NEXT:      %[[ALLTOALL_OUT:.*]] = "tf.DTensorAllToAll"(%[[TRANS_OUT_2]])
        // CHECK-SAME:      _layout = ["sharding_specs:b,y,unsharded, mesh:|b=1,x=1,y=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"]
        // CHECK-SAME:      (tensor<2x4x4xcomplex<f64>>) -> tensor<2x2x8xcomplex<f64>>
        // CHECK-NEXT:      %[[IFFT_OUT_2:.*]] = "tf.IFFT"(%[[ALLTOALL_OUT]])
        // CHECK-SAME:      _layout = ["sharding_specs:b,y,unsharded, mesh:|b=1,x=1,y=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"]
        // CHECK-SAME:      (tensor<2x2x8xcomplex<f64>>) -> tensor<2x2x8xcomplex<f64>>
        // CHECK-NEXT:      tf_device.return
        %0 = "tf_device.cluster"() ({
            %1 = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<2x4x8>, layout = #dtensor.layout<sharding_specs:b,x,y, mesh:|b=1,x=1,y=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1>} : (tensor<2x4x8xcomplex<f64>>) -> tensor<2x4x8xcomplex<f64>>
            %2 = "tf.IFFT2D"(%1) : (tensor<2x4x8xcomplex<f64>>) -> tensor<2x4x8xcomplex<f64>>
            %3 = "tf.DTensorLayout"(%2) {global_shape = #tf_type.shape<2x4x8>, layout = #dtensor.layout<sharding_specs:b,y,unsharded, mesh:|b=1,x=1,y=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1>} : (tensor<2x4x8xcomplex<f64>>) -> tensor<2x4x8xcomplex<f64>>
            tf_device.return %3 : tensor<2x4x8xcomplex<f64>>
        }) {_mesh = "|b=1,x=1,y=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"} : () -> tensor<2x4x8xcomplex<f64>>
        func.return
  }
}

// -----

// Check the SPMD expansion for RFFT2D
// CHECK-LABEL: module @test_RFFT2D
module @test_RFFT2D {
    func.func @main(%arg0: tensor<i32>,
                    %arg1: tensor<2x4x12xf64> {tf._layout = "sharding_specs:b,x,y, mesh:|b=1,x=1,y=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1", tf._mesh = "|b=1,x=1,y=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"},
                    %arg2: tensor<2xi32> {tf._layout = "sharding_specs:unsharded, mesh:|b=1,x=1,y=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1", tf._mesh = "|b=1,x=1,y=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"}) {
        // CHECK:         "tf_device.cluster"
        // CHECK:           %[[CONST_OUT_1:.*]] = "tf.Const"()
        // CHECK-SAME:      _layout = ["sharding_specs:unsharded, mesh:|b=1,x=1,y=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"]
        // CHECK-NEXT:      %[[ALLGATHER_OUT:.*]] = "tf.DTensorAllGather"(%arg1)
        // CHECK-SAME:      _layout = ["sharding_specs:b,x,unsharded, mesh:|b=1,x=1,y=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"]
        // CHECK:           %[[CONST_OUT_2:.*]] = "tf.Const"()
        // CHECK:           %[[RFFT_OUT:.*]] = "tf.RFFT"(%[[ALLGATHER_OUT]], %[[CONST_OUT_2]])
        // CHECK-SAME:      _layout = ["sharding_specs:b,x,unsharded, mesh:|b=1,x=1,y=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"]
        // CHECK-SAME:      (tensor<2x4x12xf64>, tensor<1xi32>) -> tensor<2x4x6xcomplex<f64>>
        // CHECK-NEXT:      %[[CONST_OUT_3:.*]] = "tf.Const"()
        // CHECK-NEXT:      %[[TRANS_OUT_1:.*]] = "tf.Transpose"(%[[RFFT_OUT]], %[[CONST_OUT_3]])
        // CHECK-SAME:      (tensor<2x4x6xcomplex<f64>>, tensor<3xi64>) -> tensor<2x6x4xcomplex<f64>>
        // CHECK-NEXT:      %[[IDENT_OUT:.*]] = "tf.Identity"(%[[TRANS_OUT_1]])
        // CHECK-NEXT:      %[[FFT_OUT:.*]] = "tf.FFT"(%[[IDENT_OUT]])
        // CHECK-SAME:      _layout = ["sharding_specs:b,x,unsharded, mesh:|b=1,x=1,y=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"]
        // CHECK-SAME:      (tensor<2x6x4xcomplex<f64>>) -> tensor<2x6x4xcomplex<f64>>
        // CHECK-NEXT:      %[[CONST_OUT_4:.*]] = "tf.Const"()
        // CHECK-NEXT:      %[[TRANS_OUT_2:.*]] = "tf.Transpose"(%[[FFT_OUT]], %[[CONST_OUT_4]])
        // CHECK-SAME:      _layout = ["sharding_specs:b,unsharded,x, mesh:|b=1,x=1,y=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"]
        // CHECK-SAME:      (tensor<2x6x4xcomplex<f64>>, tensor<3xi64>) -> tensor<2x4x6xcomplex<f64>>
        // CHECK-NEXT:      tf_device.return
        %0 = "tf_device.cluster"() ({
            %cst = "tf.Const"() {value = dense<[4, 10]> : tensor<2xi32>} : () -> tensor<2xi32>
            %1 = "tf.DTensorLayout"(%arg2) {global_shape = #tf_type.shape<2>, layout = #dtensor.layout<sharding_specs:unsharded, mesh:|b=1,x=1,y=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1>} : (tensor<2xi32>) -> tensor<2xi32>
            %2 = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<2x4x12>, layout = #dtensor.layout<sharding_specs:b,x,y, mesh:|b=1,x=1,y=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1>} : (tensor<2x4x12xf64>) -> tensor<2x4x12xf64>
            %3 = "tf.DTensorLayout"(%cst) {global_shape = #tf_type.shape<2>, layout = #dtensor.layout<sharding_specs:unsharded, mesh:|b=1,x=1,y=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1>} : (tensor<2xi32>) -> tensor<2xi32>
            %4 = "tf.RFFT2D"(%2, %3) : (tensor<2x4x12xf64>, tensor<2xi32>) -> tensor<2x4x6xcomplex<f64>>
            %5 = "tf.DTensorLayout"(%4) {global_shape = #tf_type.shape<2x4x6>, layout = #dtensor.layout<sharding_specs:b,unsharded,x, mesh:|b=1,x=1,y=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1>} : (tensor<2x4x6xcomplex<f64>>) -> tensor<2x4x6xcomplex<f64>>
            tf_device.return %5 : tensor<2x4x6xcomplex<f64>>
        }) {_mesh = "|b=1,x=1,y=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"} : () -> tensor<2x4x6xcomplex<f64>>
        func.return
    }
}


// -----

// Check the SPMD expansion for IRFFT2D
// CHECK-LABEL: module @test_IRFFT2D
module @test_IRFFT2D {
    func.func @main(%arg0: tensor<i32>,
                    %arg1: tensor<2x4x8xcomplex<f64>> {tf._layout = "sharding_specs:b,unsharded,y, mesh:|b=1,x=1,y=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1", tf._mesh = "|b=1,x=1,y=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"},
                    %arg2: tensor<2xi32> {tf._layout = "sharding_specs:unsharded, mesh:|b=1,x=1,y=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1", tf._mesh = "|b=1,x=1,y=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"}) {
        // CHECK:         "tf_device.cluster"
        // CHECK:           %[[CONST_OUT_1:.*]] = "tf.Const"()
        // CHECK-SAME:      _layout = ["sharding_specs:unsharded, mesh:|b=1,x=1,y=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"]
        // CHECK-NEXT:      %[[CONST_OUT_2:.*]] = "tf.Const"()
        // CHECK-NEXT:      %[[TRANS_OUT_1:.*]] = "tf.Transpose"(%arg1, %[[CONST_OUT_2]])
        // CHECK-SAME:      (tensor<2x4x4xcomplex<f64>>, tensor<3xi64>) -> tensor<2x4x4xcomplex<f64>>
        // CHECK-NEXT:      %[[IFFT_OUT:.*]] = "tf.IFFT"(%[[TRANS_OUT_1]])
        // CHECK-SAME:      _layout = ["sharding_specs:b,y,unsharded, mesh:|b=1,x=1,y=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"]
        // CHECK-SAME:      (tensor<2x4x4xcomplex<f64>>) -> tensor<2x4x4xcomplex<f64>>
        // CHECK-NEXT:      %[[CONST_OUT_3:.*]] = "tf.Const"()
        // CHECK-NEXT:      %[[TRANS_OUT_2:.*]] = "tf.Transpose"(%[[IFFT_OUT]], %[[CONST_OUT_3]])
        // CHECK-SAME:      (tensor<2x4x4xcomplex<f64>>, tensor<3xi64>) -> tensor<2x4x4xcomplex<f64>>
        // CHECK-NEXT:      %[[ALLTOALL_OUT:.*]] = "tf.DTensorAllToAll"(%[[TRANS_OUT_2]])
        // CHECK-SAME:      _layout = ["sharding_specs:b,y,unsharded, mesh:|b=1,x=1,y=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"]
        // CHECK-SAME:      (tensor<2x4x4xcomplex<f64>>) -> tensor<2x2x8xcomplex<f64>>
        // CHECK-NEXT:      %[[CONST_OUT_4:.*]] = "tf.Const"()
        // CHECK-NEXT:      %[[IRFFT_OUT:.*]] = "tf.IRFFT"(%[[ALLTOALL_OUT]], %[[CONST_OUT_4]])
        // CHECK-SAME:      _layout = ["sharding_specs:b,y,unsharded, mesh:|b=1,x=1,y=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"]
        // CHECK-SAME:      (tensor<2x2x8xcomplex<f64>>, tensor<1xi32>) -> tensor<2x2x8xf64>
        // CHECK-NEXT:      tf_device.return
        %0 = "tf_device.cluster"() ({
            %cst = "tf.Const"() {value = dense<[4, 8]> : tensor<2xi32>} : () -> tensor<2xi32>
            %1 = "tf.DTensorLayout"(%arg2) {global_shape = #tf_type.shape<2>, layout = #dtensor.layout<sharding_specs:unsharded, mesh:|b=1,x=1,y=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1>} : (tensor<2xi32>) -> tensor<2xi32>
            %2 = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<2x4x8>, layout = #dtensor.layout<sharding_specs:b,unsharded,y, mesh:|b=1,x=1,y=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1>} : (tensor<2x4x8xcomplex<f64>>) -> tensor<2x4x8xcomplex<f64>>
            %3 = "tf.DTensorLayout"(%cst) {global_shape = #tf_type.shape<2>, layout = #dtensor.layout<sharding_specs:unsharded, mesh:|b=1,x=1,y=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1>} : (tensor<2xi32>) -> tensor<2xi32>
            %4 = "tf.IRFFT2D"(%2, %3) : (tensor<2x4x8xcomplex<f64>>, tensor<2xi32>) -> tensor<2x4x8xf64>
            %5 = "tf.DTensorLayout"(%4) {global_shape = #tf_type.shape<2x4x8>, layout = #dtensor.layout<sharding_specs:b,y,unsharded, mesh:|b=1,x=1,y=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1>} : (tensor<2x4x8xf64>) -> tensor<2x4x8xf64>
            tf_device.return %5 : tensor<2x4x8xf64>
        }) {_mesh = "|b=1,x=1,y=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"} : () -> tensor<2x4x8xf64>
        func.return
    }
}