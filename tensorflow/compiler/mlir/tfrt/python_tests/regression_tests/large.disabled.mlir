func.func @test(%V__0 : tensor<?x?x?xi32> { python_test_attrs.static_type = tensor<1x1x1xi32> }, %V__1 : tensor<?x?x?x?xf32> { python_test_attrs.static_type = tensor<1x1x1x1xf32> }, %V__2 : tensor<?x?x?x?xi32> { python_test_attrs.static_type = tensor<1x1x1x1xi32> }, %V__3 : tensor<?x?x?x?xi64> { python_test_attrs.static_type = tensor<1x1x1x1xi64> }, %V__4 : tensor<?x?x?x?xi1> { python_test_attrs.static_type = tensor<1x1x1x1xi1> }, %V__5 : tensor<?xi64> { python_test_attrs.static_type = tensor<1xi64> }, %V__6 : tensor<?x?x?xi64> { python_test_attrs.static_type = tensor<1x1x1xi64> }, %V__7 : tensor<?x?x?xi32> { python_test_attrs.static_type = tensor<1x1x1xi32> }, %V__8 : tensor<?x?x?xf32> { python_test_attrs.static_type = tensor<1x1x1xf32> }, %V__9 : tensor<?x?x?x?xi32> { python_test_attrs.static_type = tensor<1x1x1x1xi32> }, %V__10 : tensor<?xi32> { python_test_attrs.static_type = tensor<1xi32> }, %V__11 : tensor<?x?x?x?xf32> { python_test_attrs.static_type = tensor<1x1x1x1xf32> }, %V__12 : tensor<?x?x?x?xf32> { python_test_attrs.static_type = tensor<1x1x1x1xf32> }, %V__13 : tensor<?x?x?x?xi1> { python_test_attrs.static_type = tensor<1x1x1x1xi1> }, %V__14 : tensor<?x?x?x?xi32> { python_test_attrs.static_type = tensor<1x1x1x1xi32> }, %V__15 : tensor<?x?xi1> { python_test_attrs.static_type = tensor<1x1xi1> }, %V__16 : tensor<?x?xi32> { python_test_attrs.static_type = tensor<1x1xi32> }, %V__17 : tensor<i32> { python_test_attrs.static_type = tensor<i32> }, %V__18 : tensor<?x?x?x?xi32> { python_test_attrs.static_type = tensor<1x1x1x1xi32> }, %V__19 : tensor<?xi1> { python_test_attrs.static_type = tensor<1xi1> }, %V__20 : tensor<?x?x?x?xi32> { python_test_attrs.static_type = tensor<1x1x1x1xi32> }, %V__21 : tensor<f32> { python_test_attrs.static_type = tensor<f32> }, %V__22 : tensor<?xi32> { python_test_attrs.static_type = tensor<1xi32> }, %V__23 : tensor<?x?x?x?xi64> { python_test_attrs.static_type = tensor<1x1x1x1xi64> }, %V__24 : tensor<?xi32> { python_test_attrs.static_type = tensor<1xi32> }, %V__25 : tensor<?x?xi64> { python_test_attrs.static_type = tensor<1x1xi64> }, %V__26 : tensor<i32> { python_test_attrs.static_type = tensor<i32> }, %V__27 : tensor<?x?x?x?xf32> { python_test_attrs.static_type = tensor<1x1x1x1xf32> }, %V__28 : tensor<?x?x?x?xf32> { python_test_attrs.static_type = tensor<1x1x1x1xf32> }, %V__29 : tensor<?x?x?x?xf32> { python_test_attrs.static_type = tensor<1x1x1x1xf32> }, %V__30 : tensor<?xi32> { python_test_attrs.static_type = tensor<1xi32> }, %V__31 : tensor<?xf32> { python_test_attrs.static_type = tensor<1xf32> }, %V__32 : tensor<?x?x?x?xi32> { python_test_attrs.static_type = tensor<1x1x1x1xi32> }, %V__33 : tensor<?xi32> { python_test_attrs.static_type = tensor<1xi32> }, %V__34 : tensor<f32> { python_test_attrs.static_type = tensor<f32> }, %V__35 : tensor<f32> { python_test_attrs.static_type = tensor<f32> }, %V__36 : tensor<?x?x?x?xf32> { python_test_attrs.static_type = tensor<1x1x1x1xf32> }, %V__37 : tensor<?x?xf32> { python_test_attrs.static_type = tensor<1x1xf32> }, %V__38 : tensor<?x?x?xf32> { python_test_attrs.static_type = tensor<1x1x1xf32> }, %V__39 : tensor<?x?x?x?xf32> { python_test_attrs.static_type = tensor<1x1x1x1xf32> }, %V__40 : tensor<?x?x?x?xf32> { python_test_attrs.static_type = tensor<1x1x1x1xf32> }, %V__41 : tensor<?x?x?x?xf32> { python_test_attrs.static_type = tensor<1x1x1x1xf32> }, %V__42 : tensor<?x?x?x?xf32> { python_test_attrs.static_type = tensor<1x1x1x1xf32> }, %V__43 : tensor<?x?x?x?xf32> { python_test_attrs.static_type = tensor<1x1x1x1xf32> }, %V__44 : tensor<i1> { python_test_attrs.static_type = tensor<i1> }, %V__45 : tensor<?x?x?x?xf32> { python_test_attrs.static_type = tensor<1x1x1x1xf32> }, %V__46 : tensor<?x?x?x?xf32> { python_test_attrs.static_type = tensor<1x1x1x1xf32> }, %V__47 : tensor<?x?x?x?xf32> { python_test_attrs.static_type = tensor<1x1x1x1xf32> }, %V__48 : tensor<f32> { python_test_attrs.static_type = tensor<f32> }, %V__49 : tensor<?x?x?x?xi64> { python_test_attrs.static_type = tensor<1x1x1x1xi64> }, %V__50 : tensor<?x?x?xf32> { python_test_attrs.static_type = tensor<1x1x1xf32> }, %V__51 : tensor<?x?x?x?xf32> { python_test_attrs.static_type = tensor<1x1x1x1xf32> }, %V__52 : tensor<?x?x?x?xf32> { python_test_attrs.static_type = tensor<1x1x1x1xf32> }, %V__53 : tensor<?x?x?x?xf32> { python_test_attrs.static_type = tensor<1x1x1x1xf32> }, %V__54 : tensor<?x?x?xi1> { python_test_attrs.static_type = tensor<1x1x1xi1> }, %V__55 : tensor<?x?x?x?xi64> { python_test_attrs.static_type = tensor<1x1x1x1xi64> }, %V__56 : tensor<?x?x?xf32> { python_test_attrs.static_type = tensor<1x1x1xf32> }, %V__57 : tensor<?x?x?x?xf32> { python_test_attrs.static_type = tensor<1x1x1x1xf32> }, %V__58 : tensor<?x?x?x?xi1> { python_test_attrs.static_type = tensor<1x1x1x1xi1> }, %V__59 : tensor<?x?x?x?xi64> { python_test_attrs.static_type = tensor<1x1x1x1xi64> }, %V__60 : tensor<?x?x?x?xi32> { python_test_attrs.static_type = tensor<1x1x1x1xi32> }, %V__61 : tensor<i64> { python_test_attrs.static_type = tensor<i64> }, %V__62 : tensor<?x?x?x?xi64> { python_test_attrs.static_type = tensor<1x1x1x1xi64> }, %V__63 : tensor<i64> { python_test_attrs.static_type = tensor<i64> }, %V__64 : tensor<?x?x?x?xi32> { python_test_attrs.static_type = tensor<1x1x1x1xi32> }, %V__65 : tensor<?x?x?x?xi32> { python_test_attrs.static_type = tensor<1x1x1x1xi32> }, %V__66 : tensor<i32> { python_test_attrs.static_type = tensor<i32> }, %V__67 : tensor<?x?x?xi32> { python_test_attrs.static_type = tensor<1x1x1xi32> }, %V__68 : tensor<?x?x?x?xf32> { python_test_attrs.static_type = tensor<95x1x35x36xf32> }, %V__69 : tensor<?x?x?x?xf32> { python_test_attrs.static_type = tensor<1x1x1x1xf32> }, %V__70 : tensor<?x?x?x?xf32> { python_test_attrs.static_type = tensor<1x1x1x1xf32> }, %V__71 : tensor<?x?x?xf32> { python_test_attrs.static_type = tensor<1x1x1xf32> }, %V__72 : tensor<?x?x?x?xf32> { python_test_attrs.static_type = tensor<1x1x1x1xf32> }, %V__73 : tensor<?x?x?xi32> { python_test_attrs.static_type = tensor<1x1x1xi32> }, %V__74 : tensor<f32> { python_test_attrs.static_type = tensor<f32> }, %V__75 : tensor<?xi32> { python_test_attrs.static_type = tensor<64xi32> }, %V__76 : tensor<i32> { python_test_attrs.static_type = tensor<i32> }, %V__77 : tensor<?xi32> { python_test_attrs.static_type = tensor<1xi32> }, %V__78 : tensor<?x?x?x?xi64> { python_test_attrs.static_type = tensor<1x1x1x1xi64> }, %V__79 : tensor<?x?x?x?xi1> { python_test_attrs.static_type = tensor<1x1x1x1xi1> }, %V__80 : tensor<?x?x?x?xi64> { python_test_attrs.static_type = tensor<1x1x1x1xi64> }, %V__81 : tensor<?x?x?x?xi64> { python_test_attrs.static_type = tensor<1x1x1x1xi64> }, %V__82 : tensor<?x?x?xi64> { python_test_attrs.static_type = tensor<1x1x1xi64> }, %V__83 : tensor<?x?x?x?xi64> { python_test_attrs.static_type = tensor<1x1x1x1xi64> }, %V__84 : tensor<?x?x?x?xi64> { python_test_attrs.static_type = tensor<1x1x1x1xi64> }, %V__85 : tensor<?x?x?x?xi64> { python_test_attrs.static_type = tensor<1x1x1x1xi64> }, %V__86 : tensor<?x?x?xi1> { python_test_attrs.static_type = tensor<1x1x1xi1> }, %V__87 : tensor<?x?x?x?xi64> { python_test_attrs.static_type = tensor<1x1x1x1xi64> }, %V__88 : tensor<?x?x?x?xi64> { python_test_attrs.static_type = tensor<1x1x1x1xi64> }, %V__89 : tensor<?x?x?xi64> { python_test_attrs.static_type = tensor<1x1x1xi64> }, %V__90 : tensor<?x?x?x?xi64> { python_test_attrs.static_type = tensor<1x1x1x1xi64> }, %V__91 : tensor<?x?x?x?xi64> { python_test_attrs.static_type = tensor<1x1x1x1xi64> }, %V__92 : tensor<i64> { python_test_attrs.static_type = tensor<i64> }, %V__93 : tensor<?x?x?x?xi32> { python_test_attrs.static_type = tensor<1x1x1x1xi32> }, %V__94 : tensor<?xi64> { python_test_attrs.static_type = tensor<1xi64> }, %V__95 : tensor<?x?x?x?xi64> { python_test_attrs.static_type = tensor<1x24x7x1xi64> }, %V__96 : tensor<?x?x?xi64> { python_test_attrs.static_type = tensor<1x1x1xi64> }, %V__97 : tensor<?x?x?x?xi64> { python_test_attrs.static_type = tensor<1x1x1x1xi64> }, %V__98 : tensor<?x?x?x?xi64> { python_test_attrs.static_type = tensor<75x58x11x74xi64> }, %V__99 : tensor<i64> { python_test_attrs.static_type = tensor<i64> }, %V__100 : tensor<?x?x?x?xf32> { python_test_attrs.static_type = tensor<1x1x1x1xf32> }, %V__101 : tensor<i64> { python_test_attrs.static_type = tensor<i64> }, %V__102 : tensor<?x?x?x?xf32> { python_test_attrs.static_type = tensor<1x1x1x1xf32> }, %V__103 : tensor<?x?x?x?xi64> { python_test_attrs.static_type = tensor<1x86x1x1xi64> }, %V__104 : tensor<?x?x?x?xi64> { python_test_attrs.static_type = tensor<1x1x1x1xi64> }, %V__105 : tensor<?x?x?x?xi64> { python_test_attrs.static_type = tensor<1x1x1x1xi64> }, %V__106 : tensor<?x?x?x?xf32> { python_test_attrs.static_type = tensor<1x1x1x1xf32> }, %V__107 : tensor<?x?x?x?xi32> { python_test_attrs.static_type = tensor<1x1x1x1xi32> }, %V__108 : tensor<?x?x?x?xf32> { python_test_attrs.static_type = tensor<1x1x1x1xf32> }, %V__109 : tensor<?x?x?x?xi32> { python_test_attrs.static_type = tensor<34x38x1x1xi32> }, %V__110 : tensor<?x?x?x?xi1> { python_test_attrs.static_type = tensor<1x1x1x1xi1> }, %V__111 : tensor<?x?x?x?xi32> { python_test_attrs.static_type = tensor<7x1x1x1xi32> }, %V__112 : tensor<i1> { python_test_attrs.static_type = tensor<i1> }) -> tensor<?x?x?x?xi64> {
  %0 = "tf.RightShift"(%V__0, %V__0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  %1 = "tf.Ceil"(%V__1) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %2 = "tf.Shape"(%1) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<4xi32>
  %dims4 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %4 = "tf.Max"(%V__2, %dims4) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims5 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %5 = "tf.Min"(%4, %dims5) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims6 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %6 = "tf.Prod"(%5, %dims6) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims7 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %7 = "tf.Min"(%6, %dims7) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims8 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %8 = "tf.Prod"(%7, %dims8) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %9 = "tf.Mod"(%0, %8) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims10 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %10 = "tf.Prod"(%9, %dims10) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?xi32>
  %11 = "tf.Mod"(%V__0, %V__0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  %12 = "tf.Const"() { value = dense<[1, 1, 1]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %14 = "tf.FloorDiv"(%V__3, %V__3) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims15 = "tf.Const"() { value = dense<[0, 3, 2, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %15 = "tf.Transpose"(%14, %dims15) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %16 = "tf.Shape"(%15) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<4xi32>
  %17 = "tf.BroadcastTo"(%11, %16) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims18 = "tf.Const"() { value = dense<[0, 1, 3, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %18 = "tf.Transpose"(%17, %dims18) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims19 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %19 = "tf.Mean"(%18, %dims19) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?xi32>
  %20 = "tf.SquaredDifference"(%10, %19) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
  %21 = "tf.Squeeze"(%20) { squeeze_dims = [ 1 : i64 ], device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>) -> tensor<?xi32>
  %dims22 = "tf.Const"() { value = dense<[2, 3, 1, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %22 = "tf.Transpose"(%V__4, %dims22) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<4xi32>) -> tensor<?x?x?x?xi1>
  %23 = "tf.SelectV2"(%22, %V__3, %V__3) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims24 = "tf.Const"() { value = dense<[0, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %24 = "tf.Min"(%23, %dims24) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %dims25 = "tf.Const"() { value = dense<[2, 1, 0, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %25 = "tf.Transpose"(%V__3, %dims25) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %26 = "tf.BiasAdd"(%25, %V__5) { data_format = "NCHW", device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?xi64>) -> tensor<?x?x?x?xi64>
  %dims27 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %27 = "tf.Min"(%26, %dims27) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims28 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %28 = "tf.Min"(%27, %dims28) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?x?xi64>
  %29 = "tf.TruncateDiv"(%24, %28) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims30 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %30 = "tf.Min"(%29, %dims30) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?x?xi64>
  %31 = "tf.Pow"(%V__5, %V__5) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi64>, tensor<?xi64>) -> tensor<?xi64>
  %32 = "tf.Shape"(%V__6) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>) -> tensor<3xi32>
  %dims34 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %34 = "tf.Mean"(%V__3, %dims34) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?x?xi64>
  %dims35 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %35 = "tf.Mean"(%34, %dims35) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?x?xi64>
  %dims36 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %36 = "tf.Min"(%35, %dims36) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?x?xi64>
  %37 = "tf.Shape"(%36) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<4xi32>
  %38 = "tf.BroadcastTo"(%31, %37) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims39 = "tf.Const"() { value = dense<[3, 1, 0, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %39 = "tf.Transpose"(%38, %dims39) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims40 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %40 = "tf.Sum"(%39, %dims40) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?x?xi64>
  %41 = "tf.SquaredDifference"(%30, %40) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims42 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %42 = "tf.Prod"(%41, %dims42) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?x?xi64>
  %43 = "tf.Shape"(%42) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<4xi32>
  %44 = "tf.BroadcastTo"(%21, %43) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims45 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %45 = "tf.Prod"(%44, %dims45) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims46 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %46 = "tf.Mean"(%45, %dims46) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims47 = "tf.Const"() { value = dense<[1, 3, 2, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %47 = "tf.Transpose"(%46, %dims47) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims48 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %48 = "tf.Min"(%V__2, %dims48) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims49 = "tf.Const"() { value = dense<[2, 3, 0, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %49 = "tf.Transpose"(%48, %dims49) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %50 = "tf.Const"() { value = dense<[1, 1, 1, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %dims52 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %52 = "tf.Sum"(%V__2, %dims52) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims53 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %53 = "tf.Prod"(%52, %dims53) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims54 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %54 = "tf.Max"(%53, %dims54) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims55 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %55 = "tf.Prod"(%54, %dims55) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %56 = "tf.Square"(%55) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %57 = "tf.Sub"(%49, %56) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims58 = "tf.Const"() { value = dense<[0, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %58 = "tf.Sum"(%57, %dims58) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims59 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %59 = "tf.Max"(%58, %dims59) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?xi32>
  %60 = "tf.Less"(%V__6, %V__6) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<?x?x?xi64>) -> tensor<?x?x?xi1>
  %61 = "tf.Squeeze"(%60) { squeeze_dims = [ 1 : i64, 2 : i64 ], device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi1>) -> tensor<?xi1>
  %62 = "tf.Div"(%V__7, %V__0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  %dims63 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %63 = "tf.Prod"(%62, %dims63) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?xi32>
  %dims64 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %64 = "tf.Sum"(%V__2, %dims64) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?xi32>
  %65 = "tf.Const"() { value = dense<[1, 1, 1]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %66 = "tf.BroadcastTo"(%64, %65) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>, tensor<3xi32>) -> tensor<?x?x?xi32>
  %67 = "tf.Select"(%61, %63, %66) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi1>, tensor<?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  %68 = "tf.Abs"(%67) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  %69 = "tf.Minimum"(%59, %68) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  %70 = "tf.Const"() { value = dense<[65, 1, 1, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %71 = "tf.BroadcastTo"(%69, %70) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims72 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %72 = "tf.Mean"(%71, %dims72) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims73 = "tf.Const"() { value = dense<[2, 0, 1, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %73 = "tf.Transpose"(%72, %dims73) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims74 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %74 = "tf.Mean"(%73, %dims74) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims75 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %75 = "tf.Sum"(%74, %dims75) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims76 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %76 = "tf.Max"(%75, %dims76) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?xi32>
  %77 = "tf.BiasAdd"(%47, %76) { data_format = "NHWC", device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?xi32>) -> tensor<?x?x?x?xi32>
  %dims78 = "tf.Const"() { value = dense<[2, 1, 3, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %78 = "tf.Transpose"(%77, %dims78) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims79 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %79 = "tf.Max"(%78, %dims79) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %80 = "tf.Relu6"(%79) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims81 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %81 = "tf.Max"(%80, %dims81) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?xi32>
  %dims82 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %82 = "tf.Sum"(%V__8, %dims82) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?xf32>
  %83 = "tf.Expm1"(%82) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %dims84 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %84 = "tf.Mean"(%V__2, %dims84) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %85 = "tf.Shape"(%84) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<4xi32>
  %dims87 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %87 = "tf.Max"(%V__2, %dims87) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims88 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %88 = "tf.Min"(%87, %dims88) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %89 = "tf.Cast"(%88) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xf32>
  %dims90 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %90 = "tf.Mean"(%89, %dims90) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %dims91 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %91 = "tf.Max"(%90, %dims91) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %92 = "tf.Xlog1py"(%83, %91) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims93 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %93 = "tf.Min"(%92, %dims93) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  %94 = "tf.SelectV2"(%V__4, %V__3, %V__3) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %95 = "tf.Const"() { value = dense<[1, 1, 1, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %96 = "tf.BroadcastTo"(%V__6, %95) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims97 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %97 = "tf.Max"(%V__3, %dims97) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %dims98 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %98 = "tf.Max"(%97, %dims98) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %99 = "tf.ClipByValue"(%94, %96, %98) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims100 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %100 = "tf.Prod"(%99, %dims100) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %dims101 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %101 = "tf.Prod"(%100, %dims101) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims102 = "tf.Const"() { value = dense<[1, 3, 2, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %102 = "tf.Transpose"(%101, %dims102) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %103 = "tf.Shape"(%102) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<4xi32>
  %dims105 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %105 = "tf.Mean"(%V__2, %dims105) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %106 = "tf.Cast"(%105) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xf32>
  %dims107 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %107 = "tf.Min"(%106, %dims107) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %dims108 = "tf.Const"() { value = dense<[3, 2, 0, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %108 = "tf.Transpose"(%107, %dims108) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %109 = "tf.Softsign"(%V__1) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %110 = "tf.AddV2"(%109, %V__1) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %111 = "tf.AddV2"(%108, %110) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims112 = "tf.Const"() { value = dense<[0, 2, 3, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %112 = "tf.Transpose"(%111, %dims112) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims113 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %113 = "tf.Sum"(%112, %dims113) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?xf32>
  %dims114 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %114 = "tf.Max"(%113, %dims114) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<1xi32>) -> tensor<?x?xf32>
  %115 = "tf.Invert"(%V__3) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims116 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %116 = "tf.Mean"(%115, %dims116) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %117 = "tf.Mul"(%V__3, %V__3) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %118 = "tf.BitwiseOr"(%116, %117) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims119 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %119 = "tf.Sum"(%118, %dims119) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %120 = "tf.OnesLike"(%119) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims121 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %121 = "tf.Max"(%120, %dims121) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?x?xi64>
  %dims122 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %122 = "tf.Min"(%121, %dims122) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %dims123 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %123 = "tf.Sum"(%122, %dims123) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %124 = "tf.Shape"(%123) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<4xi32>
  %125 = "tf.BroadcastTo"(%114, %124) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %126 = "tf.Square"(%125) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims127 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %127 = "tf.Mean"(%126, %dims127) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %128 = "tf.Shape"(%127) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<4xi32>
  %129 = "tf.BroadcastTo"(%93, %128) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims130 = "tf.Const"() { value = dense<[0, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %130 = "tf.Prod"(%129, %dims130) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %dims131 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %131 = "tf.Mean"(%130, %dims131) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %132 = "tf.Cos"(%131) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims133 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %133 = "tf.Max"(%132, %dims133) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %134 = "tf.Digamma"(%133) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims135 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %135 = "tf.Min"(%V__1, %dims135) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims136 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %136 = "tf.Mean"(%V__1, %dims136) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %137 = "tf.Sub"(%135, %136) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %138 = "tf.Rsqrt"(%137) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims139 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %139 = "tf.Max"(%138, %dims139) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  %dims140 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %140 = "tf.Min"(%139, %dims140) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>, tensor<2xi32>) -> tensor<f32>
  %141 = "tf.Rsqrt"(%140) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>) -> tensor<f32>
  %142 = "tf.Cast"(%141) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>) -> tensor<i1>
  %dims143 = "tf.Const"() { value = dense<[0, 2, 3, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %143 = "tf.Transpose"(%V__9, %dims143) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims144 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %144 = "tf.Prod"(%143, %dims144) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims145 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %145 = "tf.Min"(%144, %dims145) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims146 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %146 = "tf.Mean"(%145, %dims146) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %147 = "tf.BiasAdd"(%V__9, %V__10) { data_format = "NCHW", device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?xi32>) -> tensor<?x?x?x?xi32>
  %148 = "tf.Shape"(%147) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<4xi32>
  %149 = "tf.BroadcastTo"(%146, %148) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims150 = "tf.Const"() { value = dense<[1, 0, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %150 = "tf.Transpose"(%V__9, %dims150) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims151 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %151 = "tf.Min"(%150, %dims151) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<i32>
  %dims152 = "tf.Const"() { value = dense<[0, 3, 2, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %152 = "tf.Transpose"(%V__3, %dims152) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %153 = "tf.Shape"(%152) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<4xi32>
  %154 = "tf.BroadcastTo"(%151, %153) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims155 = "tf.Const"() { value = dense<[2, 0, 3, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %155 = "tf.Transpose"(%154, %dims155) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims156 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %156 = "tf.Prod"(%155, %dims156) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %157 = "tf.Select"(%142, %149, %156) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i1>, tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %158 = "tf.Shape"(%157) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<4xi32>
  %dims159 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %159 = "tf.Prod"(%V__11, %dims159) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %dims160 = "tf.Const"() { value = dense<[0, 2, 3, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %160 = "tf.Transpose"(%159, %dims160) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %161 = "tf.Sinh"(%160) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims162 = "tf.Const"() { value = dense<[0, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %162 = "tf.Max"(%V__12, %dims162) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %dims163 = "tf.Const"() { value = dense<[2, 3, 0, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %163 = "tf.Transpose"(%162, %dims163) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims164 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %164 = "tf.Max"(%163, %dims164) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %165 = "tf.Xdivy"(%161, %164) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %166 = "tf.Shape"(%165) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<4xi32>
  %dims167 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %167 = "tf.Min"(%V__0, %dims167) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<1xi32>) -> tensor<?x?xi32>
  %168 = "tf.Const"() { value = dense<[]> : tensor<0xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<0xi32>
  %169 = "tf.Reshape"(%167, %168) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>, tensor<0xi32>) -> tensor<i32>
  %dims170 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %170 = "tf.Max"(%V__11, %dims170) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?xf32>
  %dims171 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %171 = "tf.Prod"(%170, %dims171) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>, tensor<1xi32>) -> tensor<f32>
  %172 = "tf.Cast"(%171) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>) -> tensor<i32>
  %173 = "tf.TruncateDiv"(%169, %172) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %174 = "tf.Fill"(%166, %173) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<4xi32>, tensor<i32>) -> tensor<?x?x?x?xi32>
  %dims175 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %175 = "tf.Max"(%174, %dims175) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %176 = "tf.Neg"(%175) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims177 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %177 = "tf.Min"(%176, %dims177) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %178 = "tf.Cast"(%177) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi1>
  %dims179 = "tf.Const"() { value = dense<[1, 2, 3, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %179 = "tf.Transpose"(%178, %dims179) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<4xi32>) -> tensor<?x?x?x?xi1>
  %180 = "tf.Cast"(%179) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>) -> tensor<?x?x?x?xf32>
  %dims181 = "tf.Const"() { value = dense<[1, 2, 3, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %181 = "tf.Transpose"(%180, %dims181) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims182 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %182 = "tf.Mean"(%181, %dims182) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims183 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %183 = "tf.Max"(%182, %dims183) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims184 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %184 = "tf.Min"(%183, %dims184) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<f32>
  %185 = "tf.Fill"(%158, %184) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<4xi32>, tensor<f32>) -> tensor<?x?x?x?xf32>
  %186 = "tf.Erfc"(%185) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims187 = "tf.Const"() { value = dense<[1, 2, 3, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %187 = "tf.Transpose"(%186, %dims187) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %188 = "tf.Sqrt"(%187) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims189 = "tf.Const"() { value = dense<[2, 0, 1, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %189 = "tf.Transpose"(%188, %dims189) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims190 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %190 = "tf.Sum"(%189, %dims190) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %191 = "tf.ApproximateEqual"(%134, %190) { tolerance = 1.000000e-06 : f32, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xi1>
  %192 = "tf.Shape"(%191) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>) -> tensor<4xi32>
  %193 = "tf.BroadcastTo"(%81, %192) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims194 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %194 = "tf.Max"(%193, %dims194) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims195 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %195 = "tf.Prod"(%V__9, %dims195) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims196 = "tf.Const"() { value = dense<[0, 3, 1, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %196 = "tf.Transpose"(%195, %dims196) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %197 = "tf.Invert"(%196) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims198 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %198 = "tf.Mean"(%V__9, %dims198) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims199 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %199 = "tf.Max"(%198, %dims199) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims200 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %200 = "tf.Prod"(%199, %dims200) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %201 = "tf.BitwiseAnd"(%197, %200) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims202 = "tf.Const"() { value = dense<[3, 1, 0, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %202 = "tf.Transpose"(%V__2, %dims202) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims203 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %203 = "tf.Prod"(%202, %dims203) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims204 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %204 = "tf.Sum"(%V__2, %dims204) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %205 = "tf.Pow"(%203, %204) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims206 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %206 = "tf.Max"(%205, %dims206) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims207 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %207 = "tf.Mean"(%206, %dims207) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %208 = "tf.LeftShift"(%201, %207) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %209 = "tf.Relu6"(%208) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %210 = "tf.Relu"(%209) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims211 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %211 = "tf.Prod"(%210, %dims211) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims212 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %212 = "tf.Any"(%V__4, %dims212) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<2xi32>) -> tensor<?x?x?x?xi1>
  %dims213 = "tf.Const"() { value = dense<[2, 3, 0, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %213 = "tf.Transpose"(%212, %dims213) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<4xi32>) -> tensor<?x?x?x?xi1>
  %dims214 = "tf.Const"() { value = dense<[1, 0, 3, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %214 = "tf.Transpose"(%V__13, %dims214) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<4xi32>) -> tensor<?x?x?x?xi1>
  %dims215 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %215 = "tf.All"(%214, %dims215) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<1xi32>) -> tensor<?x?x?x?xi1>
  %216 = "tf.LogicalOr"(%213, %215) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<?x?x?x?xi1>) -> tensor<?x?x?x?xi1>
  %217 = "tf.Cast"(%216) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>) -> tensor<?x?x?x?xi32>
  %dims218 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %218 = "tf.Mean"(%217, %dims218) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %219 = "tf.OnesLike"(%218) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %220 = "tf.ClipByValue"(%V__9, %V__9, %V__14) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %221 = "tf.Const"() { value = dense<[1, 79, 1, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %222 = "tf.BroadcastTo"(%220, %221) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims223 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %223 = "tf.Sum"(%222, %dims223) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %224 = "tf.Add"(%219, %223) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims225 = "tf.Const"() { value = dense<[1, 2, 3, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %225 = "tf.Transpose"(%224, %dims225) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %226 = "tf.Cast"(%V__15) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi1>) -> tensor<?x?xi32>
  %dims227 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %227 = "tf.Prod"(%226, %dims227) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>, tensor<1xi32>) -> tensor<?x?xi32>
  %dims228 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %228 = "tf.Max"(%V__16, %dims228) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>, tensor<1xi32>) -> tensor<?x?xi32>
  %229 = "tf.Minimum"(%227, %228) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
  %dims230 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %230 = "tf.Max"(%V__9, %dims230) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims231 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %231 = "tf.Mean"(%230, %dims231) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims232 = "tf.Const"() { value = dense<[2, 3, 0, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %232 = "tf.Transpose"(%231, %dims232) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims233 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %233 = "tf.Min"(%232, %dims233) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %234 = "tf.Shape"(%233) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<4xi32>
  %235 = "tf.BroadcastTo"(%229, %234) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims236 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %236 = "tf.Prod"(%235, %dims236) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims237 = "tf.Const"() { value = dense<[2, 3, 0, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %237 = "tf.Transpose"(%236, %dims237) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims238 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %238 = "tf.Sum"(%237, %dims238) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims239 = "tf.Const"() { value = dense<[0, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %239 = "tf.Min"(%238, %dims239) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims240 = "tf.Const"() { value = dense<[1, 0, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %240 = "tf.Transpose"(%239, %dims240) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims241 = "tf.Const"() { value = dense<[3, 0, 2, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %241 = "tf.Transpose"(%240, %dims241) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims242 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %242 = "tf.Max"(%241, %dims242) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %243 = "tf.ClipByValue"(%211, %225, %242) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims244 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %244 = "tf.Mean"(%243, %dims244) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims245 = "tf.Const"() { value = dense<[3, 2, 0, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %245 = "tf.Transpose"(%244, %dims245) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims246 = "tf.Const"() { value = dense<[2, 1, 0, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %246 = "tf.Transpose"(%245, %dims246) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims247 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %247 = "tf.Max"(%246, %dims247) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims248 = "tf.Const"() { value = dense<[3, 1, 2, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %248 = "tf.Transpose"(%V__4, %dims248) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<4xi32>) -> tensor<?x?x?x?xi1>
  %dims249 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %249 = "tf.Max"(%V__9, %dims249) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims250 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %250 = "tf.Max"(%V__14, %dims250) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %251 = "tf.SelectV2"(%248, %249, %250) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %252 = "tf.Cast"(%V__17) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i32>) -> tensor<f32>
  %253 = "tf.Cast"(%252) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>) -> tensor<i32>
  %dims254 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %254 = "tf.Sum"(%V__1, %dims254) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %255 = "tf.Shape"(%254) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<4xi32>
  %256 = "tf.Reshape"(%253, %255) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %257 = "tf.BitwiseXor"(%251, %256) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims258 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %258 = "tf.Max"(%257, %dims258) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %259 = "tf.Select"(%V__13, %V__2, %V__18) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims260 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %260 = "tf.Mean"(%259, %dims260) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims261 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %261 = "tf.Sum"(%260, %dims261) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims262 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %262 = "tf.Any"(%V__19, %dims262) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi1>, tensor<1xi32>) -> tensor<?xi1>
  %263 = "tf.Select"(%262, %V__14, %V__20) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi1>, tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %264 = "tf.Add"(%261, %263) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims265 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %265 = "tf.Max"(%264, %dims265) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims266 = "tf.Const"() { value = dense<[1, 3, 0, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %266 = "tf.Transpose"(%265, %dims266) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %267 = "tf.Cast"(%266) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %268 = "tf.BitwiseOr"(%258, %267) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims269 = "tf.Const"() { value = dense<[2, 0, 3, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %269 = "tf.Transpose"(%V__18, %dims269) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims270 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %270 = "tf.Min"(%269, %dims270) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims271 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %271 = "tf.Mean"(%V__20, %dims271) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %272 = "tf.BitwiseOr"(%270, %271) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims273 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %273 = "tf.Mean"(%272, %dims273) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims274 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %274 = "tf.Max"(%273, %dims274) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims275 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %275 = "tf.Mean"(%274, %dims275) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims276 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %276 = "tf.Sum"(%275, %dims276) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %277 = "tf.ApproximateEqual"(%V__21, %V__21) { tolerance = 1.000000e-05 : f32, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %dims278 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %278 = "tf.Prod"(%V__2, %dims278) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims279 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %279 = "tf.Mean"(%278, %dims279) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %280 = "tf.Sub"(%V__2, %V__18) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %281 = "tf.Select"(%277, %279, %280) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i1>, tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %282 = "tf.BitwiseOr"(%276, %281) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims283 = "tf.Const"() { value = dense<[0, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %283 = "tf.Min"(%282, %dims283) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims284 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %284 = "tf.Min"(%283, %dims284) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims285 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %285 = "tf.Min"(%284, %dims285) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims286 = "tf.Const"() { value = dense<[0, 1, 3, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %286 = "tf.Transpose"(%285, %dims286) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %287 = "tf.OnesLike"(%286) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims288 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %288 = "tf.Prod"(%287, %dims288) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims289 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %289 = "tf.Min"(%288, %dims289) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims290 = "tf.Const"() { value = dense<[0, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %290 = "tf.Prod"(%289, %dims290) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims291 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %291 = "tf.Sum"(%290, %dims291) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims292 = "tf.Const"() { value = dense<[1, 0, 3, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %292 = "tf.Transpose"(%291, %dims292) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims293 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %293 = "tf.Min"(%292, %dims293) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %294 = "tf.Sub"(%268, %293) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims295 = "tf.Const"() { value = dense<[2, 3, 1, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %295 = "tf.Transpose"(%294, %dims295) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %296 = "tf.Add"(%247, %295) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims297 = "tf.Const"() { value = dense<[3, 2, 1, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %297 = "tf.Transpose"(%296, %dims297) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims298 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %298 = "tf.Mean"(%297, %dims298) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims299 = "tf.Const"() { value = dense<[1, 0, 3, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %299 = "tf.Transpose"(%298, %dims299) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims300 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %300 = "tf.Sum"(%V__20, %dims300) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims301 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %301 = "tf.Max"(%300, %dims301) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims302 = "tf.Const"() { value = dense<[3, 0, 2, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %302 = "tf.Transpose"(%301, %dims302) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims303 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %303 = "tf.Max"(%302, %dims303) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?xi32>
  %304 = "tf.Xdivy"(%V__11, %V__1) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %305 = "tf.Softsign"(%304) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %306 = "tf.Shape"(%305) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<4xi32>
  %307 = "tf.BroadcastTo"(%303, %306) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims308 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %308 = "tf.Sum"(%307, %dims308) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims309 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %309 = "tf.Prod"(%308, %dims309) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims310 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %310 = "tf.Prod"(%309, %dims310) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims311 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %311 = "tf.Sum"(%310, %dims311) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims312 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %312 = "tf.Prod"(%311, %dims312) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %313 = "tf.Maximum"(%V__17, %V__17) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %dims314 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %314 = "tf.Prod"(%V__22, %dims314) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi32>, tensor<1xi32>) -> tensor<i32>
  %315 = "tf.BitwiseOr"(%313, %314) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %316 = "tf.Cast"(%315) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i32>) -> tensor<i32>
  %dims317 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %317 = "tf.Mean"(%V__3, %dims317) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %dims318 = "tf.Const"() { value = dense<[3, 2, 0, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %318 = "tf.Transpose"(%317, %dims318) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims319 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %319 = "tf.Max"(%318, %dims319) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims320 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %320 = "tf.Sum"(%319, %dims320) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?x?xi64>
  %dims321 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %321 = "tf.Sum"(%320, %dims321) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?x?xi64>
  %322 = "tf.Shape"(%321) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<4xi32>
  %323 = "tf.Reshape"(%316, %322) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %324 = "tf.AddV2"(%312, %323) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims325 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %325 = "tf.Prod"(%324, %dims325) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims326 = "tf.Const"() { value = dense<[0, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %326 = "tf.Prod"(%325, %dims326) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %327 = "tf.Cast"(%V__2) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims328 = "tf.Const"() { value = dense<[3, 0, 1, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %328 = "tf.Transpose"(%327, %dims328) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims329 = "tf.Const"() { value = dense<[1, 3, 2, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %329 = "tf.Transpose"(%328, %dims329) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims330 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %330 = "tf.Mean"(%329, %dims330) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims331 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %331 = "tf.Prod"(%V__16, %dims331) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>, tensor<1xi32>) -> tensor<?xi32>
  %332 = "tf.Shape"(%V__11) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<4xi32>
  %334 = "tf.Add"(%330, %331) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?xi32>) -> tensor<?x?x?x?xi32>
  %dims335 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %335 = "tf.Min"(%334, %dims335) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims336 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %336 = "tf.Min"(%335, %dims336) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims337 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %337 = "tf.Min"(%336, %dims337) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?xi32>
  %dims338 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %338 = "tf.Sum"(%337, %dims338) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?xi32>
  %dims339 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %339 = "tf.Min"(%338, %dims339) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?xi32>
  %dims340 = "tf.Const"() { value = dense<[2, 1, 0, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %340 = "tf.Transpose"(%V__2, %dims340) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims341 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %341 = "tf.Mean"(%340, %dims341) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims342 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %342 = "tf.Sum"(%341, %dims342) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?xi32>
  %343 = "tf.Selu"(%V__1) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %344 = "tf.Shape"(%343) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<4xi32>
  %345 = "tf.BroadcastTo"(%342, %344) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims346 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %346 = "tf.Mean"(%345, %dims346) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %347 = "tf.Cast"(%346) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims348 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %348 = "tf.Mean"(%347, %dims348) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims349 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %349 = "tf.Max"(%348, %dims349) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims350 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %350 = "tf.Prod"(%349, %dims350) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?xi32>
  %dims351 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %351 = "tf.Prod"(%350, %dims351) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?xi32>
  %352 = "tf.Shape"(%351) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>) -> tensor<3xi32>
  %353 = "tf.BroadcastTo"(%339, %352) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?xi32>
  %dims354 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %354 = "tf.Sum"(%353, %dims354) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<2xi32>) -> tensor<?xi32>
  %355 = "tf.BiasAdd"(%326, %354) { data_format = "NHWC", device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?xi32>) -> tensor<?x?x?x?xi32>
  %dims356 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %356 = "tf.Mean"(%V__1, %dims356) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %357 = "tf.GreaterEqual"(%356, %V__11) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xi1>
  %dims358 = "tf.Const"() { value = dense<[2, 3, 0, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %358 = "tf.Transpose"(%357, %dims358) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<4xi32>) -> tensor<?x?x?x?xi1>
  %359 = "tf.Const"() { value = dense<[1, 1, 1, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %360 = "tf.Reshape"(%V__17, %359) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims361 = "tf.Const"() { value = dense<[0, 2, 3, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %361 = "tf.Transpose"(%360, %dims361) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims362 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %362 = "tf.Prod"(%361, %dims362) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims363 = "tf.Const"() { value = dense<[1, 0, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %363 = "tf.Transpose"(%V__9, %dims363) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims364 = "tf.Const"() { value = dense<[0, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %364 = "tf.Min"(%363, %dims364) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims365 = "tf.Const"() { value = dense<[1, 2, 3, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %365 = "tf.Transpose"(%364, %dims365) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %366 = "tf.SelectV2"(%358, %362, %365) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %367 = "tf.Square"(%366) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims368 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %368 = "tf.Sum"(%V__9, %dims368) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %369 = "tf.OnesLike"(%368) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims370 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %370 = "tf.Prod"(%V__18, %dims370) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %371 = "tf.FloorDiv"(%369, %370) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims372 = "tf.Const"() { value = dense<[1, 2, 0, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %372 = "tf.Transpose"(%V__18, %dims372) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %373 = "tf.Div"(%372, %V__9) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims374 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %374 = "tf.Mean"(%373, %dims374) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %375 = "tf.BitwiseAnd"(%371, %374) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims376 = "tf.Const"() { value = dense<[0, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %376 = "tf.Mean"(%375, %dims376) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims377 = "tf.Const"() { value = dense<[2, 0, 3, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %377 = "tf.Transpose"(%376, %dims377) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims378 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %378 = "tf.Max"(%377, %dims378) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %379 = "tf.Mul"(%367, %378) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims380 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %380 = "tf.Sum"(%379, %dims380) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims381 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %381 = "tf.Min"(%V__7, %dims381) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?xi32>
  %382 = "tf.Const"() { value = dense<[37, 1, 1, 65]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %383 = "tf.BroadcastTo"(%381, %382) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims384 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %384 = "tf.Sum"(%383, %dims384) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims385 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %385 = "tf.Max"(%384, %dims385) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims386 = "tf.Const"() { value = dense<[3, 1, 2, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %386 = "tf.Transpose"(%385, %dims386) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims387 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %387 = "tf.Prod"(%386, %dims387) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?xi32>
  %388 = "tf.Squeeze"(%V__19) { squeeze_dims = [ 0 : i64 ], device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi1>) -> tensor<i1>
  %389 = "tf.Cast"(%V__5) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi64>) -> tensor<?xi32>
  %390 = "tf.Select"(%388, %389, %V__10) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i1>, tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
  %391 = "tf.Neg"(%390) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi32>) -> tensor<?xi32>
  %392 = "tf.Pow"(%387, %391) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
  %393 = "tf.OnesLike"(%V__0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  %394 = "tf.Abs"(%393) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  %dims395 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %395 = "tf.Sum"(%V__23, %dims395) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?xi64>
  %396 = "tf.Cast"(%395) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>) -> tensor<?x?x?xi32>
  %397 = "tf.Add"(%394, %396) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  %dims398 = "tf.Const"() { value = dense<[1, 3, 0, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %398 = "tf.Transpose"(%V__11, %dims398) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims399 = "tf.Const"() { value = dense<[0, 1, 3, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %399 = "tf.Transpose"(%V__1, %dims399) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %400 = "tf.RealDiv"(%398, %399) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %401 = "tf.Shape"(%400) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<4xi32>
  %402 = "tf.BroadcastTo"(%397, %401) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %403 = "tf.Shape"(%402) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<4xi32>
  %404 = "tf.BroadcastTo"(%392, %403) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims405 = "tf.Const"() { value = dense<[2, 0, 1, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %405 = "tf.Transpose"(%404, %dims405) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %406 = "tf.Div"(%380, %405) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %407 = "tf.BitwiseAnd"(%355, %406) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims408 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %408 = "tf.Prod"(%407, %dims408) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %409 = "tf.Mod"(%299, %408) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %410 = "tf.LeftShift"(%194, %409) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %411 = "tf.Atanh"(%V__8) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %dims412 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %412 = "tf.Sum"(%411, %dims412) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?xf32>
  %dims413 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %413 = "tf.Max"(%412, %dims413) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<1xi32>) -> tensor<?x?xf32>
  %414 = "tf.Sigmoid"(%413) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %415 = "tf.BiasAdd"(%V__18, %V__24) { data_format = "NHWC", device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?xi32>) -> tensor<?x?x?x?xi32>
  %416 = "tf.Shape"(%415) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<4xi32>
  %417 = "tf.BroadcastTo"(%414, %416) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %418 = "tf.IsNan"(%417) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xi1>
  %419 = "tf.LogicalNot"(%418) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>) -> tensor<?x?x?x?xi1>
  %420 = "tf.Cast"(%419) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>) -> tensor<?x?x?x?xf32>
  %421 = "tf.Cast"(%420) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xi32>
  %422 = "tf.Add"(%V__16, %V__16) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
  %dims423 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %423 = "tf.Sum"(%V__2, %dims423) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %424 = "tf.Squeeze"(%423) { squeeze_dims = [ 0 : i64, 1 : i64 ], device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?xi32>
  %425 = "tf.BitwiseAnd"(%422, %424) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
  %426 = "tf.BitwiseOr"(%V__2, %V__18) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims427 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %427 = "tf.Max"(%426, %dims427) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims428 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %428 = "tf.Prod"(%427, %dims428) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %429 = "tf.Shape"(%428) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<4xi32>
  %431 = "tf.Sub"(%421, %425) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims432 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %432 = "tf.Max"(%431, %dims432) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims433 = "tf.Const"() { value = dense<[2, 1, 0]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %433 = "tf.Transpose"(%V__7, %dims433) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?xi32>
  %dims434 = "tf.Const"() { value = dense<[2, 0, 1]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %434 = "tf.Transpose"(%V__7, %dims434) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?xi32>
  %435 = "tf.BitwiseXor"(%433, %434) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  %dims436 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %436 = "tf.Prod"(%435, %dims436) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<2xi32>) -> tensor<?xi32>
  %437 = "tf.Neg"(%V__22) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi32>) -> tensor<?xi32>
  %dims438 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %438 = "tf.Sum"(%437, %dims438) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi32>, tensor<1xi32>) -> tensor<i32>
  %439 = "tf.Shape"(%V__19) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi1>) -> tensor<1xi32>
  %441 = "tf.Minimum"(%436, %438) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi32>, tensor<i32>) -> tensor<?xi32>
  %442 = "tf.Const"() { value = dense<[1, 23, 1, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %443 = "tf.BroadcastTo"(%V__3, %442) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims444 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %444 = "tf.Min"(%443, %dims444) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?x?xi64>
  %445 = "tf.Const"() { value = dense<[1, 1, 1, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %446 = "tf.BroadcastTo"(%V__3, %445) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims447 = "tf.Const"() { value = dense<[1, 2, 0, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %447 = "tf.Transpose"(%446, %dims447) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %448 = "tf.FloorDiv"(%444, %447) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims449 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %449 = "tf.Mean"(%448, %dims449) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %dims450 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %450 = "tf.Max"(%449, %dims450) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims451 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %451 = "tf.Mean"(%450, %dims451) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %452 = "tf.Shape"(%451) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<4xi32>
  %453 = "tf.BroadcastTo"(%441, %452) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims454 = "tf.Const"() { value = dense<[1, 0, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %454 = "tf.Transpose"(%453, %dims454) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %455 = "tf.Relu6"(%454) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %456 = "tf.FloorDiv"(%432, %455) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims457 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %457 = "tf.Sum"(%456, %dims457) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %458 = "tf.Abs"(%457) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims459 = "tf.Const"() { value = dense<[1, 2, 0, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %459 = "tf.Transpose"(%458, %dims459) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims460 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %460 = "tf.Min"(%459, %dims460) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims461 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %461 = "tf.Min"(%460, %dims461) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims462 = "tf.Const"() { value = dense<[1, 0, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %462 = "tf.Transpose"(%461, %dims462) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims463 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %463 = "tf.Max"(%V__6, %dims463) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?xi64>
  %dims464 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %464 = "tf.Max"(%463, %dims464) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?xi64>
  %dims465 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %465 = "tf.Sum"(%464, %dims465) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<1xi32>) -> tensor<?x?xi64>
  %466 = "tf.Squeeze"(%V__3) { squeeze_dims = [ 0 : i64, 1 : i64 ], device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<?x?xi64>
  %467 = "tf.Mod"(%466, %V__25) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi64>, tensor<?x?xi64>) -> tensor<?x?xi64>
  %468 = "tf.GreaterEqual"(%465, %467) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi64>, tensor<?x?xi64>) -> tensor<?x?xi1>
  %469 = "tf.Squeeze"(%468) { squeeze_dims = [ 1 : i64 ], device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi1>) -> tensor<?xi1>
  %dims470 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %470 = "tf.Sum"(%V__1, %dims470) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %dims471 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %471 = "tf.Prod"(%470, %dims471) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?xf32>
  %472 = "tf.Cos"(%471) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %dims473 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %473 = "tf.Sum"(%V__7, %dims473) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?xi32>
  %474 = "tf.Cast"(%473) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>) -> tensor<?x?x?xf32>
  %dims475 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %475 = "tf.Sum"(%474, %dims475) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?xf32>
  %476 = "tf.SquaredDifference"(%472, %475) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %477 = "tf.IsNan"(%476) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xi1>
  %478 = "tf.Squeeze"(%V__3) { squeeze_dims = [ 0 : i64, 1 : i64 ], device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<?x?xi64>
  %479 = "tf.Cast"(%478) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi64>) -> tensor<?x?xi1>
  %480 = "tf.Cast"(%479) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi1>) -> tensor<?x?xi1>
  %481 = "tf.Add"(%V__8, %V__8) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %482 = "tf.Shape"(%481) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<3xi32>
  %483 = "tf.BroadcastTo"(%480, %482) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi1>, tensor<3xi32>) -> tensor<?x?x?xi1>
  %484 = "tf.Select"(%469, %477, %483) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi1>, tensor<?x?x?xi1>, tensor<?x?x?xi1>) -> tensor<?x?x?xi1>
  %485 = "tf.Neg"(%V__12) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %486 = "tf.Asin"(%485) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %487 = "tf.Tanh"(%486) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims488 = "tf.Const"() { value = dense<[0, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %488 = "tf.Mean"(%487, %dims488) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %dims489 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %489 = "tf.Min"(%488, %dims489) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims490 = "tf.Const"() { value = dense<[1, 3, 2, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %490 = "tf.Transpose"(%V__12, %dims490) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %491 = "tf.Square"(%V__1) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %492 = "tf.Sub"(%490, %491) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %493 = "tf.DivNoNan"(%489, %492) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims494 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %494 = "tf.Max"(%493, %dims494) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %495 = "tf.Cosh"(%494) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %496 = "tf.Squeeze"(%V__4) { squeeze_dims = [ 2 : i64 ], device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>) -> tensor<?x?x?xi1>
  %497 = "tf.Cast"(%496) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi1>) -> tensor<?x?x?xf32>
  %dims498 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %498 = "tf.Prod"(%V__8, %dims498) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?xf32>
  %499 = "tf.NextAfter"(%497, %498) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %dims500 = "tf.Const"() { value = dense<[1, 2, 3, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %500 = "tf.Transpose"(%V__1, %dims500) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims501 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %501 = "tf.Sum"(%500, %dims501) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %502 = "tf.Cast"(%501) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xi1>
  %dims503 = "tf.Const"() { value = dense<[2, 1, 0, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %503 = "tf.Transpose"(%502, %dims503) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<4xi32>) -> tensor<?x?x?x?xi1>
  %504 = "tf.Shape"(%503) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>) -> tensor<4xi32>
  %506 = "tf.Polygamma"(%495, %499) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %507 = "tf.Shape"(%506) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<4xi32>
  %508 = "tf.BroadcastTo"(%484, %507) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi1>, tensor<4xi32>) -> tensor<?x?x?x?xi1>
  %dims509 = "tf.Const"() { value = dense<[1, 0, 3, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %509 = "tf.Transpose"(%508, %dims509) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<4xi32>) -> tensor<?x?x?x?xi1>
  %dims510 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %510 = "tf.Any"(%509, %dims510) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<2xi32>) -> tensor<?x?x?x?xi1>
  %511 = "tf.Cast"(%510) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>) -> tensor<?x?x?x?xi32>
  %dims512 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %512 = "tf.Mean"(%511, %dims512) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims513 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %513 = "tf.Sum"(%512, %dims513) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims514 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %514 = "tf.Min"(%513, %dims514) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %515 = "tf.AddV2"(%462, %514) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims516 = "tf.Const"() { value = dense<[2, 1, 3, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %516 = "tf.Transpose"(%515, %dims516) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims517 = "tf.Const"() { value = dense<[3, 2, 1, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %517 = "tf.Transpose"(%516, %dims517) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %518 = "tf.RightShift"(%V__26, %V__26) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %519 = "tf.Shape"(%V__6) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>) -> tensor<3xi32>
  %520 = "tf.Reshape"(%518, %519) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i32>, tensor<3xi32>) -> tensor<?x?x?xi32>
  %521 = "tf.Neg"(%520) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  %dims522 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %522 = "tf.Sum"(%521, %dims522) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?xi32>
  %dims523 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %523 = "tf.Prod"(%522, %dims523) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?xi32>
  %524 = "tf.Const"() { value = dense<[1, 22, 1, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %525 = "tf.BroadcastTo"(%V__16, %524) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims526 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %526 = "tf.Prod"(%525, %dims526) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims527 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %527 = "tf.Mean"(%526, %dims527) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims528 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %528 = "tf.Sum"(%527, %dims528) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims529 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %529 = "tf.Prod"(%528, %dims529) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?xi32>
  %dims530 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %530 = "tf.Min"(%529, %dims530) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?xi32>
  %531 = "tf.Invert"(%530) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  %532 = "tf.TruncateDiv"(%523, %531) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  %dims533 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %533 = "tf.Min"(%532, %dims533) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?xi32>
  %dims534 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %534 = "tf.Prod"(%533, %dims534) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<2xi32>) -> tensor<?xi32>
  %dims535 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %535 = "tf.Prod"(%V__27, %dims535) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %536 = "tf.Softsign"(%535) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %537 = "tf.Exp"(%536) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %538 = "tf.Sin"(%V__1) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims539 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %539 = "tf.Sum"(%538, %dims539) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %dims540 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %540 = "tf.Max"(%539, %dims540) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %541 = "tf.FloorMod"(%537, %540) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %542 = "tf.Elu"(%541) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims543 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %543 = "tf.Min"(%V__27, %dims543) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims544 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %544 = "tf.Prod"(%543, %dims544) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %545 = "tf.Square"(%544) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims546 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %546 = "tf.Sum"(%545, %dims546) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %547 = "tf.Abs"(%546) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims548 = "tf.Const"() { value = dense<[0, 2, 1, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %548 = "tf.Transpose"(%547, %dims548) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims549 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %549 = "tf.Min"(%548, %dims549) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %dims550 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %550 = "tf.Mean"(%549, %dims550) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %551 = "tf.DivNoNan"(%542, %550) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %552 = "tf.Shape"(%551) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<4xi32>
  %553 = "tf.BroadcastTo"(%534, %552) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims554 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %554 = "tf.Mean"(%553, %dims554) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %555 = "tf.Const"() { value = dense<[1, 1, 1, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %556 = "tf.BroadcastTo"(%V__8, %555) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %557 = "tf.Log"(%556) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims558 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %558 = "tf.Min"(%557, %dims558) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims559 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %559 = "tf.Max"(%558, %dims559) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims560 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %560 = "tf.Max"(%559, %dims560) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %561 = "tf.Xlogy"(%V__28, %V__28) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims562 = "tf.Const"() { value = dense<[3, 2, 1, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %562 = "tf.Transpose"(%V__28, %dims562) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %563 = "tf.Mul"(%561, %562) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims564 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %564 = "tf.Min"(%563, %dims564) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %565 = "tf.ApproximateEqual"(%560, %564) { tolerance = 1.000000e-06 : f32, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xi1>
  %dims566 = "tf.Const"() { value = dense<[2, 0, 3, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %566 = "tf.Transpose"(%V__20, %dims566) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims567 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %567 = "tf.Sum"(%566, %dims567) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims568 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %568 = "tf.Max"(%567, %dims568) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims569 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %569 = "tf.Sum"(%568, %dims569) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims570 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %570 = "tf.Sum"(%569, %dims570) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims571 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %571 = "tf.Min"(%570, %dims571) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims572 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %572 = "tf.Sum"(%V__18, %dims572) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims573 = "tf.Const"() { value = dense<[0, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %573 = "tf.Max"(%572, %dims573) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims574 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %574 = "tf.Min"(%V__9, %dims574) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %575 = "tf.Minimum"(%573, %574) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %576 = "tf.Div"(%571, %575) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims577 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %577 = "tf.Prod"(%V__3, %dims577) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims578 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %578 = "tf.Sum"(%577, %dims578) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims579 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %579 = "tf.Min"(%578, %dims579) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims580 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %580 = "tf.Min"(%579, %dims580) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims581 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %581 = "tf.Min"(%580, %dims581) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %582 = "tf.Cast"(%581) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi32>
  %dims583 = "tf.Const"() { value = dense<[3, 1, 2, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %583 = "tf.Transpose"(%V__18, %dims583) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims584 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %584 = "tf.Mean"(%583, %dims584) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims585 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %585 = "tf.Mean"(%584, %dims585) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %586 = "tf.Sign"(%585) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims587 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %587 = "tf.Prod"(%586, %dims587) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %588 = "tf.FloorMod"(%582, %587) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %589 = "tf.Select"(%565, %576, %588) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %590 = "tf.AddV2"(%V__0, %V__7) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  %dims591 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %591 = "tf.Prod"(%V__28, %dims591) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %592 = "tf.Shape"(%591) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<4xi32>
  %593 = "tf.BroadcastTo"(%590, %592) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims594 = "tf.Const"() { value = dense<[0, 2, 3, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %594 = "tf.Transpose"(%593, %dims594) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %595 = "tf.Mul"(%V__20, %V__20) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims596 = "tf.Const"() { value = dense<[0, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %596 = "tf.Prod"(%595, %dims596) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims597 = "tf.Const"() { value = dense<[1, 2, 3, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %597 = "tf.Transpose"(%596, %dims597) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims598 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %598 = "tf.Mean"(%597, %dims598) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims599 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %599 = "tf.Max"(%598, %dims599) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims600 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %600 = "tf.Prod"(%599, %dims600) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %601 = "tf.RightShift"(%594, %600) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims602 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %602 = "tf.Sum"(%601, %dims602) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims603 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %603 = "tf.Mean"(%602, %dims603) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims604 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %604 = "tf.Sum"(%603, %dims604) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims605 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %605 = "tf.Sum"(%604, %dims605) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?xi32>
  %dims606 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %606 = "tf.Sum"(%V__18, %dims606) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims607 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %607 = "tf.Sum"(%606, %dims607) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %608 = "tf.IsNan"(%V__29) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xi1>
  %609 = "tf.Shape"(%608) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>) -> tensor<4xi32>
  %dims611 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %611 = "tf.Mean"(%V__18, %dims611) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims612 = "tf.Const"() { value = dense<[1, 0, 3, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %612 = "tf.Transpose"(%611, %dims612) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %613 = "tf.Shape"(%V__18) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<4xi32>
  %615 = "tf.Mod"(%607, %612) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims616 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %616 = "tf.Sum"(%615, %dims616) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims617 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %617 = "tf.Mean"(%616, %dims617) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims618 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %618 = "tf.Prod"(%617, %dims618) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims619 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %619 = "tf.Prod"(%618, %dims619) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims620 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %620 = "tf.Prod"(%619, %dims620) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims621 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %621 = "tf.Mean"(%620, %dims621) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %622 = "tf.Shape"(%621) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<4xi32>
  %623 = "tf.BroadcastTo"(%605, %622) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims624 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %624 = "tf.Prod"(%623, %dims624) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %625 = "tf.ClipByValue"(%554, %589, %624) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims626 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %626 = "tf.Min"(%625, %dims626) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %627 = "tf.SquaredDifference"(%517, %626) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %628 = "tf.Log1p"(%V__27) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims629 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %629 = "tf.Min"(%628, %dims629) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims630 = "tf.Const"() { value = dense<[0, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %630 = "tf.Max"(%629, %dims630) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %dims631 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %631 = "tf.Max"(%630, %dims631) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %632 = "tf.Shape"(%631) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<4xi32>
  %633 = "tf.Div"(%V__17, %V__26) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %dims634 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %634 = "tf.Sum"(%V__16, %dims634) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>, tensor<2xi32>) -> tensor<i32>
  %635 = "tf.AddV2"(%633, %634) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %636 = "tf.Fill"(%632, %635) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<4xi32>, tensor<i32>) -> tensor<?x?x?x?xi32>
  %dims637 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %637 = "tf.Mean"(%636, %dims637) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims638 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %638 = "tf.Prod"(%637, %dims638) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims639 = "tf.Const"() { value = dense<[1, 3, 2, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %639 = "tf.Transpose"(%V__2, %dims639) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %640 = "tf.Const"() { value = dense<[1, 1, 1, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %641 = "tf.BroadcastTo"(%639, %640) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims642 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %642 = "tf.Prod"(%V__9, %dims642) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %643 = "tf.Cast"(%642) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims644 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %644 = "tf.Sum"(%643, %dims644) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims645 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %645 = "tf.Mean"(%V__14, %dims645) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %646 = "tf.TruncateDiv"(%645, %V__18) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %647 = "tf.ClipByValue"(%641, %644, %646) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims648 = "tf.Const"() { value = dense<[3, 0, 1, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %648 = "tf.Transpose"(%647, %dims648) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %649 = "tf.Shape"(%648) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<4xi32>
  %651 = "tf.Const"() { value = dense<[13, 1, 71, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %652 = "tf.BroadcastTo"(%V__16, %651) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims653 = "tf.Const"() { value = dense<[0, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %653 = "tf.Mean"(%652, %dims653) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims654 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %654 = "tf.Prod"(%653, %dims654) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims655 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %655 = "tf.Mean"(%654, %dims655) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims656 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %656 = "tf.Min"(%V__20, %dims656) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?xi32>
  %657 = "tf.Relu6"(%656) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi32>) -> tensor<?xi32>
  %dims658 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %658 = "tf.Prod"(%V__0, %dims658) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<2xi32>) -> tensor<?xi32>
  %659 = "tf.BitwiseXor"(%657, %658) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
  %660 = "tf.BiasAdd"(%655, %659) { data_format = "NHWC", device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?xi32>) -> tensor<?x?x?x?xi32>
  %dims661 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %661 = "tf.Mean"(%660, %dims661) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?xi32>
  %662 = "tf.Squeeze"(%V__11) { squeeze_dims = [ 1 : i64, 2 : i64 ], device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?xf32>
  %663 = "tf.Cosh"(%662) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %dims664 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %664 = "tf.Sum"(%V__27, %dims664) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  %665 = "tf.Pow"(%663, %664) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %dims666 = "tf.Const"() { value = dense<[2, 1, 0, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %666 = "tf.Transpose"(%V__3, %dims666) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %667 = "tf.Pow"(%666, %V__3) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %668 = "tf.Shape"(%667) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<4xi32>
  %669 = "tf.BroadcastTo"(%665, %668) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims670 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %670 = "tf.Sum"(%669, %dims670) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %671 = "tf.Shape"(%670) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<4xi32>
  %672 = "tf.BroadcastTo"(%661, %671) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims673 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %673 = "tf.Sum"(%672, %dims673) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims674 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %674 = "tf.Mean"(%673, %dims674) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %675 = "tf.BitwiseAnd"(%638, %674) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %676 = "tf.Relu"(%675) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims677 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %677 = "tf.Min"(%676, %dims677) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims678 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %678 = "tf.Prod"(%677, %dims678) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims679 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %679 = "tf.Max"(%V__2, %dims679) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims680 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %680 = "tf.Max"(%679, %dims680) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims681 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %681 = "tf.Max"(%680, %dims681) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims682 = "tf.Const"() { value = dense<[3, 2, 0, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %682 = "tf.Transpose"(%681, %dims682) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %683 = "tf.Const"() { value = dense<[1, 1, 1, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %684 = "tf.BroadcastTo"(%V__14, %683) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %685 = "tf.OnesLike"(%684) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %686 = "tf.Div"(%682, %685) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %687 = "tf.Const"() { value = dense<[47, 8, 10, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %688 = "tf.BroadcastTo"(%V__22, %687) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims689 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %689 = "tf.Min"(%688, %dims689) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?xi32>
  %690 = "tf.Const"() { value = dense<[72, 73, 1, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %691 = "tf.BroadcastTo"(%V__16, %690) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims692 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %692 = "tf.Mean"(%691, %dims692) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?xi32>
  %693 = "tf.Pow"(%689, %692) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
  %694 = "tf.Relu"(%693) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi32>) -> tensor<?xi32>
  %695 = "tf.BiasAdd"(%686, %694) { data_format = "NHWC", device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?xi32>) -> tensor<?x?x?x?xi32>
  %696 = "tf.Relu"(%695) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %697 = "tf.Square"(%696) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims698 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %698 = "tf.Prod"(%697, %dims698) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims699 = "tf.Const"() { value = dense<[3, 1, 0, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %699 = "tf.Transpose"(%698, %dims699) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims700 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %700 = "tf.Prod"(%699, %dims700) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims701 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %701 = "tf.Prod"(%700, %dims701) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?xi32>
  %dims702 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %702 = "tf.Sum"(%701, %dims702) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?xi32>
  %dims703 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %703 = "tf.Prod"(%702, %dims703) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?xi32>
  %dims704 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %704 = "tf.Mean"(%703, %dims704) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?xi32>
  %dims705 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %705 = "tf.Max"(%704, %dims705) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?xi32>
  %dims706 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %706 = "tf.Sum"(%705, %dims706) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?xi32>
  %dims707 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %707 = "tf.Prod"(%V__2, %dims707) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %708 = "tf.Maximum"(%707, %V__14) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims709 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %709 = "tf.Prod"(%708, %dims709) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims710 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %710 = "tf.Sum"(%709, %dims710) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?xi32>
  %711 = "tf.Equal"(%V__8, %V__8) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xi1>
  %712 = "tf.LogicalNot"(%711) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi1>) -> tensor<?x?x?xi1>
  %713 = "tf.Shape"(%712) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi1>) -> tensor<3xi32>
  %714 = "tf.BroadcastTo"(%710, %713) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi32>, tensor<3xi32>) -> tensor<?x?x?xi32>
  %dims715 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %715 = "tf.Max"(%714, %dims715) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?xi32>
  %dims716 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %716 = "tf.Min"(%715, %dims716) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?xi32>
  %dims717 = "tf.Const"() { value = dense<[1, 2, 0]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %717 = "tf.Transpose"(%716, %dims717) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?xi32>
  %718 = "tf.Const"() { value = dense<[89, 1, 1]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %719 = "tf.BroadcastTo"(%717, %718) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?xi32>
  %dims720 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %720 = "tf.Max"(%719, %dims720) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?xi32>
  %721 = "tf.Mod"(%706, %720) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  %dims722 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %722 = "tf.Sum"(%721, %dims722) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<2xi32>) -> tensor<?xi32>
  %723 = "tf.BiasAdd"(%678, %722) { data_format = "NCHW", device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?xi32>) -> tensor<?x?x?x?xi32>
  %dims724 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %724 = "tf.Mean"(%723, %dims724) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims725 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %725 = "tf.Max"(%724, %dims725) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims726 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %726 = "tf.Max"(%V__7, %dims726) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<3xi32>) -> tensor<i32>
  %727 = "tf.Cast"(%726) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i32>) -> tensor<i32>
  %728 = "tf.Squeeze"(%V__30) { squeeze_dims = [ 0 : i64 ], device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi32>) -> tensor<i32>
  %729 = "tf.BitwiseAnd"(%727, %728) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %730 = "tf.Relu"(%729) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i32>) -> tensor<i32>
  %dims731 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %731 = "tf.Mean"(%V__27, %dims731) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %732 = "tf.Elu"(%731) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims733 = "tf.Const"() { value = dense<[0, 2, 3, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %733 = "tf.Transpose"(%732, %dims733) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims734 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %734 = "tf.Max"(%733, %dims734) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %735 = "tf.Shape"(%734) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<4xi32>
  %736 = "tf.Reshape"(%730, %735) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims737 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %737 = "tf.Sum"(%736, %dims737) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims738 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %738 = "tf.Min"(%V__3, %dims738) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims739 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %739 = "tf.Prod"(%V__23, %dims739) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %740 = "tf.TruncateDiv"(%738, %739) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %741 = "tf.Invert"(%740) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %742 = "tf.Cast"(%V__28) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xi64>
  %dims743 = "tf.Const"() { value = dense<[2, 3, 1, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %743 = "tf.Transpose"(%V__23, %dims743) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %744 = "tf.Sub"(%742, %743) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims745 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %745 = "tf.Min"(%744, %dims745) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %746 = "tf.AddV2"(%741, %745) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %747 = "tf.Shape"(%746) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<4xi32>
  %748 = "tf.BroadcastTo"(%737, %747) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %749 = "tf.Squeeze"(%748) { squeeze_dims = [ 1 : i64 ], device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?xi32>
  %dims750 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %750 = "tf.Prod"(%749, %dims750) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?xi32>
  %dims751 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %751 = "tf.Sum"(%750, %dims751) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?xi32>
  %dims752 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %752 = "tf.Prod"(%V__27, %dims752) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %dims753 = "tf.Const"() { value = dense<[2, 0, 1, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %753 = "tf.Transpose"(%752, %dims753) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %754 = "tf.Tanh"(%753) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims755 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %755 = "tf.Min"(%V__27, %dims755) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %756 = "tf.Sin"(%755) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %757 = "tf.Xlogy"(%754, %756) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims758 = "tf.Const"() { value = dense<[0, 2, 3, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %758 = "tf.Transpose"(%757, %dims758) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %759 = "tf.Sigmoid"(%758) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %760 = "tf.Relu6"(%759) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %761 = "tf.BiasAdd"(%V__28, %V__31) { data_format = "NCHW", device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?xf32>) -> tensor<?x?x?x?xf32>
  %762 = "tf.Elu"(%761) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims763 = "tf.Const"() { value = dense<[2, 1, 0, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %763 = "tf.Transpose"(%V__27, %dims763) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %764 = "tf.Sub"(%763, %V__28) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %765 = "tf.FloorDiv"(%762, %764) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims766 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %766 = "tf.Sum"(%765, %dims766) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %767 = "tf.Acosh"(%766) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %768 = "tf.FloorMod"(%760, %767) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims769 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %769 = "tf.Sum"(%768, %dims769) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %dims770 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %770 = "tf.Sum"(%769, %dims770) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %dims771 = "tf.Const"() { value = dense<[1, 0, 3, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %771 = "tf.Transpose"(%770, %dims771) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims772 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %772 = "tf.Sum"(%771, %dims772) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %dims773 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %773 = "tf.Mean"(%772, %dims773) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %774 = "tf.Rint"(%773) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims775 = "tf.Const"() { value = dense<[0, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %775 = "tf.Max"(%774, %dims775) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %dims776 = "tf.Const"() { value = dense<[2, 0, 3, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %776 = "tf.Transpose"(%775, %dims776) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims777 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %777 = "tf.Min"(%776, %dims777) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %778 = "tf.Shape"(%777) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<4xi32>
  %dims780 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %780 = "tf.Sum"(%V__27, %dims780) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %dims781 = "tf.Const"() { value = dense<[2, 1, 3, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %781 = "tf.Transpose"(%V__12, %dims781) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %782 = "tf.LessEqual"(%780, %781) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xi1>
  %783 = "tf.Add"(%V__20, %V__14) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims784 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %784 = "tf.Min"(%783, %dims784) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims785 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %785 = "tf.Min"(%V__7, %dims785) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?xi32>
  %786 = "tf.Const"() { value = dense<[1, 1, 1, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %788 = "tf.SelectV2"(%782, %784, %785) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<?x?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims789 = "tf.Const"() { value = dense<[2, 0, 1, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %789 = "tf.Transpose"(%788, %dims789) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims790 = "tf.Const"() { value = dense<[2, 3, 0, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %790 = "tf.Transpose"(%789, %dims790) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims791 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %791 = "tf.Max"(%V__9, %dims791) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims792 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %792 = "tf.Mean"(%791, %dims792) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims793 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %793 = "tf.Min"(%V__9, %dims793) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %794 = "tf.BitwiseXor"(%792, %793) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims795 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %795 = "tf.Sum"(%794, %dims795) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims796 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %796 = "tf.Mean"(%795, %dims796) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims797 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %797 = "tf.Max"(%796, %dims797) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %798 = "tf.Neg"(%797) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims799 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %799 = "tf.Mean"(%798, %dims799) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims800 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %800 = "tf.Min"(%799, %dims800) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims801 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %801 = "tf.Prod"(%800, %dims801) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims802 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %802 = "tf.Prod"(%801, %dims802) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %803 = "tf.Shape"(%802) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<4xi32>
  %805 = "tf.Const"() { value = dense<[68, 64, 1, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %806 = "tf.BroadcastTo"(%V__0, %805) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %807 = "tf.Sign"(%806) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims808 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %808 = "tf.Min"(%807, %dims808) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims809 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %809 = "tf.Min"(%808, %dims809) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims810 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %810 = "tf.Mean"(%809, %dims810) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims811 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %811 = "tf.Sum"(%810, %dims811) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims812 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %812 = "tf.Sum"(%811, %dims812) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims813 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %813 = "tf.Min"(%812, %dims813) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims814 = "tf.Const"() { value = dense<[3, 0, 1, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %814 = "tf.Transpose"(%V__32, %dims814) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims815 = "tf.Const"() { value = dense<[0, 2, 1, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %815 = "tf.Transpose"(%814, %dims815) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %816 = "tf.Square"(%V__33) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi32>) -> tensor<?xi32>
  %817 = "tf.BiasAdd"(%815, %816) { data_format = "NHWC", device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?xi32>) -> tensor<?x?x?x?xi32>
  %818 = "tf.Sign"(%817) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %819 = "tf.Neg"(%818) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims820 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %820 = "tf.Sum"(%819, %dims820) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims821 = "tf.Const"() { value = dense<[3, 1, 2, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %821 = "tf.Transpose"(%820, %dims821) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims822 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %822 = "tf.Prod"(%V__0, %dims822) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?xi32>
  %dims823 = "tf.Const"() { value = dense<[2, 0, 1]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %823 = "tf.Transpose"(%822, %dims823) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?xi32>
  %dims824 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %824 = "tf.Sum"(%823, %dims824) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<1xi32>) -> tensor<?x?xi32>
  %dims825 = "tf.Const"() { value = dense<[2, 3, 1, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %825 = "tf.Transpose"(%V__23, %dims825) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %826 = "tf.Shape"(%825) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<4xi32>
  %827 = "tf.BroadcastTo"(%824, %826) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims828 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %828 = "tf.Min"(%827, %dims828) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims829 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %829 = "tf.Prod"(%828, %dims829) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %830 = "tf.ClipByValue"(%813, %821, %829) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims831 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %831 = "tf.Mean"(%830, %dims831) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %832 = "tf.Mod"(%790, %831) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims833 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %833 = "tf.Prod"(%832, %dims833) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %834 = "tf.RightShift"(%751, %833) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %835 = "tf.Add"(%725, %834) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %836 = "tf.Mod"(%627, %835) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims837 = "tf.Const"() { value = dense<[2, 0, 3, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %837 = "tf.Transpose"(%836, %dims837) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims838 = "tf.Const"() { value = dense<[1, 3, 0, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %838 = "tf.Transpose"(%837, %dims838) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims839 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %839 = "tf.Mean"(%838, %dims839) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims840 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %840 = "tf.Mean"(%839, %dims840) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims841 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %841 = "tf.Mean"(%840, %dims841) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims842 = "tf.Const"() { value = dense<[2, 3, 0, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %842 = "tf.Transpose"(%841, %dims842) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims843 = "tf.Const"() { value = dense<[0, 3, 1, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %843 = "tf.Transpose"(%842, %dims843) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims844 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %844 = "tf.Sum"(%843, %dims844) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims845 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %845 = "tf.Sum"(%844, %dims845) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims846 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %846 = "tf.Prod"(%845, %dims846) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %847 = "tf.Equal"(%410, %846) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi1>
  %dims848 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %848 = "tf.All"(%847, %dims848) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<2xi32>) -> tensor<?x?x?x?xi1>
  %849 = "tf.ZerosLike"(%848) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>) -> tensor<?x?x?x?xi1>
  %dims850 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %850 = "tf.All"(%849, %dims850) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<2xi32>) -> tensor<?x?x?x?xi1>
  %dims851 = "tf.Const"() { value = dense<[1, 0, 3, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %851 = "tf.Transpose"(%850, %dims851) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<4xi32>) -> tensor<?x?x?x?xi1>
  %dims852 = "tf.Const"() { value = dense<[0, 2, 1, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %852 = "tf.Transpose"(%851, %dims852) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<4xi32>) -> tensor<?x?x?x?xi1>
  %853 = "tf.Squeeze"(%852) { squeeze_dims = [ 0 : i64, 1 : i64, 2 : i64, 3 : i64 ], device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>) -> tensor<i1>
  %854 = "tf.Square"(%V__20) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims855 = "tf.Const"() { value = dense<[1, 2, 3, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %855 = "tf.Transpose"(%V__20, %dims855) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %856 = "tf.Pow"(%854, %855) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %857 = "tf.Squeeze"(%856) { squeeze_dims = [ 0 : i64, 3 : i64 ], device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?xi32>
  %858 = "tf.Relu6"(%857) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>) -> tensor<?x?xi32>
  %dims859 = "tf.Const"() { value = dense<[2, 0, 1]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %859 = "tf.Transpose"(%V__0, %dims859) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?xi32>
  %860 = "tf.ZerosLike"(%V__0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  %861 = "tf.TruncateDiv"(%859, %860) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  %dims862 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %862 = "tf.Min"(%861, %dims862) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<1xi32>) -> tensor<?x?xi32>
  %863 = "tf.BitwiseXor"(%858, %862) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
  %dims864 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %864 = "tf.Max"(%863, %dims864) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>, tensor<2xi32>) -> tensor<?x?xi32>
  %865 = "tf.RightShift"(%V__9, %V__20) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims866 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %866 = "tf.Prod"(%865, %dims866) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %867 = "tf.Round"(%866) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims868 = "tf.Const"() { value = dense<[1, 0, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %868 = "tf.Transpose"(%867, %dims868) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims869 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %869 = "tf.Prod"(%868, %dims869) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims870 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %870 = "tf.Mean"(%869, %dims870) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims871 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %871 = "tf.Max"(%870, %dims871) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims872 = "tf.Const"() { value = dense<[2, 3, 0, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %872 = "tf.Transpose"(%871, %dims872) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims873 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %873 = "tf.Min"(%872, %dims873) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims874 = "tf.Const"() { value = dense<[2, 0, 3, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %874 = "tf.Transpose"(%873, %dims874) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims875 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %875 = "tf.Prod"(%874, %dims875) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims876 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %876 = "tf.Prod"(%875, %dims876) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?xi32>
  %877 = "tf.NotEqual"(%864, %876) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi1>
  %878 = "tf.Ceil"(%V__11) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims879 = "tf.Const"() { value = dense<[3, 2, 1, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %879 = "tf.Transpose"(%878, %dims879) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims880 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %880 = "tf.Max"(%V__28, %dims880) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %881 = "tf.Xlogy"(%879, %880) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims882 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %882 = "tf.Min"(%881, %dims882) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %883 = "tf.LeakyRelu"(%882) { alpha = 3.000000e-01 : f32, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims884 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %884 = "tf.Min"(%883, %dims884) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %dims885 = "tf.Const"() { value = dense<[1, 3, 0, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %885 = "tf.Transpose"(%884, %dims885) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims886 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %886 = "tf.Max"(%885, %dims886) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %887 = "tf.Const"() { value = dense<[90, 1, 1, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %888 = "tf.BroadcastTo"(%V__6, %887) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims889 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %889 = "tf.Prod"(%888, %dims889) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %890 = "tf.Shape"(%889) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<4xi32>
  %891 = "tf.Digamma"(%V__21) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>) -> tensor<f32>
  %892 = "tf.Polygamma"(%891, %V__21) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %893 = "tf.Fill"(%890, %892) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<4xi32>, tensor<f32>) -> tensor<?x?x?x?xf32>
  %dims894 = "tf.Const"() { value = dense<[0, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %894 = "tf.Sum"(%893, %dims894) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?xf32>
  %895 = "tf.BiasAdd"(%886, %894) { data_format = "NCHW", device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?xf32>) -> tensor<?x?x?x?xf32>
  %896 = "tf.OnesLike"(%895) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %897 = "tf.Tanh"(%896) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims898 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %898 = "tf.Mean"(%897, %dims898) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %dims899 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %899 = "tf.Prod"(%898, %dims899) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %900 = "tf.Ceil"(%899) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims901 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %901 = "tf.Prod"(%900, %dims901) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %dims902 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %902 = "tf.Sum"(%901, %dims902) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  %903 = "tf.Lgamma"(%V__34) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>) -> tensor<f32>
  %904 = "tf.Rsqrt"(%V__21) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>) -> tensor<f32>
  %905 = "tf.NextAfter"(%903, %904) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %906 = "tf.Relu6"(%905) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>) -> tensor<f32>
  %907 = "tf.Cast"(%906) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>) -> tensor<f32>
  %908 = "tf.Const"() { value = dense<[1, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %910 = "tf.Atan2"(%V__29, %V__11) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %911 = "tf.Floor"(%910) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %912 = "tf.Const"() { value = dense<[1, 1, 1, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %913 = "tf.BroadcastTo"(%V__8, %912) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims914 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %914 = "tf.Min"(%913, %dims914) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %915 = "tf.SquaredDifference"(%911, %914) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims916 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %916 = "tf.Max"(%915, %dims916) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %dims917 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %917 = "tf.Mean"(%916, %dims917) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %918 = "tf.Rsqrt"(%917) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims919 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %919 = "tf.Max"(%918, %dims919) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %dims920 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %920 = "tf.Min"(%919, %dims920) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  %921 = "tf.Xlog1py"(%907, %920) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %922 = "tf.SelectV2"(%877, %902, %921) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi1>, tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %dims923 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %923 = "tf.Prod"(%922, %dims923) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  %dims924 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %924 = "tf.Any"(%V__4, %dims924) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<1xi32>) -> tensor<?x?x?x?xi1>
  %925 = "tf.Shape"(%V__14) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<4xi32>
  %926 = "tf.BroadcastTo"(%924, %925) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<4xi32>) -> tensor<?x?x?x?xi1>
  %927 = "tf.Cast"(%926) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>) -> tensor<?x?x?x?xf32>
  %928 = "tf.Tanh"(%927) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims929 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %929 = "tf.Prod"(%V__12, %dims929) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims930 = "tf.Const"() { value = dense<[3, 0, 1, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %930 = "tf.Transpose"(%929, %dims930) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims931 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %931 = "tf.Max"(%930, %dims931) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %932 = "tf.LeakyRelu"(%931) { alpha = 3.000000e-01 : f32, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims933 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %933 = "tf.Min"(%932, %dims933) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %934 = "tf.Abs"(%933) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %935 = "tf.Xlogy"(%928, %934) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims936 = "tf.Const"() { value = dense<[2, 3, 1, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %936 = "tf.Transpose"(%935, %dims936) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims937 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %937 = "tf.Max"(%936, %dims937) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %dims938 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %938 = "tf.Mean"(%937, %dims938) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims939 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %939 = "tf.Min"(%V__29, %dims939) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<f32>
  %940 = "tf.Softplus"(%939) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>) -> tensor<f32>
  %941 = "tf.Cosh"(%940) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>) -> tensor<f32>
  %942 = "tf.Const"() { value = dense<[1, 1, 1, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %943 = "tf.Reshape"(%941, %942) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %944 = "tf.Div"(%V__1, %V__27) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %945 = "tf.Elu"(%944) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims946 = "tf.Const"() { value = dense<[0, 1, 3, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %946 = "tf.Transpose"(%945, %dims946) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims947 = "tf.Const"() { value = dense<[2, 0, 1, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %947 = "tf.Transpose"(%946, %dims947) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %948 = "tf.Softplus"(%947) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %949 = "tf.FloorMod"(%943, %948) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims950 = "tf.Const"() { value = dense<[3, 2, 0, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %950 = "tf.Transpose"(%949, %dims950) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %951 = "tf.NextAfter"(%938, %950) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims952 = "tf.Const"() { value = dense<[3, 0, 2, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %952 = "tf.Transpose"(%951, %dims952) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %953 = "tf.LeakyRelu"(%952) { alpha = 1.000000e-01 : f32, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims954 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %954 = "tf.Mean"(%953, %dims954) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims955 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %955 = "tf.Mean"(%954, %dims955) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %956 = "tf.Log"(%955) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims957 = "tf.Const"() { value = dense<[0, 3, 2, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %957 = "tf.Transpose"(%956, %dims957) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims958 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %958 = "tf.Prod"(%957, %dims958) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims959 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %959 = "tf.Max"(%958, %dims959) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  %dims960 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %960 = "tf.Mean"(%959, %dims960) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>, tensor<1xi32>) -> tensor<?x?xf32>
  %dims961 = "tf.Const"() { value = dense<[0, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %961 = "tf.Max"(%V__29, %dims961) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %dims962 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %962 = "tf.Max"(%961, %dims962) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %dims963 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %963 = "tf.Max"(%962, %dims963) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %964 = "tf.Relu6"(%963) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims965 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %965 = "tf.Mean"(%964, %dims965) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  %966 = "tf.Cos"(%965) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %dims967 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %967 = "tf.Min"(%V__8, %dims967) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?xf32>
  %968 = "tf.Const"() { value = dense<[42, 1, 1]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %969 = "tf.BroadcastTo"(%967, %968) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  %dims970 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %970 = "tf.Min"(%969, %dims970) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<1xi32>) -> tensor<?x?xf32>
  %dims971 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %971 = "tf.Max"(%970, %dims971) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  %972 = "tf.Acos"(%971) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %dims973 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %973 = "tf.Sum"(%V__1, %dims973) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  %dims974 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %974 = "tf.Min"(%973, %dims974) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>, tensor<1xi32>) -> tensor<?x?xf32>
  %dims975 = "tf.Const"() { value = dense<[0, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %975 = "tf.Sum"(%V__29, %dims975) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %dims976 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %976 = "tf.Max"(%975, %dims976) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  %977 = "tf.SquaredDifference"(%974, %976) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %978 = "tf.ClipByValue"(%966, %972, %977) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %979 = "tf.ApproximateEqual"(%V__35, %V__35) { tolerance = 1.000000e-04 : f32, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %dims980 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %980 = "tf.Max"(%V__12, %dims980) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %dims981 = "tf.Const"() { value = dense<[3, 1, 2, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %981 = "tf.Transpose"(%980, %dims981) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %982 = "tf.Erfc"(%V__29) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %983 = "tf.OnesLike"(%982) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %984 = "tf.Select"(%979, %981, %983) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i1>, tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims985 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %985 = "tf.Prod"(%984, %dims985) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  %dims986 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %986 = "tf.Sum"(%V__27, %dims986) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %987 = "tf.Acosh"(%V__36) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %988 = "tf.FloorDiv"(%986, %987) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %989 = "tf.Asinh"(%988) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims990 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %990 = "tf.Prod"(%989, %dims990) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %dims991 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %991 = "tf.Sum"(%990, %dims991) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims992 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %992 = "tf.Min"(%991, %dims992) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?xf32>
  %993 = "tf.Squeeze"(%992) { squeeze_dims = [ 1 : i64 ], device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?xf32>
  %994 = "tf.Div"(%985, %993) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %995 = "tf.Zeta"(%978, %994) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %996 = "tf.DivNoNan"(%960, %995) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %997 = "tf.Mod"(%923, %996) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %dims998 = "tf.Const"() { value = dense<[0, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %998 = "tf.Min"(%V__20, %dims998) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?xi32>
  %dims999 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %999 = "tf.Min"(%998, %dims999) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32>
  %1000 = "tf.Cast"(%999) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi32>) -> tensor<?xi1>
  %dims1001 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1001 = "tf.Max"(%V__1, %dims1001) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %1002 = "tf.Sigmoid"(%1001) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1003 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1003 = "tf.Sum"(%1002, %dims1003) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims1004 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1004 = "tf.Prod"(%V__1, %dims1004) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims1005 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1005 = "tf.Min"(%1004, %dims1005) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %dims1006 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1006 = "tf.Max"(%1005, %dims1006) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %1007 = "tf.Select"(%1000, %1003, %1006) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi1>, tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1008 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1008 = "tf.Sum"(%1007, %dims1008) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims1009 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1009 = "tf.Min"(%1008, %dims1009) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<f32>
  %1010 = "tf.Const"() { value = dense<[29, 1, 1]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1011 = "tf.BroadcastTo"(%V__37, %1010) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  %dims1012 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1012 = "tf.Max"(%1011, %dims1012) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  %1013 = "tf.ZerosLike"(%1012) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %1014 = "tf.Tanh"(%1013) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %1015 = "tf.Rint"(%V__38) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %1016 = "tf.LeakyRelu"(%1015) { alpha = 3.000000e-01 : f32, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %dims1017 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1017 = "tf.Min"(%1016, %dims1017) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?xf32>
  %1018 = "tf.Softsign"(%1017) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %dims1019 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1019 = "tf.Sum"(%1018, %dims1019) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?xf32>
  %1020 = "tf.FloorMod"(%1014, %1019) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %dims1021 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1021 = "tf.Min"(%1020, %dims1021) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<3xi32>) -> tensor<f32>
  %1022 = "tf.ApproximateEqual"(%1009, %1021) { tolerance = 1.000000e-05 : f32, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %1023 = "tf.NextAfter"(%V__27, %V__28) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1024 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1024 = "tf.Sum"(%1023, %dims1024) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims1025 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1025 = "tf.Prod"(%V__39, %dims1025) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims1026 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1026 = "tf.Max"(%1025, %dims1026) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims1027 = "tf.Const"() { value = dense<[0, 2, 3, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1027 = "tf.Transpose"(%1026, %dims1027) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %1028 = "tf.Add"(%1024, %1027) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1029 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1029 = "tf.Sum"(%1028, %dims1029) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims1030 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1030 = "tf.Mean"(%1029, %dims1030) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  %1031 = "tf.Round"(%1030) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %dims1032 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1032 = "tf.Prod"(%1031, %dims1032) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>, tensor<2xi32>) -> tensor<f32>
  %1033 = "tf.Ceil"(%1032) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>) -> tensor<f32>
  %dims1034 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1034 = "tf.Mean"(%V__23, %dims1034) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %dims1035 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1035 = "tf.Sum"(%1034, %dims1035) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims1036 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1036 = "tf.Mean"(%1035, %dims1036) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims1037 = "tf.Const"() { value = dense<[0, 1, 3, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1037 = "tf.Transpose"(%1036, %dims1037) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %1038 = "tf.IsInf"(%V__1) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xi1>
  %1039 = "tf.SelectV2"(%1038, %V__23, %V__23) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %1040 = "tf.Maximum"(%1037, %1039) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims1041 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1041 = "tf.Max"(%1040, %dims1041) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %1042 = "tf.Shape"(%1041) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<4xi32>
  %1043 = "tf.Reshape"(%1033, %1042) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %1044 = "tf.Softmax"(%1043) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1045 = "tf.Cast"(%V__23) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xf32>
  %1046 = "tf.ClipByValue"(%1045, %V__1, %V__11) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1047 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1047 = "tf.Prod"(%1046, %dims1047) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %dims1048 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1048 = "tf.Min"(%V__7, %dims1048) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?xi32>
  %1049 = "tf.Const"() { value = dense<[68, 1, 60, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1050 = "tf.BroadcastTo"(%1048, %1049) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %1051 = "tf.Cast"(%1050) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xf32>
  %dims1052 = "tf.Const"() { value = dense<[0, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1052 = "tf.Prod"(%1051, %dims1052) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %1053 = "tf.DivNoNan"(%1047, %1052) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1054 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1054 = "tf.Prod"(%V__40, %dims1054) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims1055 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1055 = "tf.Prod"(%1054, %dims1055) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %1056 = "tf.Log"(%1055) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1057 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1057 = "tf.Sum"(%1056, %dims1057) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %dims1058 = "tf.Const"() { value = dense<[0, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1058 = "tf.Sum"(%V__36, %dims1058) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %1059 = "tf.OnesLike"(%1058) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1060 = "tf.Log1p"(%1059) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1061 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1061 = "tf.Prod"(%1060, %dims1061) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %1062 = "tf.Div"(%1057, %1061) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1063 = "tf.Const"() { value = dense<[0, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1063 = "tf.Max"(%1062, %dims1063) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %dims1064 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1064 = "tf.Min"(%1063, %dims1064) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %1065 = "tf.Xdivy"(%1053, %1064) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1066 = "tf.Selu"(%1065) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1067 = "tf.Const"() { value = dense<[1, 2, 0, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1067 = "tf.Transpose"(%1066, %dims1067) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %1068 = "tf.Select"(%1022, %1044, %1067) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i1>, tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1069 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1069 = "tf.Prod"(%1068, %dims1069) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims1070 = "tf.Const"() { value = dense<[1, 0]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1070 = "tf.Transpose"(%V__15, %dims1070) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi1>, tensor<2xi32>) -> tensor<?x?xi1>
  %1071 = "tf.Shape"(%V__27) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<4xi32>
  %1072 = "tf.BroadcastTo"(%1070, %1071) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi1>, tensor<4xi32>) -> tensor<?x?x?x?xi1>
  %dims1073 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1073 = "tf.Any"(%1072, %dims1073) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<3xi32>) -> tensor<?xi1>
  %1074 = "tf.Const"() { value = dense<[71, 1, 1]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1076 = "tf.Const"() { value = dense<[1, 71, 1, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1077 = "tf.BroadcastTo"(%1073, %1076) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi1>, tensor<4xi32>) -> tensor<?x?x?x?xi1>
  %dims1078 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1078 = "tf.All"(%1077, %dims1078) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<1xi32>) -> tensor<?x?x?x?xi1>
  %dims1079 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1079 = "tf.Prod"(%V__1, %dims1079) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %1080 = "tf.Minimum"(%1079, %V__11) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1081 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1081 = "tf.Sum"(%1080, %dims1081) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %dims1082 = "tf.Const"() { value = dense<[0, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1082 = "tf.Min"(%1081, %dims1082) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %dims1083 = "tf.Const"() { value = dense<[2, 0, 1, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1083 = "tf.Transpose"(%1082, %dims1083) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims1084 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1084 = "tf.Min"(%1083, %dims1084) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims1085 = "tf.Const"() { value = dense<[0, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1085 = "tf.Min"(%V__12, %dims1085) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %dims1086 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1086 = "tf.Prod"(%1085, %dims1086) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %dims1087 = "tf.Const"() { value = dense<[0, 3, 2, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1087 = "tf.Transpose"(%1086, %dims1087) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims1088 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1088 = "tf.Max"(%1087, %dims1088) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %1089 = "tf.Square"(%1088) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1090 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1090 = "tf.Max"(%1089, %dims1090) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %1091 = "tf.FloorMod"(%1084, %1090) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1092 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1092 = "tf.Prod"(%1091, %dims1092) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims1093 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1093 = "tf.Prod"(%1092, %dims1093) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %1094 = "tf.Sqrt"(%1093) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1095 = "tf.Sinh"(%1094) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1096 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1096 = "tf.Mean"(%1095, %dims1096) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims1097 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1097 = "tf.Mean"(%1096, %dims1097) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %1098 = "tf.Cosh"(%1097) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1099 = "tf.Lgamma"(%1098) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1100 = "tf.OnesLike"(%1099) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1101 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1101 = "tf.Prod"(%1100, %dims1101) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %1102 = "tf.Ceil"(%1101) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1103 = "tf.Cos"(%1102) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1104 = "tf.Const"() { value = dense<[3, 1, 2, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1104 = "tf.Transpose"(%V__39, %dims1104) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims1105 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1105 = "tf.Prod"(%1104, %dims1105) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims1106 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1106 = "tf.Mean"(%1105, %dims1106) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %1107 = "tf.Asin"(%1106) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1108 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1108 = "tf.Sum"(%1107, %dims1108) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims1109 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1109 = "tf.Min"(%V__41, %dims1109) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %dims1110 = "tf.Const"() { value = dense<[0, 2, 3, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1110 = "tf.Transpose"(%1109, %dims1110) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims1111 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1111 = "tf.Min"(%1110, %dims1111) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims1112 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1112 = "tf.Sum"(%1111, %dims1112) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %dims1113 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1113 = "tf.Sum"(%1112, %dims1113) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %1114 = "tf.Polygamma"(%1108, %1113) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1115 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1115 = "tf.Prod"(%V__40, %dims1115) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %dims1116 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1116 = "tf.Min"(%1115, %dims1116) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %dims1117 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1117 = "tf.Sum"(%V__42, %dims1117) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %1118 = "tf.Xlogy"(%1116, %1117) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1119 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1119 = "tf.Max"(%1118, %dims1119) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %1120 = "tf.Inv"(%1119) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1121 = "tf.Digamma"(%1120) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1122 = "tf.Elu"(%1121) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1123 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1123 = "tf.Max"(%1122, %dims1123) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims1124 = "tf.Const"() { value = dense<[3, 2, 1, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1124 = "tf.Transpose"(%1123, %dims1124) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims1125 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1125 = "tf.Min"(%1124, %dims1125) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %1126 = "tf.MulNoNan"(%1114, %1125) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1127 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1127 = "tf.Prod"(%1126, %dims1127) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %1128 = "tf.Select"(%1078, %1103, %1127) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1129 = "tf.Digamma"(%1128) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1130 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1130 = "tf.Max"(%1129, %dims1130) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %1131 = "tf.Tan"(%1130) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1132 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1132 = "tf.Min"(%1131, %dims1132) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %1133 = "tf.NotEqual"(%1069, %1132) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xi1>
  %dims1134 = "tf.Const"() { value = dense<[0, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1134 = "tf.Any"(%1133, %dims1134) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<3xi32>) -> tensor<?x?x?x?xi1>
  %1135 = "tf.Shape"(%1134) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>) -> tensor<4xi32>
  %1136 = "tf.BroadcastTo"(%997, %1135) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %1137 = "tf.Log"(%1136) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1138 = "tf.Const"() { value = dense<[1, 1, 91, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1139 = "tf.BroadcastTo"(%V__25, %1138) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims1140 = "tf.Const"() { value = dense<[1, 2, 3, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1140 = "tf.Transpose"(%1139, %dims1140) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %1141 = "tf.Cast"(%1140) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xf32>
  %dims1142 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1142 = "tf.Min"(%1141, %dims1142) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %1143 = "tf.Acos"(%1142) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1144 = "tf.ClipByValue"(%V__43, %V__42, %V__39) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1145 = "tf.Tan"(%1144) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1146 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1146 = "tf.Min"(%1145, %dims1146) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %1147 = "tf.Xlogy"(%1143, %1146) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1148 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1148 = "tf.Sum"(%1147, %dims1148) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %dims1149 = "tf.Const"() { value = dense<[3, 0, 1, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1149 = "tf.Transpose"(%1148, %dims1149) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims1150 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1150 = "tf.Sum"(%1149, %dims1150) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %1151 = "tf.Log"(%1150) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1152 = "tf.Ceil"(%1151) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1153 = "tf.Asin"(%1152) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1154 = "tf.Cast"(%V__44) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i1>) -> tensor<f32>
  %1155 = "tf.Less"(%1154, %V__21) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %dims1156 = "tf.Const"() { value = dense<[3, 0, 1, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1156 = "tf.Transpose"(%V__28, %dims1156) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims1157 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1157 = "tf.Prod"(%1156, %dims1157) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %1158 = "tf.Softplus"(%V__27) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1159 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1159 = "tf.Mean"(%1158, %dims1159) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %1160 = "tf.Select"(%1155, %1157, %1159) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i1>, tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1161 = "tf.Abs"(%1160) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1162 = "tf.Asinh"(%1161) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1163 = "tf.Const"() { value = dense<[1, 2, 0, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1163 = "tf.Transpose"(%1162, %dims1163) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims1164 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1164 = "tf.Prod"(%1163, %dims1164) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %1165 = "tf.Sin"(%1164) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1166 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1166 = "tf.Max"(%1165, %dims1166) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims1167 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1167 = "tf.Max"(%1166, %dims1167) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %dims1168 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1168 = "tf.Mean"(%1167, %dims1168) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %1169 = "tf.Maximum"(%1153, %1168) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1170 = "tf.Rsqrt"(%1169) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1171 = "tf.Sin"(%1170) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1172 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1172 = "tf.Max"(%1171, %dims1172) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %dims1173 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1173 = "tf.Max"(%1172, %dims1173) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %1174 = "tf.Const"() { value = dense<[1, 1, 1]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1175 = "tf.BroadcastTo"(%V__31, %1174) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  %dims1176 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1176 = "tf.Sum"(%1175, %dims1176) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?xf32>
  %1177 = "tf.Relu6"(%V__38) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %1178 = "tf.Shape"(%1177) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<3xi32>
  %dims1180 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1180 = "tf.Max"(%V__12, %dims1180) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims1181 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1181 = "tf.Min"(%1180, %dims1181) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims1182 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1182 = "tf.Prod"(%V__43, %dims1182) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %1183 = "tf.Greater"(%1181, %1182) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xi1>
  %1184 = "tf.Shape"(%1183) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>) -> tensor<4xi32>
  %1185 = "tf.BroadcastTo"(%1176, %1184) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims1186 = "tf.Const"() { value = dense<[3, 1, 2, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1186 = "tf.Transpose"(%1185, %dims1186) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims1187 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1187 = "tf.Min"(%1186, %dims1187) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %dims1188 = "tf.Const"() { value = dense<[0, 3, 2, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1188 = "tf.Transpose"(%1187, %dims1188) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims1189 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1189 = "tf.Max"(%1188, %dims1189) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims1190 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1190 = "tf.Min"(%1189, %dims1190) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims1191 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1191 = "tf.Min"(%V__29, %dims1191) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims1192 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1192 = "tf.Prod"(%1191, %dims1192) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims1193 = "tf.Const"() { value = dense<[2, 3, 1, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1193 = "tf.Transpose"(%1192, %dims1193) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims1194 = "tf.Const"() { value = dense<[1, 3, 0, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1194 = "tf.Transpose"(%1193, %dims1194) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims1195 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1195 = "tf.Min"(%1194, %dims1195) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %dims1196 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1196 = "tf.Sum"(%1195, %dims1196) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %1197 = "tf.ZerosLike"(%1196) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1198 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1198 = "tf.Sum"(%1197, %dims1198) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %dims1199 = "tf.Const"() { value = dense<[0, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1199 = "tf.Sum"(%1198, %dims1199) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %dims1200 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1200 = "tf.Max"(%V__28, %dims1200) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %dims1201 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1201 = "tf.Max"(%1200, %dims1201) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims1202 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1202 = "tf.Mean"(%1201, %dims1202) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %dims1203 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1203 = "tf.Sum"(%V__1, %dims1203) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %dims1204 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1204 = "tf.Prod"(%1203, %dims1204) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %1205 = "tf.Tan"(%1204) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1206 = "tf.NextAfter"(%1202, %1205) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1207 = "tf.RealDiv"(%1199, %1206) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1208 = "tf.Const"() { value = dense<[1, 3, 2, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1208 = "tf.Transpose"(%1207, %dims1208) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %1209 = "tf.Div"(%1190, %1208) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1210 = "tf.Div"(%1173, %1209) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1211 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1211 = "tf.Sum"(%V__28, %dims1211) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %1212 = "tf.LeakyRelu"(%1211) { alpha = 1.000000e-01 : f32, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1213 = "tf.Cast"(%1212) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1214 = "tf.Acosh"(%V__45) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1215 = "tf.Const"() { value = dense<[3, 2, 1, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1215 = "tf.Transpose"(%1214, %dims1215) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %1216 = "tf.SquaredDifference"(%1213, %1215) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1217 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1217 = "tf.Sum"(%V__45, %dims1217) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %1218 = "tf.Round"(%1217) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1219 = "tf.Const"() { value = dense<[3, 1, 2, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1219 = "tf.Transpose"(%V__46, %dims1219) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %1220 = "tf.Shape"(%1219) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<4xi32>
  %1222 = "tf.DivNoNan"(%1216, %1218) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1223 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1223 = "tf.Min"(%1222, %dims1223) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %dims1224 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1224 = "tf.Max"(%1223, %dims1224) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %dims1225 = "tf.Const"() { value = dense<[3, 2, 1, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1225 = "tf.Transpose"(%V__11, %dims1225) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims1226 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1226 = "tf.Min"(%1225, %dims1226) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims1227 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1227 = "tf.Sum"(%1226, %dims1227) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %dims1228 = "tf.Const"() { value = dense<[3, 2, 1, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1228 = "tf.Transpose"(%1227, %dims1228) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %1229 = "tf.Floor"(%V__47) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1230 = "tf.Inv"(%1229) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1231 = "tf.Rsqrt"(%1230) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1232 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1232 = "tf.Max"(%1231, %dims1232) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %1233 = "tf.Sub"(%1228, %1232) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1234 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1234 = "tf.Sum"(%1233, %dims1234) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims1235 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1235 = "tf.Min"(%1234, %dims1235) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %1236 = "tf.LeakyRelu"(%1235) { alpha = 1.000000e-01 : f32, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1237 = "tf.Const"() { value = dense<[3, 1, 0, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1237 = "tf.Transpose"(%1236, %dims1237) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %1238 = "tf.Floor"(%1237) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1239 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1239 = "tf.Max"(%1238, %dims1239) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %1240 = "tf.Shape"(%1239) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<4xi32>
  %1241 = "tf.BroadcastTo"(%1224, %1240) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims1242 = "tf.Const"() { value = dense<[0, 3, 2, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1242 = "tf.Transpose"(%1241, %dims1242) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims1243 = "tf.Const"() { value = dense<[1, 3, 2, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1243 = "tf.Transpose"(%1242, %dims1243) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %1244 = "tf.Log1p"(%1243) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1245 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1245 = "tf.Sum"(%1244, %dims1245) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %1246 = "tf.Rint"(%1245) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1247 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1247 = "tf.All"(%V__4, %dims1247) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<2xi32>) -> tensor<?x?xi1>
  %1248 = "tf.Rint"(%V__37) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %1249 = "tf.Cos"(%V__37) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %1250 = "tf.Select"(%1247, %1248, %1249) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi1>, tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %1251 = "tf.Const"() { value = dense<[43, 91, 1, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1252 = "tf.BroadcastTo"(%1250, %1251) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims1253 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1253 = "tf.Max"(%1252, %dims1253) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %dims1254 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1254 = "tf.Max"(%1253, %dims1254) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %1255 = "tf.Elu"(%V__48) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>) -> tensor<f32>
  %1256 = "tf.Const"() { value = dense<[48, 79]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1257 = "tf.BroadcastTo"(%1255, %1256) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>, tensor<2xi32>) -> tensor<?x?xf32>
  %1258 = "tf.Cast"(%1257) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %dims1259 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1259 = "tf.Mean"(%1258, %dims1259) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>, tensor<1xi32>) -> tensor<?xf32>
  %dims1260 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1260 = "tf.Mean"(%1259, %dims1260) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>, tensor<1xi32>) -> tensor<f32>
  %1261 = "tf.Rint"(%1260) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>) -> tensor<f32>
  %dims1262 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1262 = "tf.Mean"(%V__2, %dims1262) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims1263 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1263 = "tf.Min"(%1262, %dims1263) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims1264 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1264 = "tf.Max"(%1263, %dims1264) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims1265 = "tf.Const"() { value = dense<[0, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1265 = "tf.Sum"(%1264, %dims1265) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims1266 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1266 = "tf.Prod"(%1265, %dims1266) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?xi32>
  %1267 = "tf.Shape"(%1266) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi32>) -> tensor<1xi32>
  %1268 = "tf.BroadcastTo"(%1261, %1267) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>, tensor<1xi32>) -> tensor<?xf32>
  %dims1269 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1269 = "tf.Min"(%1268, %dims1269) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>, tensor<1xi32>) -> tensor<?xf32>
  %1270 = "tf.BiasAdd"(%1254, %1269) { data_format = "NHWC", device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?xf32>) -> tensor<?x?x?x?xf32>
  %dims1271 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1271 = "tf.Mean"(%1270, %dims1271) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %dims1272 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1272 = "tf.Mean"(%1271, %dims1272) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims1273 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1273 = "tf.Prod"(%1272, %dims1273) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %1274 = "tf.Exp"(%1273) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1275 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1275 = "tf.Max"(%1274, %dims1275) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %dims1276 = "tf.Const"() { value = dense<[3, 2, 1, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1276 = "tf.Transpose"(%1275, %dims1276) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims1277 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1277 = "tf.Min"(%1276, %dims1277) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %1278 = "tf.Xlog1py"(%1246, %1277) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1279 = "tf.Square"(%1278) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1280 = "tf.Abs"(%1279) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1281 = "tf.Const"() { value = dense<[0, 1, 3, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1281 = "tf.Transpose"(%1280, %dims1281) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %1282 = "tf.DivNoNan"(%1210, %1281) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1283 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1283 = "tf.Sum"(%1282, %dims1283) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %dims1284 = "tf.Const"() { value = dense<[1, 0, 3, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1284 = "tf.Transpose"(%1283, %dims1284) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims1285 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1285 = "tf.Prod"(%1284, %dims1285) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %1286 = "tf.Asinh"(%1285) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1287 = "tf.Rsqrt"(%1286) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1288 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1288 = "tf.Prod"(%V__27, %dims1288) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims1289 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1289 = "tf.Prod"(%1288, %dims1289) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims1290 = "tf.Const"() { value = dense<[0, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1290 = "tf.Mean"(%V__39, %dims1290) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %1291 = "tf.SquaredDifference"(%1289, %1290) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1292 = "tf.Neg"(%V__41) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1293 = "tf.Log"(%1292) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1294 = "tf.LeakyRelu"(%V__29) { alpha = 3.000000e-01 : f32, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1295 = "tf.Div"(%1293, %1294) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1296 = "tf.Zeta"(%1291, %1295) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1297 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1297 = "tf.Max"(%1296, %dims1297) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %dims1298 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1298 = "tf.Min"(%1297, %dims1298) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %1299 = "tf.Relu6"(%1298) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1300 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1300 = "tf.Sum"(%1299, %dims1300) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %1301 = "tf.Sinh"(%1300) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1302 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1302 = "tf.Sum"(%1301, %dims1302) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims1303 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1303 = "tf.Min"(%V__8, %dims1303) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<2xi32>) -> tensor<?xf32>
  %1304 = "tf.Erfc"(%1303) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>) -> tensor<?xf32>
  %1305 = "tf.Erfc"(%1304) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>) -> tensor<?xf32>
  %dims1306 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1306 = "tf.Max"(%V__29, %dims1306) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %1307 = "tf.Shape"(%1306) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<4xi32>
  %1308 = "tf.BroadcastTo"(%1305, %1307) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims1309 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1309 = "tf.Mean"(%1308, %dims1309) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %dims1310 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1310 = "tf.Sum"(%V__1, %dims1310) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims1311 = "tf.Const"() { value = dense<[0, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1311 = "tf.Mean"(%1310, %dims1311) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %dims1312 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1312 = "tf.Sum"(%1311, %dims1312) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %dims1313 = "tf.Const"() { value = dense<[1, 3, 0, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1313 = "tf.Transpose"(%1312, %dims1313) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims1314 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1314 = "tf.Mean"(%1313, %dims1314) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %dims1315 = "tf.Const"() { value = dense<[1, 2, 0, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1315 = "tf.Transpose"(%1314, %dims1315) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims1316 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1316 = "tf.Mean"(%1315, %dims1316) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %1317 = "tf.Xdivy"(%1309, %1316) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1318 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1318 = "tf.Mean"(%1317, %dims1318) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %1319 = "tf.SquaredDifference"(%1302, %1318) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1320 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1320 = "tf.Prod"(%1319, %dims1320) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?xf32>
  %dims1321 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1321 = "tf.Prod"(%1320, %dims1321) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?xf32>
  %1322 = "tf.Atan"(%1321) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %dims1323 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1323 = "tf.Any"(%V__13, %dims1323) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<3xi32>) -> tensor<?x?x?x?xi1>
  %dims1324 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1324 = "tf.Any"(%1323, %dims1324) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<2xi32>) -> tensor<?x?x?x?xi1>
  %dims1325 = "tf.Const"() { value = dense<[3, 1, 0, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1325 = "tf.Transpose"(%V__23, %dims1325) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims1326 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1326 = "tf.Sum"(%1325, %dims1326) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %1327 = "tf.Neg"(%V__23) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %1328 = "tf.SelectV2"(%1324, %1326, %1327) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims1329 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1329 = "tf.Sum"(%V__3, %dims1329) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %1330 = "tf.TruncateDiv"(%1329, %V__3) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims1331 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1331 = "tf.Min"(%V__49, %dims1331) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %1332 = "tf.Pow"(%1331, %V__49) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %1333 = "tf.Pow"(%1330, %1332) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %1334 = "tf.LessEqual"(%1328, %1333) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi1>
  %dims1335 = "tf.Const"() { value = dense<[2, 0, 3, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1335 = "tf.Transpose"(%1334, %dims1335) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<4xi32>) -> tensor<?x?x?x?xi1>
  %dims1336 = "tf.Const"() { value = dense<[0, 3, 1, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1336 = "tf.Transpose"(%V__4, %dims1336) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<4xi32>) -> tensor<?x?x?x?xi1>
  %dims1337 = "tf.Const"() { value = dense<[3, 2, 1, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1337 = "tf.Transpose"(%V__32, %dims1337) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims1338 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1338 = "tf.Min"(%V__20, %dims1338) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %1339 = "tf.SelectV2"(%1336, %1337, %1338) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims1340 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1340 = "tf.Prod"(%1339, %dims1340) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims1341 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1341 = "tf.Prod"(%V__14, %dims1341) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?xi32>
  %1342 = "tf.Shape"(%V__1) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<4xi32>
  %1343 = "tf.BroadcastTo"(%1341, %1342) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims1344 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1344 = "tf.Mean"(%1343, %dims1344) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims1345 = "tf.Const"() { value = dense<[2, 1, 3, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1345 = "tf.Transpose"(%1344, %dims1345) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %1346 = "tf.LessEqual"(%1340, %1345) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi1>
  %dims1347 = "tf.Const"() { value = dense<[1, 0, 3, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1347 = "tf.Transpose"(%1346, %dims1347) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<4xi32>) -> tensor<?x?x?x?xi1>
  %dims1348 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1348 = "tf.All"(%1347, %dims1348) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<2xi32>) -> tensor<?x?x?x?xi1>
  %dims1349 = "tf.Const"() { value = dense<[2, 0, 1, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1349 = "tf.Transpose"(%1348, %dims1349) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<4xi32>) -> tensor<?x?x?x?xi1>
  %1350 = "tf.LogicalAnd"(%1335, %1349) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<?x?x?x?xi1>) -> tensor<?x?x?x?xi1>
  %1351 = "tf.ZerosLike"(%1350) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>) -> tensor<?x?x?x?xi1>
  %1352 = "tf.Shape"(%1351) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>) -> tensor<4xi32>
  %1353 = "tf.BroadcastTo"(%1322, %1352) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %1354 = "tf.Tan"(%1353) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1355 = "tf.Const"() { value = dense<[3, 0, 1, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1355 = "tf.Transpose"(%1354, %dims1355) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %1356 = "tf.Floor"(%V__46) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1357 = "tf.MulNoNan"(%1356, %V__47) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1358 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1358 = "tf.Mean"(%1357, %dims1358) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %dims1359 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1359 = "tf.Min"(%1358, %dims1359) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %dims1360 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1360 = "tf.Mean"(%1359, %dims1360) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %1361 = "tf.Atan2"(%V__8, %V__38) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %dims1362 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1362 = "tf.Min"(%1361, %dims1362) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<1xi32>) -> tensor<?x?xf32>
  %dims1363 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1363 = "tf.Min"(%1362, %dims1363) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>, tensor<1xi32>) -> tensor<?x?xf32>
  %1364 = "tf.Squeeze"(%1363) { squeeze_dims = [ 1 : i64 ], device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>) -> tensor<?xf32>
  %1365 = "tf.BiasAdd"(%1360, %1364) { data_format = "NCHW", device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?xf32>) -> tensor<?x?x?x?xf32>
  %dims1366 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1366 = "tf.Min"(%1365, %dims1366) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %dims1367 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1367 = "tf.Prod"(%1366, %dims1367) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %1368 = "tf.Asin"(%1367) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1369 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1369 = "tf.Prod"(%1368, %dims1369) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %dims1370 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1370 = "tf.Max"(%1369, %dims1370) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %1371 = "tf.Inv"(%V__40) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1372 = "tf.Const"() { value = dense<[2, 0, 1, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1372 = "tf.Transpose"(%V__12, %dims1372) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %1373 = "tf.DivNoNan"(%1371, %1372) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1374 = "tf.Squeeze"(%1373) { squeeze_dims = [ 3 : i64 ], device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?xf32>
  %1375 = "tf.Sqrt"(%1374) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %dims1376 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1376 = "tf.Max"(%V__40, %dims1376) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %1377 = "tf.Sqrt"(%1376) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1378 = "tf.Lgamma"(%1377) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1379 = "tf.Rsqrt"(%1378) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1380 = "tf.Const"() { value = dense<[0, 2, 3, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1380 = "tf.Transpose"(%1379, %dims1380) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %1381 = "tf.Shape"(%1380) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<4xi32>
  %1382 = "tf.BroadcastTo"(%1375, %1381) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims1383 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1383 = "tf.Mean"(%1382, %dims1383) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %dims1384 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1384 = "tf.Mean"(%1383, %dims1384) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %1385 = "tf.Sign"(%1384) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1386 = "tf.Xdivy"(%1370, %1385) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1387 = "tf.Digamma"(%1386) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1388 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1388 = "tf.Prod"(%1387, %dims1388) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims1389 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1389 = "tf.Sum"(%1388, %dims1389) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims1390 = "tf.Const"() { value = dense<[1, 2, 3, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1390 = "tf.Transpose"(%V__27, %dims1390) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims1391 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1391 = "tf.Mean"(%1390, %dims1391) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims1392 = "tf.Const"() { value = dense<[3, 0, 1, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1392 = "tf.Transpose"(%1391, %dims1392) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims1393 = "tf.Const"() { value = dense<[2, 3, 1, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1393 = "tf.Transpose"(%1392, %dims1393) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %1394 = "tf.Inv"(%1393) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1395 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1395 = "tf.Prod"(%V__8, %dims1395) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?xf32>
  %1396 = "tf.Shape"(%V__29) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<4xi32>
  %1397 = "tf.BroadcastTo"(%1395, %1396) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims1398 = "tf.Const"() { value = dense<[0, 2, 3, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1398 = "tf.Transpose"(%1397, %dims1398) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %1399 = "tf.MulNoNan"(%1394, %1398) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1400 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1400 = "tf.Min"(%1399, %dims1400) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims1401 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1401 = "tf.Mean"(%1400, %dims1401) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %1402 = "tf.Neg"(%1401) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1403 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1403 = "tf.Min"(%1402, %dims1403) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims1404 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1404 = "tf.Prod"(%1403, %dims1404) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims1405 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1405 = "tf.Prod"(%1404, %dims1405) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %dims1406 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1406 = "tf.Any"(%V__13, %dims1406) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<2xi32>) -> tensor<?x?xi1>
  %dims1407 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1407 = "tf.All"(%1406, %dims1407) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi1>, tensor<1xi32>) -> tensor<?xi1>
  %1408 = "tf.Relu6"(%V__31) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>) -> tensor<?xf32>
  %1409 = "tf.LeakyRelu"(%1408) { alpha = 3.000000e-01 : f32, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>) -> tensor<?xf32>
  %1410 = "tf.Inv"(%V__31) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>) -> tensor<?xf32>
  %1411 = "tf.SelectV2"(%1407, %1409, %1410) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi1>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %dims1412 = "tf.Const"() { value = dense<[2, 1, 0, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1412 = "tf.Transpose"(%V__4, %dims1412) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<4xi32>) -> tensor<?x?x?x?xi1>
  %dims1413 = "tf.Const"() { value = dense<[3, 1, 2, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1413 = "tf.Transpose"(%V__4, %dims1413) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<4xi32>) -> tensor<?x?x?x?xi1>
  %1414 = "tf.LogicalAnd"(%1412, %1413) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<?x?x?x?xi1>) -> tensor<?x?x?x?xi1>
  %dims1415 = "tf.Const"() { value = dense<[3, 1, 0, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1415 = "tf.Transpose"(%1414, %dims1415) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<4xi32>) -> tensor<?x?x?x?xi1>
  %dims1416 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1416 = "tf.All"(%1415, %dims1416) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<1xi32>) -> tensor<?x?x?x?xi1>
  %1417 = "tf.Shape"(%1416) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>) -> tensor<4xi32>
  %1418 = "tf.BroadcastTo"(%1411, %1417) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims1419 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1419 = "tf.Prod"(%1418, %dims1419) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %1420 = "tf.NextAfter"(%1405, %1419) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1421 = "tf.Square"(%1420) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1422 = "tf.Const"() { value = dense<[0, 3, 1, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1422 = "tf.Transpose"(%1421, %dims1422) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %1423 = "tf.RealDiv"(%1389, %1422) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1424 = "tf.Reciprocal"(%1423) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1425 = "tf.Neg"(%1424) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1426 = "tf.Atan"(%1425) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1427 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1427 = "tf.Mean"(%1426, %dims1427) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %1428 = "tf.Atan2"(%1355, %1427) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1429 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1429 = "tf.Prod"(%1428, %dims1429) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims1430 = "tf.Const"() { value = dense<[0, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1430 = "tf.Mean"(%1429, %dims1430) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %dims1431 = "tf.Const"() { value = dense<[3, 1, 2, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1431 = "tf.Transpose"(%1430, %dims1431) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims1432 = "tf.Const"() { value = dense<[0, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1432 = "tf.Prod"(%1431, %dims1432) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %dims1433 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1433 = "tf.Prod"(%1432, %dims1433) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %dims1434 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1434 = "tf.Max"(%1433, %dims1434) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims1435 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1435 = "tf.Sum"(%1434, %dims1435) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %1436 = "tf.Mod"(%1287, %1435) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1437 = "tf.Const"() { value = dense<[3, 0, 2, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1437 = "tf.Transpose"(%1436, %dims1437) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims1438 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1438 = "tf.Min"(%V__50, %dims1438) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?xf32>
  %1439 = "tf.Atan2"(%1438, %V__8) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %1440 = "tf.LeakyRelu"(%V__36) { alpha = 1.000000e-01 : f32, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1441 = "tf.Tanh"(%1440) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1442 = "tf.Shape"(%1441) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<4xi32>
  %1443 = "tf.BroadcastTo"(%1439, %1442) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims1444 = "tf.Const"() { value = dense<[0, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1444 = "tf.Prod"(%1443, %dims1444) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %1445 = "tf.Atan2"(%V__45, %V__41) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1446 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1446 = "tf.Max"(%1445, %dims1446) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims1447 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1447 = "tf.Max"(%1446, %dims1447) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims1448 = "tf.Const"() { value = dense<[2, 1, 0, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1448 = "tf.Transpose"(%1447, %dims1448) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims1449 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1449 = "tf.Max"(%1448, %dims1449) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %dims1450 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1450 = "tf.Max"(%1449, %dims1450) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %dims1451 = "tf.Const"() { value = dense<[2, 0, 1, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1451 = "tf.Transpose"(%1450, %dims1451) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %1452 = "tf.Log"(%1451) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1453 = "tf.MulNoNan"(%1444, %1452) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1454 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1454 = "tf.Min"(%1453, %dims1454) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %1455 = "tf.Cos"(%1454) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1456 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1456 = "tf.Prod"(%1455, %dims1456) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %dims1457 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1457 = "tf.Prod"(%1456, %dims1457) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?xf32>
  %1458 = "tf.Cast"(%1457) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xi32>
  %dims1459 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1459 = "tf.Min"(%1458, %dims1459) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?xi32>
  %dims1460 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1460 = "tf.Max"(%1459, %dims1460) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?xi32>
  %dims1461 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1461 = "tf.Sum"(%V__43, %dims1461) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %1462 = "tf.NextAfter"(%1461, %V__41) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1463 = "tf.Squeeze"(%1462) { squeeze_dims = [ 2 : i64, 3 : i64 ], device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?xf32>
  %dims1464 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1464 = "tf.Min"(%V__9, %dims1464) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims1465 = "tf.Const"() { value = dense<[0, 2, 1, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1465 = "tf.Transpose"(%1464, %dims1465) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %1466 = "tf.Squeeze"(%1465) { squeeze_dims = [ 0 : i64, 3 : i64 ], device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?xi32>
  %1467 = "tf.Cast"(%1466) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>) -> tensor<?x?xf32>
  %1468 = "tf.GreaterEqual"(%1463, %1467) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xi1>
  %1469 = "tf.Squeeze"(%1468) { squeeze_dims = [ 0 : i64, 1 : i64 ], device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi1>) -> tensor<i1>
  %dims1470 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1470 = "tf.Sum"(%V__23, %dims1470) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %1471 = "tf.Shape"(%V__4) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>) -> tensor<4xi32>
  %dims1473 = "tf.Const"() { value = dense<[2, 0, 3, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1473 = "tf.Transpose"(%V__23, %dims1473) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %1474 = "tf.Cast"(%V__3) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %1475 = "tf.TruncateDiv"(%1473, %1474) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %1476 = "tf.RightShift"(%1470, %1475) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %1477 = "tf.Shape"(%1476) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<4xi32>
  %1478 = "tf.BroadcastTo"(%1469, %1477) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i1>, tensor<4xi32>) -> tensor<?x?x?x?xi1>
  %dims1479 = "tf.Const"() { value = dense<[3, 1, 0, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1479 = "tf.Transpose"(%1478, %dims1479) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<4xi32>) -> tensor<?x?x?x?xi1>
  %1480 = "tf.Shape"(%1479) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>) -> tensor<4xi32>
  %1481 = "tf.BroadcastTo"(%1460, %1480) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %1482 = "tf.Relu6"(%1481) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %1483 = "tf.Cast"(%1482) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xf32>
  %dims1484 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1484 = "tf.Max"(%1483, %dims1484) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?xf32>
  %dims1485 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1485 = "tf.Max"(%1484, %dims1485) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<2xi32>) -> tensor<?xf32>
  %dims1486 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1486 = "tf.Min"(%V__45, %dims1486) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %dims1487 = "tf.Const"() { value = dense<[1, 0, 3, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1487 = "tf.Transpose"(%1486, %dims1487) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %1488 = "tf.Inv"(%1487) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1489 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1489 = "tf.Sum"(%V__51, %dims1489) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %dims1490 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1490 = "tf.Prod"(%1489, %dims1490) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %dims1491 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1491 = "tf.Min"(%1490, %dims1491) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %1492 = "tf.Maximum"(%1488, %1491) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1493 = "tf.Const"() { value = dense<[3, 0, 2, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1493 = "tf.Transpose"(%V__29, %dims1493) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %1494 = "tf.Exp"(%1493) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1495 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1495 = "tf.Prod"(%1494, %dims1495) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims1496 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1496 = "tf.Max"(%1495, %dims1496) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %1497 = "tf.Square"(%1496) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1498 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1498 = "tf.Sum"(%1497, %dims1498) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %1499 = "tf.Rint"(%1498) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1500 = "tf.Shape"(%1499) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<4xi32>
  %1501 = "tf.BroadcastTo"(%1492, %1500) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %1502 = "tf.Neg"(%1501) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1503 = "tf.Digamma"(%1502) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1504 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1504 = "tf.Sum"(%1503, %dims1504) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %dims1505 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1505 = "tf.Sum"(%1504, %dims1505) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %1506 = "tf.Sign"(%1505) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1507 = "tf.Digamma"(%1506) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1508 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1508 = "tf.Prod"(%1507, %dims1508) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims1509 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1509 = "tf.Sum"(%1508, %dims1509) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims1510 = "tf.Const"() { value = dense<[2, 0, 1, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1510 = "tf.Transpose"(%1509, %dims1510) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %1511 = "tf.ZerosLike"(%1510) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1512 = "tf.Squeeze"(%1511) { squeeze_dims = [ 0 : i64, 2 : i64, 3 : i64 ], device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?xf32>
  %dims1513 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1513 = "tf.Prod"(%V__37, %dims1513) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>, tensor<1xi32>) -> tensor<?xf32>
  %1514 = "tf.Shape"(%V__15) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi1>) -> tensor<2xi32>
  %1515 = "tf.BroadcastTo"(%1513, %1514) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  %dims1516 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1516 = "tf.Mean"(%1515, %dims1516) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  %dims1517 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1517 = "tf.Sum"(%V__20, %dims1517) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %1518 = "tf.Cast"(%1517) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xf32>
  %dims1519 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1519 = "tf.Mean"(%1518, %dims1519) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %dims1520 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1520 = "tf.Sum"(%1519, %dims1520) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  %1521 = "tf.Mod"(%1516, %1520) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %dims1522 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1522 = "tf.Mean"(%1521, %dims1522) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>, tensor<1xi32>) -> tensor<?xf32>
  %1523 = "tf.Ceil"(%1522) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>) -> tensor<?xf32>
  %1524 = "tf.Inv"(%V__34) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>) -> tensor<f32>
  %dims1525 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1525 = "tf.Prod"(%V__31, %dims1525) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>, tensor<1xi32>) -> tensor<f32>
  %1526 = "tf.RealDiv"(%1524, %1525) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %1527 = "tf.Const"() { value = dense<[97, 93, 18, 40]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1528 = "tf.BroadcastTo"(%1526, %1527) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims1529 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1529 = "tf.Min"(%1528, %dims1529) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims1530 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1530 = "tf.Prod"(%1529, %dims1530) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?xf32>
  %1531 = "tf.Ceil"(%1530) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>) -> tensor<?xf32>
  %1532 = "tf.Xlog1py"(%1523, %1531) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %1533 = "tf.Atan2"(%1512, %1532) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %1534 = "tf.ApproximateEqual"(%1485, %1533) { tolerance = 1.000000e-06 : f32, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xi1>
  %dims1535 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1535 = "tf.Max"(%V__52, %dims1535) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %dims1536 = "tf.Const"() { value = dense<[1, 0, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1536 = "tf.Transpose"(%1535, %dims1536) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims1537 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1537 = "tf.Max"(%1536, %dims1537) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %1538 = "tf.LeakyRelu"(%V__36) { alpha = 3.000000e-01 : f32, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1539 = "tf.Rsqrt"(%1538) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1540 = "tf.Const"() { value = dense<[0, 3, 1, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1540 = "tf.Transpose"(%1539, %dims1540) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %1541 = "tf.Tanh"(%V__11) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1542 = "tf.Inv"(%1541) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1543 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1543 = "tf.Mean"(%1542, %dims1543) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %1544 = "tf.ClipByValue"(%1537, %1540, %1543) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1545 = "tf.Sin"(%V__53) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1546 = "tf.Const"() { value = dense<[1, 0, 3, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1546 = "tf.Transpose"(%1545, %dims1546) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %1547 = "tf.Zeta"(%V__43, %V__52) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1548 = "tf.SquaredDifference"(%1546, %1547) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1549 = "tf.Const"() { value = dense<[1, 2, 3, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1549 = "tf.Transpose"(%1548, %dims1549) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %1550 = "tf.Expm1"(%1549) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1551 = "tf.Const"() { value = dense<[1, 3, 2, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1551 = "tf.Transpose"(%1550, %dims1551) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims1552 = "tf.Const"() { value = dense<[1, 0, 3, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1552 = "tf.Transpose"(%1551, %dims1552) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims1553 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1553 = "tf.Prod"(%1552, %dims1553) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %1554 = "tf.Xdivy"(%1544, %1553) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1555 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1555 = "tf.Mean"(%1554, %dims1555) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %dims1556 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1556 = "tf.Mean"(%1555, %dims1556) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %dims1557 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1557 = "tf.Sum"(%1556, %dims1557) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims1558 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1558 = "tf.Prod"(%1557, %dims1558) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %dims1559 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1559 = "tf.Min"(%V__3, %dims1559) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?x?xi64>
  %dims1560 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1560 = "tf.Mean"(%1559, %dims1560) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %dims1561 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1561 = "tf.Max"(%1560, %dims1561) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?x?xi64>
  %dims1562 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1562 = "tf.Sum"(%1561, %dims1562) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims1563 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1563 = "tf.Mean"(%1562, %dims1563) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %dims1564 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1564 = "tf.Min"(%1563, %dims1564) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?x?xi64>
  %dims1565 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1565 = "tf.Sum"(%1564, %dims1565) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?x?xi64>
  %1566 = "tf.FloorDiv"(%V__6, %V__6) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<?x?x?xi64>) -> tensor<?x?x?xi64>
  %1567 = "tf.Const"() { value = dense<[1, 1, 1, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1568 = "tf.BroadcastTo"(%1566, %1567) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %1569 = "tf.Neg"(%1568) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims1570 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1570 = "tf.Sum"(%V__23, %dims1570) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %dims1571 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1571 = "tf.Sum"(%1570, %dims1571) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?xi64>
  %dims1572 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1572 = "tf.All"(%V__4, %dims1572) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<4xi32>) -> tensor<?x?x?x?xi1>
  %1573 = "tf.Shape"(%1572) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>) -> tensor<4xi32>
  %1574 = "tf.BroadcastTo"(%1571, %1573) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %1575 = "tf.Square"(%1574) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %1576 = "tf.ClipByValue"(%1565, %1569, %1575) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims1577 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1577 = "tf.Max"(%1576, %dims1577) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims1578 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1578 = "tf.Prod"(%1577, %dims1578) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims1579 = "tf.Const"() { value = dense<[2, 0, 3, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1579 = "tf.Transpose"(%1578, %dims1579) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %1580 = "tf.Cast"(%1579) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xf32>
  %1581 = "tf.SquaredDifference"(%1558, %1580) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1582 = "tf.LessEqual"(%V__31, %V__31) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xi1>
  %dims1583 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1583 = "tf.Any"(%V__54, %dims1583) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi1>, tensor<2xi32>) -> tensor<?xi1>
  %1584 = "tf.LogicalOr"(%1582, %1583) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi1>, tensor<?xi1>) -> tensor<?xi1>
  %dims1585 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1585 = "tf.Sum"(%V__3, %dims1585) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %1586 = "tf.Add"(%1585, %V__55) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims1587 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1587 = "tf.Max"(%1586, %dims1587) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %dims1588 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1588 = "tf.Min"(%1587, %dims1588) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims1589 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1589 = "tf.Min"(%V__3, %dims1589) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?x?xi64>
  %dims1590 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1590 = "tf.Prod"(%V__3, %dims1590) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %1591 = "tf.RightShift"(%1589, %1590) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims1592 = "tf.Const"() { value = dense<[0, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1592 = "tf.Sum"(%1591, %dims1592) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %1593 = "tf.Select"(%1584, %1588, %1592) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi1>, tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %1594 = "tf.Abs"(%1593) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims1595 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1595 = "tf.Prod"(%1594, %dims1595) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %dims1596 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1596 = "tf.Sum"(%1595, %dims1596) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %dims1597 = "tf.Const"() { value = dense<[3, 2, 0, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1597 = "tf.Transpose"(%1596, %dims1597) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims1598 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1598 = "tf.Sum"(%1597, %dims1598) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims1599 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1599 = "tf.Prod"(%1598, %dims1599) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %1600 = "tf.Cast"(%1599) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xf32>
  %dims1601 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1601 = "tf.Max"(%1600, %dims1601) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %1602 = "tf.TruncateDiv"(%V__6, %V__6) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<?x?x?xi64>) -> tensor<?x?x?xi64>
  %1603 = "tf.Shape"(%V__45) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<4xi32>
  %dims1605 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1605 = "tf.All"(%V__4, %dims1605) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<1xi32>) -> tensor<?x?x?x?xi1>
  %dims1606 = "tf.Const"() { value = dense<[3, 0, 1, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1606 = "tf.Transpose"(%1605, %dims1606) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<4xi32>) -> tensor<?x?x?x?xi1>
  %dims1607 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1607 = "tf.All"(%1606, %dims1607) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<2xi32>) -> tensor<?x?x?x?xi1>
  %1608 = "tf.Cast"(%1607) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>) -> tensor<?x?x?x?xi64>
  %1609 = "tf.NotEqual"(%1602, %1608) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi1>
  %1610 = "tf.Cast"(%1609) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>) -> tensor<?x?x?x?xf32>
  %1611 = "tf.Neg"(%1610) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1612 = "tf.Log1p"(%1611) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1613 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1613 = "tf.Min"(%1612, %dims1613) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %dims1614 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1614 = "tf.Min"(%1613, %dims1614) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims1615 = "tf.Const"() { value = dense<[0, 2, 3, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1615 = "tf.Transpose"(%1614, %dims1615) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims1616 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1616 = "tf.Mean"(%1615, %dims1616) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %1617 = "tf.Tan"(%1616) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1618 = "tf.Sqrt"(%1617) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1619 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1619 = "tf.Min"(%1618, %dims1619) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %dims1620 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1620 = "tf.Sum"(%1619, %dims1620) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %dims1621 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1621 = "tf.Prod"(%1620, %dims1621) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %dims1622 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1622 = "tf.Min"(%1621, %dims1622) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %dims1623 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1623 = "tf.Min"(%1622, %dims1623) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %1624 = "tf.FloorMod"(%1601, %1623) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1625 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1625 = "tf.Min"(%1624, %dims1625) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims1626 = "tf.Const"() { value = dense<[0, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1626 = "tf.Min"(%1625, %dims1626) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %dims1627 = "tf.Const"() { value = dense<[2, 1, 0, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1627 = "tf.Transpose"(%1626, %dims1627) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims1628 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1628 = "tf.Max"(%1627, %dims1628) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %dims1629 = "tf.Const"() { value = dense<[1, 2, 3, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1629 = "tf.Transpose"(%1628, %dims1629) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims1630 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1630 = "tf.Max"(%1629, %dims1630) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %1631 = "tf.Atan2"(%1581, %1630) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1632 = "tf.Cast"(%V__6) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>) -> tensor<?x?x?xf32>
  %1633 = "tf.Relu6"(%1632) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %1634 = "tf.Cos"(%1633) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %1635 = "tf.Neg"(%1634) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %1636 = "tf.Cosh"(%1635) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %1637 = "tf.SquaredDifference"(%V__38, %V__38) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %1638 = "tf.Softmax"(%1637) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %1639 = "tf.Log1p"(%1638) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %1640 = "tf.Erfc"(%1639) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %1641 = "tf.FloorMod"(%1636, %1640) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %dims1642 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1642 = "tf.Min"(%1641, %dims1642) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<2xi32>) -> tensor<?xf32>
  %dims1643 = "tf.Const"() { value = dense<[1, 2, 0, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1643 = "tf.Transpose"(%V__29, %dims1643) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %1644 = "tf.Lgamma"(%1643) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1645 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1645 = "tf.Max"(%1644, %dims1645) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %1646 = "tf.Asin"(%1645) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1647 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1647 = "tf.Mean"(%1646, %dims1647) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?xf32>
  %1648 = "tf.Erf"(%V__38) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %dims1649 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1649 = "tf.Mean"(%V__38, %dims1649) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?xf32>
  %1650 = "tf.Minimum"(%1648, %1649) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %dims1651 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1651 = "tf.Prod"(%1650, %dims1651) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<2xi32>) -> tensor<?xf32>
  %1652 = "tf.Polygamma"(%1647, %1651) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %1653 = "tf.Minimum"(%1642, %1652) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %1654 = "tf.Const"() { value = dense<[77, 1, 7, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1655 = "tf.BroadcastTo"(%1653, %1654) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %1656 = "tf.Relu"(%1655) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1657 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1657 = "tf.Min"(%1656, %dims1657) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %1658 = "tf.Cast"(%V__5) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi64>) -> tensor<?xi32>
  %1659 = "tf.Cast"(%1658) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi32>) -> tensor<?xf32>
  %dims1660 = "tf.Const"() { value = dense<[0, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1660 = "tf.Max"(%V__53, %dims1660) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?xf32>
  %1661 = "tf.Log"(%1660) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>) -> tensor<?xf32>
  %1662 = "tf.Pow"(%1659, %1661) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %1663 = "tf.IsNan"(%1662) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>) -> tensor<?xi1>
  %1664 = "tf.Squeeze"(%1663) { squeeze_dims = [ 0 : i64 ], device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi1>) -> tensor<i1>
  %1665 = "tf.Xlogy"(%V__50, %V__38) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %1666 = "tf.Lgamma"(%V__8) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %1667 = "tf.AddV2"(%1665, %1666) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %1668 = "tf.Sigmoid"(%1667) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %1669 = "tf.Square"(%1668) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %dims1670 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1670 = "tf.Prod"(%1669, %dims1670) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?xf32>
  %1671 = "tf.LeakyRelu"(%V__47) { alpha = 1.000000e-01 : f32, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1672 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1672 = "tf.Min"(%1671, %dims1672) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims1673 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1673 = "tf.Prod"(%1672, %dims1673) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %dims1674 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1674 = "tf.Sum"(%1673, %dims1674) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims1675 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1675 = "tf.Prod"(%1674, %dims1675) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?xf32>
  %1676 = "tf.Inv"(%1675) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %1677 = "tf.Asinh"(%1676) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %1678 = "tf.Select"(%1664, %1670, %1677) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i1>, tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %1679 = "tf.Asin"(%V__8) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %1680 = "tf.Erf"(%V__8) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %1681 = "tf.Maximum"(%1679, %1680) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %dims1682 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1682 = "tf.Mean"(%V__46, %dims1682) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?xf32>
  %1683 = "tf.Softsign"(%V__50) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %1684 = "tf.FloorMod"(%1682, %1683) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %1685 = "tf.NotEqual"(%1681, %1684) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xi1>
  %1686 = "tf.Cast"(%1685) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi1>) -> tensor<?x?x?xf32>
  %dims1687 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1687 = "tf.Mean"(%1686, %dims1687) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?xf32>
  %1688 = "tf.SquaredDifference"(%V__38, %V__56) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %1689 = "tf.Atanh"(%V__38) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %1690 = "tf.Minimum"(%1688, %1689) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %dims1691 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1691 = "tf.Sum"(%V__50, %dims1691) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?xf32>
  %1692 = "tf.Ceil"(%1691) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %1693 = "tf.Asin"(%1692) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %1694 = "tf.Abs"(%1693) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %1695 = "tf.Xdivy"(%1690, %1694) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %1696 = "tf.Minimum"(%1687, %1695) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %dims1697 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1697 = "tf.Min"(%1696, %dims1697) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?xf32>
  %1698 = "tf.MulNoNan"(%1678, %1697) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %1699 = "tf.Asinh"(%1698) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %dims1700 = "tf.Const"() { value = dense<[1, 0, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1700 = "tf.Transpose"(%1699, %dims1700) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  %1701 = "tf.Erf"(%1700) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %1702 = "tf.Squeeze"(%1701) { squeeze_dims = [ 1 : i64, 2 : i64 ], device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?xf32>
  %1703 = "tf.BiasAdd"(%1657, %1702) { data_format = "NCHW", device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?xf32>) -> tensor<?x?x?x?xf32>
  %dims1704 = "tf.Const"() { value = dense<[0, 1, 3, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1704 = "tf.Transpose"(%1703, %dims1704) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims1705 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1705 = "tf.Sum"(%1704, %dims1705) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %1706 = "tf.Select"(%1534, %1631, %1705) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi1>, tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1707 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1707 = "tf.Max"(%1706, %dims1707) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %1708 = "tf.Relu6"(%1707) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1709 = "tf.ClipByValue"(%1137, %1437, %1708) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1710 = "tf.Sinh"(%1709) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1711 = "tf.Const"() { value = dense<[0, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1711 = "tf.Min"(%1710, %dims1711) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %dims1712 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1712 = "tf.Max"(%1711, %dims1712) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %1713 = "tf.LeakyRelu"(%1712) { alpha = 3.000000e-01 : f32, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1714 = "tf.Cast"(%1713) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xi32>
  %dims1715 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1715 = "tf.Min"(%1714, %dims1715) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims1716 = "tf.Const"() { value = dense<[3, 1, 2, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1716 = "tf.Transpose"(%1715, %dims1716) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims1717 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1717 = "tf.Min"(%1716, %dims1717) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %1718 = "tf.Abs"(%1717) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %1719 = "tf.Ceil"(%V__12) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1720 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1720 = "tf.Mean"(%1719, %dims1720) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %dims1721 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1721 = "tf.Prod"(%1720, %dims1721) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %dims1722 = "tf.Const"() { value = dense<[0, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1722 = "tf.Sum"(%1721, %dims1722) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %1723 = "tf.Tanh"(%V__40) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1724 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1724 = "tf.Max"(%1723, %dims1724) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %1725 = "tf.Squeeze"(%1724) { squeeze_dims = [ 0 : i64, 2 : i64, 3 : i64 ], device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?xf32>
  %1726 = "tf.BiasAdd"(%1722, %1725) { data_format = "NCHW", device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?xf32>) -> tensor<?x?x?x?xf32>
  %dims1727 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1727 = "tf.Min"(%1726, %dims1727) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %dims1728 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1728 = "tf.Sum"(%1727, %dims1728) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %1729 = "tf.Abs"(%1728) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1730 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1730 = "tf.Max"(%1729, %dims1730) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims1731 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1731 = "tf.Mean"(%1730, %dims1731) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?xf32>
  %1732 = "tf.OnesLike"(%1731) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>) -> tensor<?xf32>
  %1733 = "tf.Shape"(%V__4) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>) -> tensor<4xi32>
  %1734 = "tf.Fill"(%1733, %V__21) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<4xi32>, tensor<f32>) -> tensor<?x?x?x?xf32>
  %dims1735 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1735 = "tf.Sum"(%V__53, %dims1735) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %1736 = "tf.Softmax"(%1735) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1737 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1737 = "tf.Min"(%1736, %dims1737) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %1738 = "tf.Polygamma"(%V__28, %V__12) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1739 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1739 = "tf.Min"(%1738, %dims1739) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %1740 = "tf.ClipByValue"(%1734, %1737, %1739) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1741 = "tf.Reciprocal"(%1740) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1742 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1742 = "tf.Sum"(%1741, %dims1742) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %1743 = "tf.Shape"(%1742) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<4xi32>
  %1744 = "tf.BroadcastTo"(%1732, %1743) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims1745 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1745 = "tf.Min"(%1744, %dims1745) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %1746 = "tf.Erfc"(%1745) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1747 = "tf.Cosh"(%1746) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1748 = "tf.Neg"(%1747) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1749 = "tf.Digamma"(%1748) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1750 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1750 = "tf.Min"(%1749, %dims1750) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %dims1751 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1751 = "tf.Mean"(%1750, %dims1751) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims1752 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1752 = "tf.Prod"(%1751, %dims1752) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<f32>
  %1753 = "tf.Abs"(%1752) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>) -> tensor<f32>
  %1754 = "tf.ZerosLike"(%1753) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>) -> tensor<f32>
  %1755 = "tf.Neg"(%1754) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>) -> tensor<f32>
  %1756 = "tf.Sqrt"(%1755) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>) -> tensor<f32>
  %1757 = "tf.BiasAdd"(%V__45, %V__31) { data_format = "NCHW", device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?xf32>) -> tensor<?x?x?x?xf32>
  %dims1758 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1758 = "tf.Mean"(%V__57, %dims1758) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %dims1759 = "tf.Const"() { value = dense<[3, 1, 0, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1759 = "tf.Transpose"(%1758, %dims1759) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %1760 = "tf.AddV2"(%1757, %1759) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1761 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1761 = "tf.Min"(%1760, %dims1761) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims1762 = "tf.Const"() { value = dense<[3, 1, 2, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1762 = "tf.Transpose"(%1761, %dims1762) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims1763 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1763 = "tf.Prod"(%1762, %dims1763) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %dims1764 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1764 = "tf.Prod"(%1763, %dims1764) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %dims1765 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1765 = "tf.Sum"(%1764, %dims1765) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %1766 = "tf.Abs"(%1765) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1767 = "tf.Round"(%1766) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1768 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1768 = "tf.Prod"(%1767, %dims1768) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %dims1769 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1769 = "tf.Mean"(%1768, %dims1769) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %dims1770 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1770 = "tf.Mean"(%1769, %dims1770) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %dims1771 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1771 = "tf.Min"(%1770, %dims1771) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?xf32>
  %dims1772 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1772 = "tf.Sum"(%1771, %dims1772) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>, tensor<1xi32>) -> tensor<f32>
  %1773 = "tf.Sign"(%1772) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>) -> tensor<f32>
  %1774 = "tf.Log"(%1773) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>) -> tensor<f32>
  %dims1775 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1775 = "tf.Max"(%V__45, %dims1775) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?xf32>
  %1776 = "tf.Sqrt"(%1775) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %dims1777 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1777 = "tf.Sum"(%1776, %dims1777) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?xf32>
  %dims1778 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1778 = "tf.Prod"(%1777, %dims1778) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  %dims1779 = "tf.Const"() { value = dense<[1, 2, 0]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1779 = "tf.Transpose"(%1778, %dims1779) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  %1780 = "tf.OnesLike"(%1779) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %dims1781 = "tf.Const"() { value = dense<[1, 0, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1781 = "tf.Transpose"(%1780, %dims1781) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  %dims1782 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1782 = "tf.Mean"(%1781, %dims1782) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<1xi32>) -> tensor<?x?xf32>
  %dims1783 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1783 = "tf.Max"(%V__7, %dims1783) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<1xi32>) -> tensor<?x?xi32>
  %1784 = "tf.Const"() { value = dense<[1, 78, 1, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1785 = "tf.BroadcastTo"(%1783, %1784) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims1786 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1786 = "tf.Min"(%1785, %dims1786) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %1787 = "tf.Squeeze"(%1786) { squeeze_dims = [ 0 : i64, 1 : i64 ], device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?xi32>
  %1788 = "tf.Cast"(%1787) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>) -> tensor<?x?xf32>
  %1789 = "tf.Sinh"(%1788) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %1790 = "tf.Selu"(%1789) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %1791 = "tf.DivNoNan"(%1782, %1790) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %1792 = "tf.Cos"(%1791) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %dims1793 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1793 = "tf.Min"(%1792, %dims1793) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>, tensor<2xi32>) -> tensor<f32>
  %1794 = "tf.Mod"(%1774, %1793) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %1795 = "tf.Round"(%1794) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>) -> tensor<f32>
  %1796 = "tf.RealDiv"(%1756, %1795) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %1797 = "tf.Round"(%1796) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>) -> tensor<f32>
  %1798 = "tf.Cast"(%V__35) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>) -> tensor<i1>
  %1799 = "tf.Select"(%1798, %V__56, %V__50) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i1>, tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %dims1800 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1800 = "tf.All"(%V__54, %dims1800) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi1>, tensor<1xi32>) -> tensor<?x?x?xi1>
  %1801 = "tf.SelectV2"(%1800, %V__38, %V__38) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi1>, tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %1802 = "tf.Xlog1py"(%1799, %1801) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %1803 = "tf.Rsqrt"(%1802) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %dims1804 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1804 = "tf.Min"(%1803, %dims1804) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?xf32>
  %dims1805 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1805 = "tf.Min"(%1804, %dims1805) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<2xi32>) -> tensor<?xf32>
  %1806 = "tf.Const"() { value = dense<[20, 44, 31, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1807 = "tf.BroadcastTo"(%1805, %1806) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims1808 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1808 = "tf.Min"(%1807, %dims1808) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?xf32>
  %dims1809 = "tf.Const"() { value = dense<[0, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1809 = "tf.All"(%V__4, %dims1809) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<3xi32>) -> tensor<?x?x?x?xi1>
  %dims1810 = "tf.Const"() { value = dense<[1, 2, 3, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1810 = "tf.Transpose"(%1809, %dims1810) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<4xi32>) -> tensor<?x?x?x?xi1>
  %dims1811 = "tf.Const"() { value = dense<[3, 2, 0, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1811 = "tf.Transpose"(%V__57, %dims1811) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %1812 = "tf.Erf"(%1811) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1813 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1813 = "tf.Min"(%V__45, %dims1813) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %1814 = "tf.Select"(%1810, %1812, %1813) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1815 = "tf.Squeeze"(%1814) { squeeze_dims = [ 0 : i64 ], device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?xf32>
  %dims1816 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1816 = "tf.Max"(%1815, %dims1816) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?xf32>
  %dims1817 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1817 = "tf.Prod"(%1816, %dims1817) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?xf32>
  %1818 = "tf.Minimum"(%V__45, %V__43) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1819 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1819 = "tf.Mean"(%1818, %dims1819) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %dims1820 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1820 = "tf.Max"(%1819, %dims1820) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %dims1821 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1821 = "tf.Min"(%1820, %dims1821) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?xf32>
  %1822 = "tf.Acos"(%V__38) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %1823 = "tf.Asinh"(%1822) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %1824 = "tf.Rint"(%1823) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %1825 = "tf.Asin"(%1824) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %1826 = "tf.DivNoNan"(%1821, %1825) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %1827 = "tf.Mod"(%1817, %1826) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %dims1828 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1828 = "tf.Min"(%1827, %dims1828) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?xf32>
  %dims1829 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1829 = "tf.Min"(%1828, %dims1829) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<1xi32>) -> tensor<?x?xf32>
  %dims1830 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1830 = "tf.Min"(%1829, %dims1830) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  %dims1831 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1831 = "tf.Mean"(%1830, %dims1831) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>, tensor<1xi32>) -> tensor<?xf32>
  %1832 = "tf.Cosh"(%V__50) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %1833 = "tf.Elu"(%1832) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %1834 = "tf.Atanh"(%1833) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %dims1835 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1835 = "tf.Mean"(%1834, %dims1835) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?xf32>
  %dims1836 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1836 = "tf.Min"(%V__3, %dims1836) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?x?xi64>
  %dims1837 = "tf.Const"() { value = dense<[2, 3, 1, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1837 = "tf.Transpose"(%1836, %dims1837) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims1838 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1838 = "tf.Prod"(%1837, %dims1838) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %1839 = "tf.Shape"(%1838) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<4xi32>
  %1840 = "tf.BroadcastTo"(%1835, %1839) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims1841 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1841 = "tf.Mean"(%1840, %dims1841) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %1842 = "tf.LeakyRelu"(%V__50) { alpha = 2.000000e-01 : f32, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %1843 = "tf.Const"() { value = dense<[1, 1, 1, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1844 = "tf.BroadcastTo"(%1842, %1843) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %1845 = "tf.Elu"(%1844) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1846 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1846 = "tf.Prod"(%V__57, %dims1846) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %dims1847 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1847 = "tf.Max"(%1846, %dims1847) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %dims1848 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1848 = "tf.Min"(%1847, %dims1848) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %1849 = "tf.Expm1"(%1848) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1850 = "tf.Div"(%1845, %1849) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1851 = "tf.Const"() { value = dense<[0, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1851 = "tf.Max"(%1850, %dims1851) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?xf32>
  %1852 = "tf.BiasAdd"(%1841, %1851) { data_format = "NCHW", device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?xf32>) -> tensor<?x?x?x?xf32>
  %dims1853 = "tf.Const"() { value = dense<[2, 1, 3, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1853 = "tf.Transpose"(%1852, %dims1853) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims1854 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1854 = "tf.Sum"(%1853, %dims1854) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?xf32>
  %1855 = "tf.Sqrt"(%1854) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>) -> tensor<?xf32>
  %1856 = "tf.Relu"(%1855) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>) -> tensor<?xf32>
  %1857 = "tf.ClipByValue"(%1808, %1831, %1856) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %1858 = "tf.Round"(%1857) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>) -> tensor<?xf32>
  %1859 = "tf.Squeeze"(%1858) { squeeze_dims = [ 0 : i64 ], device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>) -> tensor<f32>
  %1860 = "tf.Reciprocal"(%1859) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>) -> tensor<f32>
  %1861 = "tf.Less"(%1797, %1860) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %1862 = "tf.Cast"(%V__37) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>) -> tensor<?x?xi32>
  %dims1863 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1863 = "tf.Sum"(%1862, %dims1863) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>, tensor<1xi32>) -> tensor<?xi32>
  %1864 = "tf.Const"() { value = dense<[78, 1, 1, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1865 = "tf.BroadcastTo"(%1863, %1864) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims1866 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1866 = "tf.Prod"(%1865, %dims1866) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?xi32>
  %1867 = "tf.Abs"(%1866) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  %dims1868 = "tf.Const"() { value = dense<[2, 1, 0]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1868 = "tf.Transpose"(%1867, %dims1868) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?xi32>
  %dims1869 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1869 = "tf.Sum"(%V__0, %dims1869) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?xi32>
  %1870 = "tf.Const"() { value = dense<[1, 1, 1]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %dims1872 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1872 = "tf.Max"(%V__7, %dims1872) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?xi32>
  %1873 = "tf.Mod"(%1872, %V__7) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  %1874 = "tf.LeftShift"(%1869, %1873) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  %dims1875 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1875 = "tf.Prod"(%1874, %dims1875) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?xi32>
  %1876 = "tf.SquaredDifference"(%1868, %1875) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  %dims1877 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1877 = "tf.Mean"(%V__20, %dims1877) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?xi32>
  %1878 = "tf.Const"() { value = dense<[95, 1, 1]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1879 = "tf.BroadcastTo"(%1877, %1878) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>, tensor<3xi32>) -> tensor<?x?x?xi32>
  %dims1880 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1880 = "tf.Sum"(%1879, %dims1880) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?xi32>
  %dims1881 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1881 = "tf.Prod"(%1880, %dims1881) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?xi32>
  %dims1882 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1882 = "tf.Prod"(%1881, %dims1882) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?xi32>
  %dims1883 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1883 = "tf.Prod"(%V__9, %dims1883) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims1884 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1884 = "tf.Sum"(%1883, %dims1884) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?xi32>
  %dims1885 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1885 = "tf.Sum"(%V__7, %dims1885) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?xi32>
  %1886 = "tf.TruncateDiv"(%1884, %1885) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  %dims1887 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1887 = "tf.Max"(%1886, %dims1887) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?xi32>
  %1888 = "tf.Minimum"(%1882, %1887) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  %1889 = "tf.Round"(%1888) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  %1890 = "tf.ZerosLike"(%1889) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  %dims1891 = "tf.Const"() { value = dense<[0, 2, 1]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1891 = "tf.Transpose"(%1890, %dims1891) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?xi32>
  %dims1892 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1892 = "tf.Max"(%1891, %dims1892) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?xi32>
  %1893 = "tf.FloorDiv"(%1876, %1892) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  %1894 = "tf.Round"(%1893) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  %1895 = "tf.Neg"(%1894) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  %dims1896 = "tf.Const"() { value = dense<[1, 3, 2, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1896 = "tf.Transpose"(%V__32, %dims1896) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims1897 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1897 = "tf.Prod"(%1896, %dims1897) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?xi32>
  %dims1898 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1898 = "tf.Mean"(%1897, %dims1898) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>, tensor<1xi32>) -> tensor<?xi32>
  %dims1899 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1899 = "tf.Prod"(%V__0, %dims1899) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?xi32>
  %dims1900 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1900 = "tf.Mean"(%1899, %dims1900) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<2xi32>) -> tensor<?xi32>
  %1901 = "tf.Minimum"(%1898, %1900) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
  %1902 = "tf.Const"() { value = dense<[57, 3, 47, 77]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1903 = "tf.BroadcastTo"(%1901, %1902) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims1904 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1904 = "tf.Prod"(%1903, %dims1904) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims1905 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1905 = "tf.Min"(%1904, %dims1905) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims1906 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1906 = "tf.Sum"(%1905, %dims1906) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %1907 = "tf.Squeeze"(%1906) { squeeze_dims = [ 0 : i64 ], device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?xi32>
  %dims1908 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1908 = "tf.Min"(%1907, %dims1908) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?xi32>
  %1909 = "tf.Squeeze"(%V__13) { squeeze_dims = [ 1 : i64 ], device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>) -> tensor<?x?x?xi1>
  %dims1910 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1910 = "tf.Any"(%1909, %dims1910) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi1>, tensor<2xi32>) -> tensor<?x?x?xi1>
  %dims1911 = "tf.Const"() { value = dense<[0, 1, 3, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1911 = "tf.Transpose"(%V__20, %dims1911) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %1912 = "tf.Squeeze"(%1911) { squeeze_dims = [ 3 : i64 ], device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?xi32>
  %dims1913 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1913 = "tf.Max"(%V__32, %dims1913) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?xi32>
  %1914 = "tf.Select"(%1910, %1912, %1913) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi1>, tensor<?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  %dims1915 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1915 = "tf.Max"(%1914, %dims1915) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?xi32>
  %1916 = "tf.Sqrt"(%V__28) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1917 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1917 = "tf.Max"(%1916, %dims1917) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %1918 = "tf.Square"(%V__41) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1919 = "tf.Mul"(%1917, %1918) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1920 = "tf.Lgamma"(%1919) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1921 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1921 = "tf.Sum"(%1920, %dims1921) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?xf32>
  %1922 = "tf.Cast"(%1921) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xi32>
  %1923 = "tf.Add"(%1915, %1922) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  %1924 = "tf.Equal"(%1908, %1923) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi1>
  %1925 = "tf.Cast"(%1924) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi1>) -> tensor<?x?x?xi32>
  %1926 = "tf.BitwiseOr"(%1895, %1925) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  %dims1927 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1927 = "tf.Mean"(%1926, %dims1927) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<1xi32>) -> tensor<?x?xi32>
  %1928 = "tf.Round"(%1927) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>) -> tensor<?x?xi32>
  %dims1929 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1929 = "tf.Max"(%1928, %dims1929) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>, tensor<1xi32>) -> tensor<?xi32>
  %dims1930 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1930 = "tf.Any"(%V__58, %dims1930) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<3xi32>) -> tensor<?x?x?x?xi1>
  %1931 = "tf.SelectV2"(%1930, %V__59, %V__3) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims1932 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1932 = "tf.Sum"(%1931, %dims1932) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims1933 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1933 = "tf.Sum"(%V__49, %dims1933) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %1934 = "tf.SquaredDifference"(%1933, %V__49) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims1935 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1935 = "tf.Min"(%1934, %dims1935) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %dims1936 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1936 = "tf.Mean"(%1935, %dims1936) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %1937 = "tf.GreaterEqual"(%1932, %1936) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi1>
  %1938 = "tf.Squeeze"(%1937) { squeeze_dims = [ 0 : i64, 1 : i64, 2 : i64 ], device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>) -> tensor<?xi1>
  %1939 = "tf.Const"() { value = dense<[1, 1, 1]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1940 = "tf.BroadcastTo"(%1938, %1939) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi1>, tensor<3xi32>) -> tensor<?x?x?xi1>
  %1941 = "tf.Squeeze"(%1940) { squeeze_dims = [ 0 : i64, 1 : i64 ], device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi1>) -> tensor<?xi1>
  %dims1942 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1942 = "tf.Min"(%V__9, %dims1942) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<i32>
  %dims1943 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1943 = "tf.Prod"(%V__60, %dims1943) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<i32>
  %1944 = "tf.GreaterEqual"(%1942, %1943) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %dims1945 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1945 = "tf.Min"(%V__52, %dims1945) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %1946 = "tf.Zeta"(%1945, %V__39) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1947 = "tf.Shape"(%1946) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<4xi32>
  %1948 = "tf.Reshape"(%1944, %1947) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i1>, tensor<4xi32>) -> tensor<?x?x?x?xi1>
  %1949 = "tf.LogicalNot"(%1948) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>) -> tensor<?x?x?x?xi1>
  %dims1950 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1950 = "tf.All"(%1949, %dims1950) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<3xi32>) -> tensor<?x?x?x?xi1>
  %1951 = "tf.Cast"(%1950) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>) -> tensor<?x?x?x?xi64>
  %1952 = "tf.Const"() { value = dense<[1, 29, 85, 24]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1953 = "tf.Fill"(%1952, %V__61) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<4xi32>, tensor<i64>) -> tensor<?x?x?x?xi64>
  %dims1954 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1954 = "tf.Mean"(%1953, %dims1954) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims1955 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1955 = "tf.Max"(%1954, %dims1955) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %dims1956 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1956 = "tf.Prod"(%1955, %dims1956) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %1957 = "tf.Relu"(%V__42) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1958 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1958 = "tf.Max"(%1957, %dims1958) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %1959 = "tf.Cast"(%1958) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xi32>
  %1960 = "tf.Shape"(%1959) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<4xi32>
  %1961 = "tf.BroadcastTo"(%1956, %1960) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims1962 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1962 = "tf.Min"(%1961, %dims1962) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?x?xi64>
  %1963 = "tf.TruncateDiv"(%1951, %1962) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims1964 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1964 = "tf.Prod"(%1963, %dims1964) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?x?xi64>
  %dims1965 = "tf.Const"() { value = dense<[1, 2, 0, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1965 = "tf.Transpose"(%1964, %dims1965) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %1966 = "tf.Square"(%V__26) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i32>) -> tensor<i32>
  %1967 = "tf.Mod"(%1966, %V__26) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %dims1968 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1968 = "tf.Min"(%V__2, %dims1968) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<i32>
  %1969 = "tf.BitwiseXor"(%1968, %V__26) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %1970 = "tf.Equal"(%1967, %1969) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %dims1971 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1971 = "tf.Mean"(%V__3, %dims1971) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?x?xi64>
  %1972 = "tf.ZerosLike"(%1971) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims1973 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1973 = "tf.Min"(%1972, %dims1973) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %dims1974 = "tf.Const"() { value = dense<[2, 1, 0, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1974 = "tf.Transpose"(%V__23, %dims1974) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %1975 = "tf.LeftShift"(%1974, %V__62) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %1976 = "tf.Sub"(%1973, %1975) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %1977 = "tf.Const"() { value = dense<[1, 1, 5, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1979 = "tf.Const"() { value = dense<[1, 1, 5, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1980 = "tf.BroadcastTo"(%V__63, %1979) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims1981 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1981 = "tf.Mean"(%1980, %dims1981) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %1982 = "tf.Select"(%1970, %1976, %1981) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i1>, tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims1983 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1983 = "tf.Min"(%1982, %dims1983) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %dims1984 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1984 = "tf.Min"(%1983, %dims1984) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %1985 = "tf.Select"(%1941, %1965, %1984) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi1>, tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %1986 = "tf.Shape"(%1985) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<4xi32>
  %1987 = "tf.BroadcastTo"(%1929, %1986) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %1988 = "tf.Pow"(%V__64, %V__65) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims1989 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1989 = "tf.Sum"(%V__20, %dims1989) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims1990 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1990 = "tf.Max"(%1989, %dims1990) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %1991 = "tf.BitwiseAnd"(%1988, %1990) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims1992 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1992 = "tf.Sum"(%1991, %dims1992) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims1993 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1993 = "tf.Min"(%1992, %dims1993) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?xi32>
  %dims1994 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1994 = "tf.Prod"(%1993, %dims1994) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?xi32>
  %dims1995 = "tf.Const"() { value = dense<[1, 2, 0]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1995 = "tf.Transpose"(%V__7, %dims1995) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?xi32>
  %dims1996 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1996 = "tf.Sum"(%1995, %dims1996) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?xi32>
  %1997 = "tf.BitwiseOr"(%V__0, %V__7) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  %1998 = "tf.BitwiseOr"(%1996, %1997) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  %dims1999 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1999 = "tf.Min"(%1998, %dims1999) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?xi32>
  %dims2000 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2000 = "tf.Sum"(%1999, %dims2000) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?xi32>
  %2001 = "tf.BitwiseOr"(%1994, %2000) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  %2002 = "tf.Relu6"(%V__26) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i32>) -> tensor<i32>
  %2003 = "tf.Sign"(%V__26) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i32>) -> tensor<i32>
  %2004 = "tf.Maximum"(%2002, %2003) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %dims2005 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2005 = "tf.Sum"(%V__0, %dims2005) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<3xi32>) -> tensor<i32>
  %2006 = "tf.Minimum"(%2005, %V__66) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %2007 = "tf.Div"(%2004, %2006) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %dims2008 = "tf.Const"() { value = dense<[0, 2, 1, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2008 = "tf.Transpose"(%V__1, %dims2008) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims2009 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2009 = "tf.Max"(%2008, %dims2009) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?xf32>
  %dims2010 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2010 = "tf.Sum"(%2009, %dims2010) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  %dims2011 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2011 = "tf.Max"(%2010, %dims2011) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?xf32>
  %dims2012 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2012 = "tf.Prod"(%2011, %dims2012) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?xf32>
  %dims2013 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2013 = "tf.Mean"(%2012, %dims2013) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?xf32>
  %dims2014 = "tf.Const"() { value = dense<[0, 2, 1]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2014 = "tf.Transpose"(%2013, %dims2014) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  %2015 = "tf.Shape"(%2014) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<3xi32>
  %2016 = "tf.Reshape"(%2007, %2015) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i32>, tensor<3xi32>) -> tensor<?x?x?xi32>
  %2017 = "tf.FloorMod"(%2001, %2016) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  %dims2018 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2018 = "tf.Max"(%2017, %dims2018) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?xi32>
  %dims2019 = "tf.Const"() { value = dense<[0, 2, 1]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2019 = "tf.Transpose"(%2018, %dims2019) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?xi32>
  %2020 = "tf.Relu"(%2019) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  %dims2021 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2021 = "tf.Prod"(%2020, %dims2021) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?xi32>
  %2022 = "tf.Const"() { value = dense<[1, 1, 1, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2024 = "tf.Neg"(%V__16) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>) -> tensor<?x?xi32>
  %dims2025 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2025 = "tf.Prod"(%2024, %dims2025) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>, tensor<2xi32>) -> tensor<?x?xi32>
  %dims2026 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2026 = "tf.Sum"(%2025, %dims2026) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>, tensor<1xi32>) -> tensor<?x?xi32>
  %2027 = "tf.Const"() { value = dense<[60, 91, 1, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2028 = "tf.BroadcastTo"(%2026, %2027) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims2029 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2029 = "tf.Min"(%2028, %dims2029) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %2030 = "tf.Const"() { value = dense<[1, 97, 1, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2031 = "tf.BroadcastTo"(%V__16, %2030) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims2032 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2032 = "tf.Max"(%2031, %dims2032) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %2033 = "tf.Cast"(%V__9) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi64>
  %dims2034 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2034 = "tf.Mean"(%2033, %dims2034) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %2035 = "tf.Shape"(%2034) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<4xi32>
  %2037 = "tf.Div"(%2029, %2032) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %2038 = "tf.Relu6"(%2037) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %2039 = "tf.ZerosLike"(%2038) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims2040 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2040 = "tf.Min"(%V__2, %dims2040) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %2041 = "tf.AddV2"(%2040, %V__32) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims2042 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2042 = "tf.Mean"(%V__16, %dims2042) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>, tensor<2xi32>) -> tensor<i32>
  %2043 = "tf.Const"() { value = dense<[1, 1, 1, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2044 = "tf.Reshape"(%2042, %2043) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %2045 = "tf.Pow"(%2041, %2044) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims2046 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2046 = "tf.Prod"(%V__20, %dims2046) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %2047 = "tf.BiasAdd"(%2046, %V__10) { data_format = "NCHW", device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?xi32>) -> tensor<?x?x?x?xi32>
  %2048 = "tf.Const"() { value = dense<[1, 1, 1, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2050 = "tf.FloorMod"(%2047, %V__67) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %2051 = "tf.Mod"(%2045, %2050) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims2052 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2052 = "tf.Max"(%2051, %dims2052) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %2053 = "tf.Cast"(%2052) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %2054 = "tf.TruncateDiv"(%2039, %2053) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %2055 = "tf.Invert"(%2054) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %2056 = "tf.Invert"(%2055) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims2057 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2057 = "tf.Mean"(%V__18, %dims2057) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims2058 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2058 = "tf.Mean"(%V__32, %dims2058) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %2059 = "tf.BitwiseOr"(%2057, %2058) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims2060 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2060 = "tf.Sum"(%V__20, %dims2060) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims2061 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2061 = "tf.Min"(%2060, %dims2061) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims2062 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2062 = "tf.Prod"(%2061, %dims2062) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?xi32>
  %2063 = "tf.Squeeze"(%2062) { squeeze_dims = [ 1 : i64, 2 : i64 ], device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>) -> tensor<?xi32>
  %2064 = "tf.BiasAdd"(%2059, %2063) { data_format = "NCHW", device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?xi32>) -> tensor<?x?x?x?xi32>
  %dims2065 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2065 = "tf.Prod"(%2064, %dims2065) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims2066 = "tf.Const"() { value = dense<[0, 2, 3, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2066 = "tf.Transpose"(%2065, %dims2066) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims2067 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2067 = "tf.Min"(%2066, %dims2067) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims2068 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2068 = "tf.Sum"(%2067, %dims2068) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims2069 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2069 = "tf.Max"(%2068, %dims2069) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims2070 = "tf.Const"() { value = dense<[2, 3, 0, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2070 = "tf.Transpose"(%2069, %dims2070) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims2071 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2071 = "tf.Max"(%V__2, %dims2071) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims2072 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2072 = "tf.Min"(%2071, %dims2072) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims2073 = "tf.Const"() { value = dense<[1, 0, 3, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2073 = "tf.Transpose"(%V__64, %dims2073) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %2074 = "tf.Shape"(%2073) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<4xi32>
  %2076 = "tf.Const"() { value = dense<[1, 1, 1, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %dims2078 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2078 = "tf.Min"(%V__64, %dims2078) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %2079 = "tf.Div"(%V__0, %2078) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %2080 = "tf.Add"(%2072, %2079) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims2081 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2081 = "tf.Sum"(%2080, %dims2081) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %2082 = "tf.Shape"(%2081) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<4xi32>
  %2083 = "tf.BroadcastTo"(%2070, %2082) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims2084 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2084 = "tf.Sum"(%2083, %dims2084) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims2085 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2085 = "tf.Mean"(%2084, %dims2085) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims2086 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2086 = "tf.Sum"(%2085, %dims2086) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims2087 = "tf.Const"() { value = dense<[3, 1, 2, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2087 = "tf.Transpose"(%2086, %dims2087) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims2088 = "tf.Const"() { value = dense<[2, 0, 3, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2088 = "tf.Transpose"(%2087, %dims2088) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims2089 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2089 = "tf.Max"(%2088, %dims2089) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims2090 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2090 = "tf.Min"(%2089, %dims2090) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims2091 = "tf.Const"() { value = dense<[0, 2, 1, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2091 = "tf.Transpose"(%2090, %dims2091) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims2092 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2092 = "tf.Mean"(%2091, %dims2092) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %2093 = "tf.AddV2"(%2056, %2092) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %2094 = "tf.Abs"(%2093) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims2095 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2095 = "tf.Sum"(%2094, %dims2095) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %2096 = "tf.RightShift"(%2021, %2095) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims2097 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2097 = "tf.Mean"(%2096, %dims2097) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %2098 = "tf.Select"(%1861, %1987, %2097) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i1>, tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %2099 = "tf.Relu"(%2098) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %2100 = "tf.Relu6"(%2099) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims2101 = "tf.Const"() { value = dense<[0, 1, 3, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2101 = "tf.Transpose"(%2100, %dims2101) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims2102 = "tf.Const"() { value = dense<[3, 1, 2, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2102 = "tf.Transpose"(%2101, %dims2102) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %2103 = "tf.Const"() { value = dense<[1, 1, 1, 26]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2104 = "tf.BroadcastTo"(%2102, %2103) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims2105 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2105 = "tf.Mean"(%2104, %dims2105) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %2106 = "tf.Select"(%853, %1718, %2105) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i1>, tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims2107 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2107 = "tf.Min"(%2106, %dims2107) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %2108 = "tf.Cast"(%2107) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi1>
  %dims2109 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2109 = "tf.Any"(%2108, %dims2109) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<1xi32>) -> tensor<?x?x?xi1>
  %dims2110 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2110 = "tf.All"(%2109, %dims2110) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi1>, tensor<2xi32>) -> tensor<?x?x?xi1>
  %dims2111 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2111 = "tf.Any"(%2110, %dims2111) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi1>, tensor<2xi32>) -> tensor<?xi1>
  %2112 = "tf.Const"() { value = dense<[1, 61, 1, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2113 = "tf.BroadcastTo"(%V__33, %2112) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %2114 = "tf.ZerosLike"(%2113) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims2115 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2115 = "tf.Mean"(%2114, %dims2115) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims2116 = "tf.Const"() { value = dense<[3, 0, 1, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2116 = "tf.Transpose"(%V__20, %dims2116) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims2117 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2117 = "tf.Sum"(%2116, %dims2117) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %2118 = "tf.Abs"(%2117) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %2119 = "tf.FloorDiv"(%2115, %2118) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %2120 = "tf.Shape"(%2119) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<4xi32>
  %dims2121 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2121 = "tf.Mean"(%V__6, %dims2121) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<2xi32>) -> tensor<?xi64>
  %dims2122 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2122 = "tf.Max"(%2121, %dims2122) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi64>, tensor<1xi32>) -> tensor<i64>
  %2123 = "tf.Relu6"(%2122) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i64>) -> tensor<i64>
  %2124 = "tf.Cast"(%2123) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i64>) -> tensor<f32>
  %2125 = "tf.FloorMod"(%V__34, %V__48) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %2126 = "tf.LeakyRelu"(%2125) { alpha = 3.000000e-01 : f32, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>) -> tensor<f32>
  %2127 = "tf.SquaredDifference"(%2124, %2126) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %2128 = "tf.Tanh"(%2127) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>) -> tensor<f32>
  %2129 = "tf.Fill"(%2120, %2128) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<4xi32>, tensor<f32>) -> tensor<?x?x?x?xf32>
  %dims2130 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2130 = "tf.Sum"(%2129, %dims2130) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %2131 = "tf.Cast"(%2130) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims2132 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2132 = "tf.Mean"(%V__68, %dims2132) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %2133 = "tf.Cos"(%2132) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims2134 = "tf.Const"() { value = dense<[0, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2134 = "tf.Max"(%2133, %dims2134) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %2135 = "tf.Xlogy"(%V__51, %V__51) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %2136 = "tf.Equal"(%2134, %2135) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xi1>
  %dims2137 = "tf.Const"() { value = dense<[1, 2, 3, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2137 = "tf.Transpose"(%V__27, %dims2137) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims2138 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2138 = "tf.Min"(%V__41, %dims2138) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %2139 = "tf.Xlog1py"(%2137, %2138) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims2140 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2140 = "tf.Min"(%2139, %dims2140) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims2141 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2141 = "tf.Max"(%2140, %dims2141) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims2142 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2142 = "tf.Mean"(%2141, %dims2142) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims2143 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2143 = "tf.Max"(%V__41, %dims2143) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims2144 = "tf.Const"() { value = dense<[0, 2, 3, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2144 = "tf.Transpose"(%2143, %dims2144) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims2145 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2145 = "tf.Prod"(%2144, %dims2145) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %2146 = "tf.Pow"(%V__69, %V__52) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %2147 = "tf.AddV2"(%2145, %2146) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %2148 = "tf.SelectV2"(%2136, %2142, %2147) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %2149 = "tf.Div"(%2131, %2148) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %2150 = "tf.BitwiseOr"(%V__65, %V__14) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %2151 = "tf.Cast"(%2150) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xf32>
  %2152 = "tf.Log1p"(%2151) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims2153 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2153 = "tf.Sum"(%V__53, %dims2153) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %2154 = "tf.Atanh"(%2153) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims2155 = "tf.Const"() { value = dense<[1, 2, 0, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2155 = "tf.Transpose"(%2154, %dims2155) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims2156 = "tf.Const"() { value = dense<[3, 1, 0, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2156 = "tf.Transpose"(%2155, %dims2156) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %2157 = "tf.RealDiv"(%2152, %2156) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %2158 = "tf.Sin"(%2157) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims2159 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2159 = "tf.Sum"(%V__38, %dims2159) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<2xi32>) -> tensor<?xf32>
  %2160 = "tf.Tan"(%V__31) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>) -> tensor<?xf32>
  %2161 = "tf.RealDiv"(%2159, %2160) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %dims2162 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2162 = "tf.Sum"(%V__52, %dims2162) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims2163 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2163 = "tf.Max"(%2162, %dims2163) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims2164 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2164 = "tf.Sum"(%2163, %dims2164) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims2165 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2165 = "tf.Max"(%2164, %dims2165) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?xf32>
  %2166 = "tf.Mul"(%2161, %2165) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %2167 = "tf.BiasAdd"(%2158, %2166) { data_format = "NHWC", device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?xf32>) -> tensor<?x?x?x?xf32>
  %2168 = "tf.Floor"(%2167) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims2169 = "tf.Const"() { value = dense<[3, 1, 0, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2169 = "tf.Transpose"(%V__46, %dims2169) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %2170 = "tf.Cos"(%2169) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims2171 = "tf.Const"() { value = dense<[0, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2171 = "tf.Min"(%2170, %dims2171) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %dims2172 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2172 = "tf.Prod"(%V__45, %dims2172) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims2173 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2173 = "tf.Max"(%2172, %dims2173) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %2174 = "tf.Selu"(%2173) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %2175 = "tf.Add"(%2171, %2174) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims2176 = "tf.Const"() { value = dense<[3, 2, 0, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2176 = "tf.Transpose"(%2175, %dims2176) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %2177 = "tf.Cos"(%V__1) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %2178 = "tf.Asinh"(%2177) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims2179 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2179 = "tf.Max"(%2178, %dims2179) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %2180 = "tf.Sin"(%V__39) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %2181 = "tf.LeakyRelu"(%2180) { alpha = 2.000000e-01 : f32, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims2182 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2182 = "tf.Min"(%2181, %dims2182) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %2183 = "tf.Zeta"(%2179, %2182) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims2184 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2184 = "tf.Min"(%2183, %dims2184) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %2185 = "tf.Div"(%2176, %2184) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims2186 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2186 = "tf.Min"(%2185, %dims2186) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %dims2187 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2187 = "tf.Min"(%2186, %dims2187) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %2188 = "tf.Digamma"(%2187) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %2189 = "tf.AddV2"(%2168, %2188) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %2190 = "tf.Maximum"(%2149, %2189) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %2191 = "tf.Cast"(%V__63) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i64>) -> tensor<i32>
  %2192 = "tf.Const"() { value = dense<[1, 1, 1, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2193 = "tf.Reshape"(%2191, %2192) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims2194 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2194 = "tf.Max"(%2193, %dims2194) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims2195 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2195 = "tf.Min"(%V__32, %dims2195) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims2196 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2196 = "tf.Sum"(%2195, %dims2196) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims2197 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2197 = "tf.Min"(%2196, %dims2197) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %2198 = "tf.TruncateDiv"(%2194, %2197) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims2199 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2199 = "tf.Prod"(%2198, %dims2199) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims2200 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2200 = "tf.Max"(%V__64, %dims2200) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %2201 = "tf.Mul"(%2200, %V__14) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims2202 = "tf.Const"() { value = dense<[0, 2, 1, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2202 = "tf.Transpose"(%2201, %dims2202) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %2203 = "tf.Neg"(%2202) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims2204 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2204 = "tf.Sum"(%2203, %dims2204) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims2205 = "tf.Const"() { value = dense<[2, 1, 3, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2205 = "tf.Transpose"(%2204, %dims2205) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims2206 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2206 = "tf.Sum"(%2205, %dims2206) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims2207 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2207 = "tf.Mean"(%2206, %dims2207) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims2208 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2208 = "tf.Max"(%2207, %dims2208) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %2209 = "tf.LeftShift"(%2199, %2208) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %2210 = "tf.BiasAdd"(%V__65, %V__24) { data_format = "NCHW", device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?xi32>) -> tensor<?x?x?x?xi32>
  %dims2211 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2211 = "tf.Mean"(%2210, %dims2211) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims2212 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2212 = "tf.Prod"(%V__2, %dims2212) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims2213 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2213 = "tf.Min"(%2212, %dims2213) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims2214 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2214 = "tf.Prod"(%2213, %dims2214) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %2215 = "tf.FloorMod"(%2211, %2214) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims2216 = "tf.Const"() { value = dense<[3, 0, 1, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2216 = "tf.Transpose"(%V__70, %dims2216) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims2217 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2217 = "tf.Prod"(%2216, %dims2217) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %2218 = "tf.Digamma"(%2217) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims2219 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2219 = "tf.Prod"(%2218, %dims2219) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %dims2220 = "tf.Const"() { value = dense<[0, 2, 3, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2220 = "tf.Transpose"(%2219, %dims2220) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims2221 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2221 = "tf.Sum"(%2220, %dims2221) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %2222 = "tf.Shape"(%2221) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<4xi32>
  %2223 = "tf.BroadcastTo"(%2215, %2222) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims2224 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2224 = "tf.Prod"(%2223, %dims2224) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims2225 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2225 = "tf.Prod"(%2224, %dims2225) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %2226 = "tf.Cast"(%2225) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xf32>
  %2227 = "tf.Shape"(%2226) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<4xi32>
  %2228 = "tf.BroadcastTo"(%2209, %2227) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims2229 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2229 = "tf.Min"(%2228, %dims2229) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims2230 = "tf.Const"() { value = dense<[2, 1, 0, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2230 = "tf.Transpose"(%2229, %dims2230) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims2231 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2231 = "tf.Mean"(%2230, %dims2231) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims2232 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2232 = "tf.Any"(%V__13, %dims2232) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<3xi32>) -> tensor<?x?x?x?xi1>
  %2233 = "tf.Shape"(%V__60) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<4xi32>
  %2234 = "tf.BroadcastTo"(%2232, %2233) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<4xi32>) -> tensor<?x?x?x?xi1>
  %dims2235 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2235 = "tf.All"(%2234, %dims2235) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<1xi32>) -> tensor<?x?x?x?xi1>
  %dims2236 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2236 = "tf.Any"(%2235, %dims2236) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<1xi32>) -> tensor<?x?x?x?xi1>
  %dims2237 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2237 = "tf.Mean"(%V__20, %dims2237) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims2238 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2238 = "tf.Mean"(%V__16, %dims2238) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>, tensor<1xi32>) -> tensor<?xi32>
  %2239 = "tf.BiasAdd"(%2237, %2238) { data_format = "NHWC", device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?xi32>) -> tensor<?x?x?x?xi32>
  %dims2240 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2240 = "tf.Sum"(%2239, %dims2240) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %2241 = "tf.Shape"(%2240) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<4xi32>
  %2242 = "tf.BroadcastTo"(%2236, %2241) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<4xi32>) -> tensor<?x?x?x?xi1>
  %dims2243 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2243 = "tf.Any"(%2242, %dims2243) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<1xi32>) -> tensor<?x?x?x?xi1>
  %dims2244 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2244 = "tf.Mean"(%V__9, %dims2244) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?xi32>
  %2245 = "tf.Shape"(%V__4) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>) -> tensor<4xi32>
  %2246 = "tf.BroadcastTo"(%2244, %2245) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims2247 = "tf.Const"() { value = dense<[2, 3, 1, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2247 = "tf.Transpose"(%2246, %dims2247) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims2248 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2248 = "tf.Mean"(%2247, %dims2248) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims2249 = "tf.Const"() { value = dense<[1, 0, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2249 = "tf.Transpose"(%2248, %dims2249) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %2250 = "tf.Atanh"(%V__53) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %2251 = "tf.Mul"(%2250, %V__42) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims2252 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2252 = "tf.Max"(%2251, %dims2252) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims2253 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2253 = "tf.Min"(%2252, %dims2253) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %2254 = "tf.Shape"(%2253) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<4xi32>
  %dims2256 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2256 = "tf.Mean"(%V__14, %dims2256) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims2257 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2257 = "tf.Mean"(%V__60, %dims2257) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %2258 = "tf.Mul"(%2256, %2257) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims2259 = "tf.Const"() { value = dense<[0, 1, 3, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2259 = "tf.Transpose"(%2258, %dims2259) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims2260 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2260 = "tf.Mean"(%2259, %dims2260) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims2261 = "tf.Const"() { value = dense<[2, 1, 0, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2261 = "tf.Transpose"(%V__60, %dims2261) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims2262 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2262 = "tf.Min"(%V__14, %dims2262) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %2263 = "tf.Sub"(%2261, %2262) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims2264 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2264 = "tf.Sum"(%2263, %dims2264) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %2265 = "tf.BitwiseXor"(%2260, %2264) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims2266 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2266 = "tf.Max"(%2265, %dims2266) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %2267 = "tf.SelectV2"(%2243, %2249, %2266) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims2268 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2268 = "tf.Min"(%2267, %dims2268) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %2269 = "tf.Maximum"(%2231, %2268) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %2270 = "tf.Cast"(%2269) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xf32>
  %2271 = "tf.Abs"(%2270) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %2272 = "tf.NotEqual"(%2190, %2271) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xi1>
  %dims2273 = "tf.Const"() { value = dense<[0, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2273 = "tf.Any"(%2272, %dims2273) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<3xi32>) -> tensor<?x?x?x?xi1>
  %dims2274 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2274 = "tf.Any"(%2273, %dims2274) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<3xi32>) -> tensor<?x?x?x?xi1>
  %dims2275 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2275 = "tf.All"(%V__13, %dims2275) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<2xi32>) -> tensor<?x?x?x?xi1>
  %dims2276 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2276 = "tf.Any"(%2275, %dims2276) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<2xi32>) -> tensor<?x?x?x?xi1>
  %dims2277 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2277 = "tf.Mean"(%V__32, %dims2277) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims2278 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2278 = "tf.Prod"(%V__64, %dims2278) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %2279 = "tf.Select"(%2276, %2277, %2278) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims2280 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2280 = "tf.Min"(%2279, %dims2280) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims2281 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2281 = "tf.Mean"(%2280, %dims2281) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims2282 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2282 = "tf.Max"(%2281, %dims2282) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims2283 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2283 = "tf.Sum"(%2282, %dims2283) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims2284 = "tf.Const"() { value = dense<[1, 2, 3, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2284 = "tf.Transpose"(%2283, %dims2284) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims2285 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2285 = "tf.Mean"(%2284, %dims2285) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %2286 = "tf.Abs"(%2285) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims2287 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2287 = "tf.Sum"(%2286, %dims2287) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims2288 = "tf.Const"() { value = dense<[3, 2, 0, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2288 = "tf.Transpose"(%2287, %dims2288) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %2289 = "tf.Neg"(%2288) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims2290 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2290 = "tf.Min"(%2289, %dims2290) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims2291 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2291 = "tf.Max"(%2290, %dims2291) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?xi32>
  %2292 = "tf.Sign"(%V__60) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %2293 = "tf.Cast"(%2292) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi64>
  %dims2294 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2294 = "tf.Max"(%2293, %dims2294) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims2295 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2295 = "tf.Sum"(%2294, %dims2295) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims2296 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2296 = "tf.Sum"(%2295, %dims2296) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %dims2297 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2297 = "tf.Sum"(%2296, %dims2297) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?xi64>
  %dims2298 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2298 = "tf.Sum"(%2297, %dims2298) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi64>, tensor<2xi32>) -> tensor<i64>
  %dims2299 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2299 = "tf.Sum"(%V__64, %dims2299) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims2300 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2300 = "tf.Prod"(%2299, %dims2300) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims2301 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2301 = "tf.Sum"(%2300, %dims2301) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %2302 = "tf.Cast"(%2301) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi64>
  %dims2303 = "tf.Const"() { value = dense<[3, 1, 0, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2303 = "tf.Transpose"(%2302, %dims2303) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %2304 = "tf.Shape"(%2303) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<4xi32>
  %2305 = "tf.BroadcastTo"(%2298, %2304) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %2306 = "tf.OnesLike"(%2305) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims2307 = "tf.Const"() { value = dense<[1, 3, 0, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2307 = "tf.Transpose"(%2306, %dims2307) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %2308 = "tf.Shape"(%2307) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<4xi32>
  %2309 = "tf.BroadcastTo"(%2291, %2308) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims2310 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2310 = "tf.Sum"(%2309, %dims2310) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims2311 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2311 = "tf.Max"(%2310, %dims2311) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims2312 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2312 = "tf.Max"(%2311, %dims2312) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims2313 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2313 = "tf.Sum"(%2312, %dims2313) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims2314 = "tf.Const"() { value = dense<[0, 3, 2, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2314 = "tf.Transpose"(%2313, %dims2314) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims2315 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2315 = "tf.Prod"(%2314, %dims2315) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims2316 = "tf.Const"() { value = dense<[2, 0, 1, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2316 = "tf.Transpose"(%2315, %dims2316) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims2317 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2317 = "tf.Sum"(%2316, %dims2317) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims2318 = "tf.Const"() { value = dense<[1, 3, 0, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2318 = "tf.Transpose"(%2317, %dims2318) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims2319 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2319 = "tf.Sum"(%2318, %dims2319) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %2320 = "tf.Const"() { value = dense<[1, 1, 1, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2321 = "tf.BroadcastTo"(%V__7, %2320) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims2322 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2322 = "tf.Sum"(%2321, %dims2322) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims2323 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2323 = "tf.Max"(%2322, %dims2323) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?xi32>
  %2324 = "tf.Cast"(%V__6) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>) -> tensor<?x?x?xi32>
  %dims2325 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2325 = "tf.Mean"(%V__0, %dims2325) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?xi32>
  %2326 = "tf.FloorDiv"(%2324, %2325) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  %2327 = "tf.Mod"(%2323, %2326) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  %dims2328 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2328 = "tf.Mean"(%2327, %dims2328) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<1xi32>) -> tensor<?x?xi32>
  %dims2329 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2329 = "tf.Prod"(%V__67, %dims2329) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<1xi32>) -> tensor<?x?xi32>
  %2330 = "tf.Const"() { value = dense<[1, 1, 1, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2332 = "tf.Mod"(%V__60, %V__20) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims2333 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2333 = "tf.Min"(%2332, %dims2333) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %2334 = "tf.BitwiseOr"(%2329, %2333) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims2335 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2335 = "tf.Max"(%2334, %dims2335) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %2336 = "tf.Shape"(%2335) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<4xi32>
  %2337 = "tf.BroadcastTo"(%2328, %2336) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %2338 = "tf.Invert"(%2337) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims2339 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2339 = "tf.Mean"(%V__9, %dims2339) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %2340 = "tf.Minimum"(%2339, %V__60) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims2341 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2341 = "tf.Max"(%2340, %dims2341) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims2342 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2342 = "tf.Sum"(%2341, %dims2342) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims2343 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2343 = "tf.Min"(%2342, %dims2343) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims2344 = "tf.Const"() { value = dense<[3, 2, 0, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2344 = "tf.Transpose"(%2343, %dims2344) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims2345 = "tf.Const"() { value = dense<[2, 1, 0, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2345 = "tf.Transpose"(%2344, %dims2345) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims2346 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2346 = "tf.Sum"(%2345, %dims2346) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %2347 = "tf.Abs"(%2346) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims2348 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2348 = "tf.Prod"(%2347, %dims2348) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims2349 = "tf.Const"() { value = dense<[3, 2, 1, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2349 = "tf.Transpose"(%V__20, %dims2349) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims2350 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2350 = "tf.Min"(%2349, %dims2350) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims2351 = "tf.Const"() { value = dense<[2, 0, 1, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2351 = "tf.Transpose"(%2350, %dims2351) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims2352 = "tf.Const"() { value = dense<[2, 1, 0, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2352 = "tf.Transpose"(%2351, %dims2352) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims2353 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2353 = "tf.Mean"(%V__32, %dims2353) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims2354 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2354 = "tf.Max"(%2353, %dims2354) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims2355 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2355 = "tf.Sum"(%2354, %dims2355) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %2356 = "tf.FloorMod"(%2352, %2355) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims2357 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2357 = "tf.Min"(%2356, %dims2357) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %2358 = "tf.Maximum"(%2348, %2357) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %2359 = "tf.Pow"(%2338, %2358) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %2360 = "tf.GreaterEqual"(%2319, %2359) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi1>
  %2361 = "tf.Squeeze"(%2360) { squeeze_dims = [ 0 : i64, 3 : i64 ], device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>) -> tensor<?x?xi1>
  %dims2362 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2362 = "tf.Prod"(%V__37, %dims2362) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>, tensor<2xi32>) -> tensor<f32>
  %2363 = "tf.Const"() { value = dense<[1, 91, 1, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2364 = "tf.BroadcastTo"(%2362, %2363) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims2365 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2365 = "tf.Mean"(%2364, %dims2365) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %dims2366 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2366 = "tf.Min"(%V__11, %dims2366) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims2367 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2367 = "tf.Mean"(%2366, %dims2367) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %dims2368 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2368 = "tf.Sum"(%2367, %dims2368) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %2369 = "tf.Minimum"(%2365, %2368) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims2370 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2370 = "tf.Sum"(%2369, %dims2370) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %dims2371 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2371 = "tf.Min"(%2370, %dims2371) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims2372 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2372 = "tf.Min"(%2371, %dims2372) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %dims2373 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2373 = "tf.Max"(%2372, %dims2373) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %dims2374 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2374 = "tf.Mean"(%V__69, %dims2374) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims2375 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2375 = "tf.Min"(%2374, %dims2375) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims2376 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2376 = "tf.Sum"(%2375, %dims2376) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims2377 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2377 = "tf.Max"(%2376, %dims2377) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims2378 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2378 = "tf.Sum"(%V__45, %dims2378) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %2379 = "tf.Softmax"(%2378) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims2380 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2380 = "tf.Min"(%2379, %dims2380) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %2381 = "tf.Add"(%2377, %2380) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims2382 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2382 = "tf.Min"(%2381, %dims2382) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %2383 = "tf.Abs"(%2382) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %2384 = "tf.Sigmoid"(%2383) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims2385 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2385 = "tf.Sum"(%2384, %dims2385) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %2386 = "tf.Greater"(%2373, %2385) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xi1>
  %dims2387 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2387 = "tf.Any"(%2386, %dims2387) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<2xi32>) -> tensor<?x?x?x?xi1>
  %2388 = "tf.Cast"(%2387) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>) -> tensor<?x?x?x?xi1>
  %dims2389 = "tf.Const"() { value = dense<[1, 0, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2389 = "tf.Transpose"(%V__54, %dims2389) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi1>, tensor<3xi32>) -> tensor<?x?x?xi1>
  %2390 = "tf.Const"() { value = dense<[1, 60, 1]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2391 = "tf.BroadcastTo"(%2389, %2390) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi1>, tensor<3xi32>) -> tensor<?x?x?xi1>
  %dims2392 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2392 = "tf.All"(%2391, %dims2392) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi1>, tensor<2xi32>) -> tensor<?x?x?xi1>
  %2393 = "tf.Const"() { value = dense<[1, 1, 1]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2394 = "tf.BroadcastTo"(%V__37, %2393) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  %2395 = "tf.Sigmoid"(%2394) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %2396 = "tf.Sigmoid"(%2395) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %2397 = "tf.MulNoNan"(%V__8, %V__71) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %dims2398 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2398 = "tf.Sum"(%2397, %dims2398) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?xf32>
  %2399 = "tf.SelectV2"(%2392, %2396, %2398) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi1>, tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %2400 = "tf.Cast"(%V__44) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i1>) -> tensor<i32>
  %dims2401 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2401 = "tf.Prod"(%V__33, %dims2401) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi32>, tensor<1xi32>) -> tensor<i32>
  %2402 = "tf.BitwiseAnd"(%2400, %2401) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %dims2403 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2403 = "tf.Sum"(%V__23, %dims2403) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims2404 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2404 = "tf.Mean"(%2403, %dims2404) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %dims2405 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2405 = "tf.Min"(%2404, %dims2405) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?x?xi64>
  %2406 = "tf.Shape"(%2405) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<4xi32>
  %2407 = "tf.BroadcastTo"(%2402, %2406) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims2408 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2408 = "tf.Sum"(%2407, %dims2408) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims2409 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2409 = "tf.Sum"(%2408, %dims2409) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %2410 = "tf.OnesLike"(%2409) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %2411 = "tf.Shape"(%2410) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<4xi32>
  %2412 = "tf.BroadcastTo"(%2399, %2411) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %2413 = "tf.Const"() { value = dense<[22, 1, 1, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2414 = "tf.BroadcastTo"(%V__4, %2413) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<4xi32>) -> tensor<?x?x?x?xi1>
  %dims2415 = "tf.Const"() { value = dense<[0, 3, 2, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2415 = "tf.Transpose"(%2414, %dims2415) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<4xi32>) -> tensor<?x?x?x?xi1>
  %dims2416 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2416 = "tf.Any"(%2415, %dims2416) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<4xi32>) -> tensor<i1>
  %2417 = "tf.Acosh"(%V__40) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims2418 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2418 = "tf.Max"(%2417, %dims2418) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %dims2419 = "tf.Const"() { value = dense<[0, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2419 = "tf.Max"(%2418, %dims2419) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %dims2420 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2420 = "tf.Prod"(%V__42, %dims2420) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %2421 = "tf.Reciprocal"(%2420) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims2422 = "tf.Const"() { value = dense<[3, 2, 0, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2422 = "tf.Transpose"(%2421, %dims2422) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %2423 = "tf.Select"(%2416, %2419, %2422) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i1>, tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %2424 = "tf.Selu"(%2423) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %2425 = "tf.Atan"(%V__51) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims2426 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2426 = "tf.Mean"(%2425, %dims2426) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %2427 = "tf.Exp"(%2426) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims2428 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2428 = "tf.Sum"(%2427, %dims2428) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %2429 = "tf.Relu6"(%V__40) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %2430 = "tf.Rsqrt"(%2429) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %2431 = "tf.Reciprocal"(%2430) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %2432 = "tf.Erf"(%V__72) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %2433 = "tf.Tan"(%2432) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims2434 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2434 = "tf.Min"(%2433, %dims2434) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %2435 = "tf.ClipByValue"(%2428, %2431, %2434) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims2436 = "tf.Const"() { value = dense<[2, 0, 3, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2436 = "tf.Transpose"(%2435, %dims2436) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %2437 = "tf.RealDiv"(%2424, %2436) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %2438 = "tf.Select"(%2388, %2412, %2437) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims2439 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2439 = "tf.Min"(%2438, %dims2439) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %2440 = "tf.Atan"(%2439) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims2441 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2441 = "tf.Min"(%2440, %dims2441) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %2442 = "tf.Erfc"(%2441) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %2443 = "tf.Sinh"(%2442) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims2444 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2444 = "tf.Sum"(%2443, %dims2444) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %2445 = "tf.Tanh"(%2444) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %2446 = "tf.Shape"(%2445) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<4xi32>
  %2448 = "tf.LogicalOr"(%2274, %2361) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<?x?xi1>) -> tensor<?x?x?x?xi1>
  %2449 = "tf.ZerosLike"(%2448) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>) -> tensor<?x?x?x?xi1>
  %2450 = "tf.Cast"(%2449) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>) -> tensor<?x?x?x?xi64>
  %2451 = "tf.Round"(%2450) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims2452 = "tf.Const"() { value = dense<[1, 2, 0, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2452 = "tf.Transpose"(%V__4, %dims2452) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<4xi32>) -> tensor<?x?x?x?xi1>
  %dims2453 = "tf.Const"() { value = dense<[3, 0, 2, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2453 = "tf.Transpose"(%2452, %dims2453) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<4xi32>) -> tensor<?x?x?x?xi1>
  %dims2454 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2454 = "tf.All"(%2453, %dims2454) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<1xi32>) -> tensor<?x?x?x?xi1>
  %dims2455 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2455 = "tf.Any"(%2454, %dims2455) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<1xi32>) -> tensor<?x?x?x?xi1>
  %2456 = "tf.Squeeze"(%2455) { squeeze_dims = [ 3 : i64 ], device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>) -> tensor<?x?x?xi1>
  %2457 = "tf.Cast"(%V__19) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi1>) -> tensor<?xi1>
  %2458 = "tf.Select"(%2457, %V__73, %V__0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi1>, tensor<?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  %dims2459 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2459 = "tf.Min"(%V__16, %dims2459) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>, tensor<2xi32>) -> tensor<?x?xi32>
  %2460 = "tf.Shape"(%V__73) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>) -> tensor<3xi32>
  %2461 = "tf.BroadcastTo"(%2459, %2460) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>, tensor<3xi32>) -> tensor<?x?x?xi32>
  %2462 = "tf.Select"(%2456, %2458, %2461) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi1>, tensor<?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  %2463 = "tf.Const"() { value = dense<[1, 1, 1, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2464 = "tf.Reshape"(%V__17, %2463) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims2465 = "tf.Const"() { value = dense<[1, 0, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2465 = "tf.Transpose"(%V__60, %dims2465) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %2466 = "tf.OnesLike"(%2465) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %2467 = "tf.Add"(%2464, %2466) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims2468 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2468 = "tf.Mean"(%2467, %dims2468) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?xi32>
  %dims2469 = "tf.Const"() { value = dense<[3, 1, 0, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2469 = "tf.Transpose"(%V__64, %dims2469) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims2470 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2470 = "tf.Sum"(%V__60, %dims2470) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %2471 = "tf.RightShift"(%2469, %2470) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims2472 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2472 = "tf.Min"(%2471, %dims2472) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?xi32>
  %dims2473 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2473 = "tf.Sum"(%2472, %dims2473) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?xi32>
  %2474 = "tf.LeftShift"(%2468, %2473) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  %2475 = "tf.Cast"(%V__25) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi64>) -> tensor<?x?xi32>
  %dims2476 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2476 = "tf.Min"(%2475, %dims2476) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>, tensor<1xi32>) -> tensor<?x?xi32>
  %2477 = "tf.Relu6"(%2476) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>) -> tensor<?x?xi32>
  %dims2478 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2478 = "tf.Mean"(%2477, %dims2478) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>, tensor<2xi32>) -> tensor<i32>
  %2479 = "tf.Const"() { value = dense<[1, 1, 1, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2480 = "tf.Reshape"(%2478, %2479) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims2481 = "tf.Const"() { value = dense<[3, 0, 1, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2481 = "tf.Transpose"(%2480, %dims2481) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims2482 = "tf.Const"() { value = dense<[2, 0, 3, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2482 = "tf.Transpose"(%2481, %dims2482) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims2483 = "tf.Const"() { value = dense<[3, 1, 2, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2483 = "tf.Transpose"(%2482, %dims2483) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %2484 = "tf.Abs"(%2483) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims2485 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2485 = "tf.Max"(%2484, %dims2485) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims2486 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2486 = "tf.Prod"(%2485, %dims2486) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?xi32>
  %2487 = "tf.ClipByValue"(%2462, %2474, %2486) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  %2488 = "tf.Add"(%V__20, %V__32) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims2489 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2489 = "tf.Max"(%V__18, %dims2489) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %2490 = "tf.FloorDiv"(%2488, %2489) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims2491 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2491 = "tf.Mean"(%2490, %dims2491) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims2492 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2492 = "tf.Max"(%2491, %dims2492) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims2493 = "tf.Const"() { value = dense<[3, 0, 2, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2493 = "tf.Transpose"(%V__18, %dims2493) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims2494 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2494 = "tf.Max"(%2493, %dims2494) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims2495 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2495 = "tf.Sum"(%2494, %dims2495) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims2496 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2496 = "tf.Prod"(%2495, %dims2496) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims2497 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2497 = "tf.Mean"(%2496, %dims2497) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims2498 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2498 = "tf.Mean"(%2497, %dims2498) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims2499 = "tf.Const"() { value = dense<[3, 1, 0, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2499 = "tf.Transpose"(%V__18, %dims2499) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %2500 = "tf.Sign"(%2499) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims2501 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2501 = "tf.Sum"(%2500, %dims2501) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims2502 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2502 = "tf.Max"(%2501, %dims2502) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims2503 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2503 = "tf.Min"(%2502, %dims2503) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims2504 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2504 = "tf.Min"(%2503, %dims2504) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %2505 = "tf.ClipByValue"(%2492, %2498, %2504) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %2506 = "tf.Const"() { value = dense<[68, 1, 1, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2507 = "tf.BroadcastTo"(%V__73, %2506) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims2508 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2508 = "tf.Sum"(%2507, %dims2508) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims2509 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2509 = "tf.Max"(%2508, %dims2509) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims2510 = "tf.Const"() { value = dense<[2, 3, 1, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2510 = "tf.Transpose"(%2509, %dims2510) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %2511 = "tf.Invert"(%2510) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims2512 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2512 = "tf.Prod"(%V__65, %dims2512) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims2513 = "tf.Const"() { value = dense<[1, 0, 3, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2513 = "tf.Transpose"(%V__32, %dims2513) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %2514 = "tf.Cast"(%V__27) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xi32>
  %2515 = "tf.ClipByValue"(%2512, %2513, %2514) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %2516 = "tf.TruncateDiv"(%2511, %2515) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims2517 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2517 = "tf.Mean"(%2516, %dims2517) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims2518 = "tf.Const"() { value = dense<[1, 0, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2518 = "tf.Transpose"(%2517, %dims2518) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims2519 = "tf.Const"() { value = dense<[0, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2519 = "tf.Max"(%2518, %dims2519) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims2520 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2520 = "tf.Prod"(%2519, %dims2520) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims2521 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2521 = "tf.Min"(%2520, %dims2521) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims2522 = "tf.Const"() { value = dense<[0, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2522 = "tf.Max"(%2521, %dims2522) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims2523 = "tf.Const"() { value = dense<[0, 3, 2, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2523 = "tf.Transpose"(%2522, %dims2523) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %2524 = "tf.FloorMod"(%2505, %2523) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %2525 = "tf.Cast"(%2524) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims2526 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2526 = "tf.Max"(%2525, %dims2526) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims2527 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2527 = "tf.Prod"(%2526, %dims2527) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?xi32>
  %2528 = "tf.Less"(%2487, %2527) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi1>
  %2529 = "tf.Shape"(%2528) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi1>) -> tensor<3xi32>
  %dims2530 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2530 = "tf.Max"(%V__3, %dims2530) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %2531 = "tf.Const"() { value = dense<[1, 1, 1, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2532 = "tf.BroadcastTo"(%2530, %2531) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims2533 = "tf.Const"() { value = dense<[0, 3, 1, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2533 = "tf.Transpose"(%2532, %dims2533) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims2534 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2534 = "tf.Min"(%2533, %dims2534) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?xi64>
  %2535 = "tf.BitwiseXor"(%V__3, %V__62) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims2536 = "tf.Const"() { value = dense<[0, 3, 2, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2536 = "tf.Transpose"(%2535, %dims2536) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims2537 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2537 = "tf.Prod"(%2536, %dims2537) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?xi64>
  %2538 = "tf.Sub"(%2534, %2537) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi64>, tensor<?x?xi64>) -> tensor<?x?xi64>
  %2539 = "tf.Const"() { value = dense<[1, 1, 1, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %dims2541 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2541 = "tf.Sum"(%V__62, %dims2541) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?x?xi64>
  %dims2542 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2542 = "tf.Prod"(%2541, %dims2542) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %dims2543 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2543 = "tf.Prod"(%2542, %dims2543) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims2544 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2544 = "tf.Sum"(%2543, %dims2544) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?x?xi64>
  %dims2545 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2545 = "tf.Prod"(%V__55, %dims2545) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %2546 = "tf.BiasAdd"(%2545, %V__5) { data_format = "NHWC", device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?xi64>) -> tensor<?x?x?x?xi64>
  %2547 = "tf.Less"(%2544, %2546) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi1>
  %2548 = "tf.Cast"(%2547) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>) -> tensor<?x?x?x?xi32>
  %2549 = "tf.Squeeze"(%V__19) { squeeze_dims = [ 0 : i64 ], device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi1>) -> tensor<i1>
  %2550 = "tf.LogicalAnd"(%2549, %V__44) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i1>, tensor<i1>) -> tensor<i1>
  %dims2551 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2551 = "tf.Mean"(%V__14, %dims2551) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims2552 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2552 = "tf.Min"(%2551, %dims2552) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims2553 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2553 = "tf.Mean"(%V__65, %dims2553) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims2554 = "tf.Const"() { value = dense<[3, 0, 1, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2554 = "tf.Transpose"(%2553, %dims2554) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %2555 = "tf.Select"(%2550, %2552, %2554) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i1>, tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %2556 = "tf.Mul"(%2548, %2555) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %2557 = "tf.Shape"(%2556) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<4xi32>
  %2558 = "tf.BroadcastTo"(%2538, %2557) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims2559 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2559 = "tf.Prod"(%2558, %dims2559) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims2560 = "tf.Const"() { value = dense<[1, 0, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2560 = "tf.Transpose"(%V__4, %dims2560) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<4xi32>) -> tensor<?x?x?x?xi1>
  %2561 = "tf.LogicalAnd"(%2560, %V__58) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<?x?x?x?xi1>) -> tensor<?x?x?x?xi1>
  %dims2562 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2562 = "tf.Any"(%2561, %dims2562) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<2xi32>) -> tensor<?x?x?x?xi1>
  %2563 = "tf.Sqrt"(%V__74) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>) -> tensor<f32>
  %2564 = "tf.Round"(%2563) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>) -> tensor<f32>
  %2565 = "tf.Shape"(%2564) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>) -> tensor<0xi32>
  %2566 = "tf.Reshape"(%2562, %2565) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<0xi32>) -> tensor<i1>
  %2567 = "tf.Cast"(%2566) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i1>) -> tensor<i32>
  %2568 = "tf.Abs"(%2567) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i32>) -> tensor<i32>
  %dims2569 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2569 = "tf.Max"(%V__73, %dims2569) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?xi32>
  %dims2570 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2570 = "tf.Prod"(%V__2, %dims2570) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?xi32>
  %dims2571 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2571 = "tf.Min"(%V__9, %dims2571) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?xi32>
  %2572 = "tf.ClipByValue"(%2569, %2570, %2571) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  %dims2573 = "tf.Const"() { value = dense<[1, 0, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2573 = "tf.Transpose"(%2572, %dims2573) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?xi32>
  %2574 = "tf.Invert"(%2573) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  %dims2575 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2575 = "tf.Mean"(%2574, %dims2575) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<3xi32>) -> tensor<i32>
  %2576 = "tf.Invert"(%2575) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i32>) -> tensor<i32>
  %2577 = "tf.Pow"(%2568, %2576) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %dims2578 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2578 = "tf.Prod"(%V__75, %dims2578) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32>
  %dims2579 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2579 = "tf.Mean"(%2578, %dims2579) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32>
  %dims2580 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2580 = "tf.Max"(%2579, %dims2580) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi32>, tensor<1xi32>) -> tensor<i32>
  %2581 = "tf.Cast"(%2580) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i32>) -> tensor<i32>
  %2582 = "tf.Maximum"(%V__66, %V__76) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %2583 = "tf.Cast"(%2582) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i32>) -> tensor<i32>
  %2584 = "tf.RightShift"(%2581, %2583) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %2585 = "tf.Abs"(%2584) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i32>) -> tensor<i32>
  %dims2586 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2586 = "tf.Prod"(%V__20, %dims2586) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?xi32>
  %2587 = "tf.TruncateDiv"(%2586, %V__77) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
  %2588 = "tf.Cast"(%V__54) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi1>) -> tensor<?x?x?xi32>
  %2589 = "tf.Cast"(%2588) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  %2590 = "tf.Squeeze"(%2589) { squeeze_dims = [ 0 : i64, 1 : i64 ], device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>) -> tensor<?xi32>
  %2591 = "tf.RightShift"(%2587, %2590) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
  %2592 = "tf.Squeeze"(%2591) { squeeze_dims = [ 0 : i64 ], device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi32>) -> tensor<i32>
  %2593 = "tf.Abs"(%2592) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i32>) -> tensor<i32>
  %2594 = "tf.BitwiseAnd"(%2585, %2593) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %2595 = "tf.ZerosLike"(%2594) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i32>) -> tensor<i32>
  %2596 = "tf.FloorMod"(%2577, %2595) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %2597 = "tf.Shape"(%2596) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i32>) -> tensor<0xi32>
  %2598 = "tf.Reshape"(%2559, %2597) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<0xi32>) -> tensor<i64>
  %2599 = "tf.Fill"(%2529, %2598) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<3xi32>, tensor<i64>) -> tensor<?x?x?xi64>
  %dims2600 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2600 = "tf.Min"(%2599, %dims2600) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?xi64>
  %dims2601 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2601 = "tf.Max"(%2600, %dims2601) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?xi64>
  %2602 = "tf.Invert"(%2601) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>) -> tensor<?x?x?xi64>
  %2603 = "tf.Round"(%V__21) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>) -> tensor<f32>
  %2604 = "tf.ZerosLike"(%V__21) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>) -> tensor<f32>
  %2605 = "tf.ApproximateEqual"(%2603, %2604) { tolerance = 1.000000e-06 : f32, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %2606 = "tf.Relu6"(%V__25) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi64>) -> tensor<?x?xi64>
  %2607 = "tf.Const"() { value = dense<[1, 1, 1, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2608 = "tf.BroadcastTo"(%2606, %2607) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %2609 = "tf.BitwiseOr"(%V__78, %V__62) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims2610 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2610 = "tf.Sum"(%2609, %dims2610) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %2611 = "tf.Select"(%2605, %2608, %2610) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i1>, tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims2612 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2612 = "tf.Sum"(%V__40, %dims2612) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  %2613 = "tf.Div"(%2612, %V__37) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %2614 = "tf.Selu"(%2613) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %2615 = "tf.Digamma"(%2614) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %dims2616 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2616 = "tf.Max"(%V__3, %dims2616) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims2617 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2617 = "tf.Prod"(%2616, %dims2617) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?x?xi64>
  %dims2618 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2618 = "tf.Min"(%2617, %dims2618) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?x?xi64>
  %dims2619 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2619 = "tf.Sum"(%2618, %dims2619) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?x?xi64>
  %2620 = "tf.Shape"(%2619) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<4xi32>
  %2621 = "tf.BroadcastTo"(%2615, %2620) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %2622 = "tf.Shape"(%2621) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<4xi32>
  %2623 = "tf.BroadcastTo"(%2611, %2622) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims2624 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2624 = "tf.Min"(%2623, %dims2624) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?x?xi64>
  %dims2625 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2625 = "tf.Mean"(%2624, %dims2625) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims2626 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2626 = "tf.Max"(%2625, %dims2626) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?x?xi64>
  %dims2627 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2627 = "tf.Sum"(%2626, %dims2627) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims2628 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2628 = "tf.Any"(%V__13, %dims2628) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<4xi32>) -> tensor<i1>
  %2629 = "tf.Select"(%2628, %V__4, %V__79) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i1>, tensor<?x?x?x?xi1>, tensor<?x?x?x?xi1>) -> tensor<?x?x?x?xi1>
  %dims2630 = "tf.Const"() { value = dense<[1, 3, 0, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2630 = "tf.Transpose"(%2629, %dims2630) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<4xi32>) -> tensor<?x?x?x?xi1>
  %dims2631 = "tf.Const"() { value = dense<[3, 1, 0, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2631 = "tf.Transpose"(%2630, %dims2631) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<4xi32>) -> tensor<?x?x?x?xi1>
  %dims2632 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2632 = "tf.Any"(%2631, %dims2632) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<1xi32>) -> tensor<?x?x?x?xi1>
  %dims2633 = "tf.Const"() { value = dense<[3, 1, 2, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2633 = "tf.Transpose"(%2632, %dims2633) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<4xi32>) -> tensor<?x?x?x?xi1>
  %dims2634 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2634 = "tf.All"(%2633, %dims2634) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<3xi32>) -> tensor<?xi1>
  %dims2635 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2635 = "tf.Sum"(%V__14, %dims2635) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims2636 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2636 = "tf.Sum"(%2635, %dims2636) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?xi32>
  %dims2637 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2637 = "tf.Sum"(%V__12, %dims2637) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %2638 = "tf.Shape"(%2637) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<4xi32>
  %2639 = "tf.BroadcastTo"(%2636, %2638) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims2640 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2640 = "tf.Sum"(%2639, %dims2640) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims2641 = "tf.Const"() { value = dense<[2, 3, 0, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2641 = "tf.Transpose"(%2640, %dims2641) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims2642 = "tf.Const"() { value = dense<[1, 0, 3, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2642 = "tf.Transpose"(%V__9, %dims2642) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims2643 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2643 = "tf.Sum"(%2642, %dims2643) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?xi32>
  %2644 = "tf.Shape"(%V__62) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<4xi32>
  %2645 = "tf.BroadcastTo"(%2643, %2644) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims2646 = "tf.Const"() { value = dense<[1, 3, 2, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2646 = "tf.Transpose"(%2645, %dims2646) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims2647 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2647 = "tf.Min"(%2646, %dims2647) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims2648 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2648 = "tf.Max"(%2647, %dims2648) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %2649 = "tf.Select"(%2634, %2641, %2648) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi1>, tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims2650 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2650 = "tf.Max"(%2649, %dims2650) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims2651 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2651 = "tf.Mean"(%2650, %dims2651) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims2652 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2652 = "tf.Min"(%2651, %dims2652) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %2653 = "tf.Shape"(%2652) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<4xi32>
  %2654 = "tf.BroadcastTo"(%2627, %2653) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims2655 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2655 = "tf.Max"(%V__3, %dims2655) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?x?xi64>
  %dims2656 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2656 = "tf.Mean"(%2655, %dims2656) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims2657 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2657 = "tf.Min"(%2656, %dims2657) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %2658 = "tf.SquaredDifference"(%V__57, %V__41) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %2659 = "tf.Shape"(%2658) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<4xi32>
  %2660 = "tf.BroadcastTo"(%2657, %2659) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims2661 = "tf.Const"() { value = dense<[0, 2, 3, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2661 = "tf.Transpose"(%2660, %dims2661) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %2662 = "tf.Maximum"(%V__61, %V__61) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %2663 = "tf.Const"() { value = dense<[1, 1, 53, 50]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2664 = "tf.BroadcastTo"(%2662, %2663) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims2665 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2665 = "tf.Mean"(%2664, %dims2665) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims2666 = "tf.Const"() { value = dense<[3, 1, 0, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2666 = "tf.Transpose"(%2665, %dims2666) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims2667 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2667 = "tf.Max"(%V__23, %dims2667) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?x?xi64>
  %2668 = "tf.Cast"(%2667) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %2669 = "tf.Neg"(%V__80) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims2670 = "tf.Const"() { value = dense<[2, 3, 1, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2670 = "tf.Transpose"(%2669, %dims2670) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %2671 = "tf.LeftShift"(%2668, %2670) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims2672 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2672 = "tf.Prod"(%2671, %dims2672) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %dims2673 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2673 = "tf.Mean"(%2672, %dims2673) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %2674 = "tf.ClipByValue"(%2661, %2666, %2673) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims2675 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2675 = "tf.Min"(%2674, %dims2675) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims2676 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2676 = "tf.Sum"(%2675, %dims2676) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims2677 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2677 = "tf.Min"(%2676, %dims2677) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims2678 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2678 = "tf.All"(%V__79, %dims2678) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<3xi32>) -> tensor<?xi1>
  %dims2679 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2679 = "tf.Max"(%V__78, %dims2679) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?xi64>
  %dims2680 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2680 = "tf.Max"(%V__6, %dims2680) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?xi64>
  %2681 = "tf.Select"(%2678, %2679, %2680) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi1>, tensor<?x?x?xi64>, tensor<?x?x?xi64>) -> tensor<?x?x?xi64>
  %dims2682 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2682 = "tf.Max"(%V__62, %dims2682) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?xi64>
  %dims2683 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2683 = "tf.Prod"(%2682, %dims2683) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi64>, tensor<1xi32>) -> tensor<i64>
  %dims2684 = "tf.Const"() { value = dense<[1, 2, 0]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2684 = "tf.Transpose"(%V__67, %dims2684) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?xi32>
  %2685 = "tf.Shape"(%2684) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>) -> tensor<3xi32>
  %2687 = "tf.Mul"(%2681, %2683) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<i64>) -> tensor<?x?x?xi64>
  %dims2688 = "tf.Const"() { value = dense<[0, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2688 = "tf.Max"(%V__36, %dims2688) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %dims2689 = "tf.Const"() { value = dense<[0, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2689 = "tf.Max"(%2688, %dims2689) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %2690 = "tf.Sign"(%2689) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %2691 = "tf.Cos"(%2690) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims2692 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2692 = "tf.Any"(%V__54, %dims2692) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi1>, tensor<2xi32>) -> tensor<?xi1>
  %2693 = "tf.Select"(%2692, %V__27, %V__40) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi1>, tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %2694 = "tf.Polygamma"(%2691, %2693) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims2695 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2695 = "tf.Mean"(%2694, %dims2695) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims2696 = "tf.Const"() { value = dense<[1, 0, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2696 = "tf.Transpose"(%2695, %dims2696) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %2697 = "tf.Sin"(%2696) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %2698 = "tf.Shape"(%2697) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<4xi32>
  %2700 = "tf.Mul"(%2677, %2687) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %2701 = "tf.Neg"(%2700) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims2702 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2702 = "tf.Mean"(%2701, %dims2702) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %2703 = "tf.ZerosLike"(%2702) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %2704 = "tf.OnesLike"(%V__13) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>) -> tensor<?x?x?x?xi1>
  %dims2705 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2705 = "tf.All"(%2704, %dims2705) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<2xi32>) -> tensor<?x?x?x?xi1>
  %dims2706 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2706 = "tf.Prod"(%V__62, %dims2706) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims2707 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2707 = "tf.Prod"(%V__55, %dims2707) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %2708 = "tf.Select"(%2705, %2706, %2707) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims2709 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2709 = "tf.Min"(%V__49, %dims2709) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?x?xi64>
  %2710 = "tf.Const"() { value = dense<[1, 1, 1, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2711 = "tf.BroadcastTo"(%2709, %2710) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims2712 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2712 = "tf.Sum"(%2711, %dims2712) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %2713 = "tf.OnesLike"(%2712) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims2714 = "tf.Const"() { value = dense<[2, 0, 1, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2714 = "tf.Transpose"(%2713, %dims2714) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %2715 = "tf.BitwiseXor"(%2708, %2714) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims2716 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2716 = "tf.Any"(%V__4, %dims2716) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<1xi32>) -> tensor<?x?x?x?xi1>
  %dims2717 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2717 = "tf.Sum"(%V__59, %dims2717) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %2718 = "tf.Select"(%2716, %2717, %V__81) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims2719 = "tf.Const"() { value = dense<[2, 3, 0, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2719 = "tf.Transpose"(%V__59, %dims2719) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims2720 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2720 = "tf.Min"(%V__3, %dims2720) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %2721 = "tf.Mul"(%2719, %2720) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %2722 = "tf.SquaredDifference"(%2718, %2721) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims2723 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2723 = "tf.Prod"(%2722, %dims2723) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %2724 = "tf.OnesLike"(%2723) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims2725 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2725 = "tf.Min"(%2724, %dims2725) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %2726 = "tf.BitwiseOr"(%2715, %2725) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims2727 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2727 = "tf.Sum"(%2726, %dims2727) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?xi64>
  %dims2728 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2728 = "tf.All"(%V__19, %dims2728) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi1>, tensor<1xi32>) -> tensor<i1>
  %dims2729 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2729 = "tf.Prod"(%V__24, %dims2729) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi32>, tensor<1xi32>) -> tensor<i32>
  %dims2730 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2730 = "tf.Prod"(%V__67, %dims2730) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<3xi32>) -> tensor<i32>
  %2731 = "tf.SelectV2"(%2728, %2729, %2730) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
  %2732 = "tf.Const"() { value = dense<[1, 1, 33, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2733 = "tf.BroadcastTo"(%V__32, %2732) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims2734 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2734 = "tf.Mean"(%2733, %dims2734) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims2735 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2735 = "tf.Mean"(%2734, %dims2735) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<i32>
  %2736 = "tf.Invert"(%2735) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i32>) -> tensor<i32>
  %2737 = "tf.Div"(%2731, %2736) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %dims2738 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2738 = "tf.Mean"(%V__64, %dims2738) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %2739 = "tf.ClipByValue"(%2738, %V__2, %V__60) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims2740 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2740 = "tf.Mean"(%V__2, %dims2740) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %2741 = "tf.AddV2"(%2740, %V__32) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims2742 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2742 = "tf.Prod"(%2741, %dims2742) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %2743 = "tf.BitwiseOr"(%2739, %2742) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %2744 = "tf.OnesLike"(%2743) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %2745 = "tf.Shape"(%2744) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<4xi32>
  %2746 = "tf.Reshape"(%2737, %2745) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims2747 = "tf.Const"() { value = dense<[2, 1, 3, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2747 = "tf.Transpose"(%2746, %dims2747) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims2748 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2748 = "tf.Sum"(%2747, %dims2748) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims2749 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2749 = "tf.Prod"(%2748, %dims2749) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %2750 = "tf.Shape"(%2749) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<4xi32>
  %2751 = "tf.BroadcastTo"(%2727, %2750) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %2752 = "tf.ClipByValue"(%2654, %2703, %2751) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims2753 = "tf.Const"() { value = dense<[0, 2, 3, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2753 = "tf.Transpose"(%2752, %dims2753) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims2754 = "tf.Const"() { value = dense<[1, 2, 3, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2754 = "tf.Transpose"(%2753, %dims2754) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims2755 = "tf.Const"() { value = dense<[3, 0, 2, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2755 = "tf.Transpose"(%2754, %dims2755) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims2756 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2756 = "tf.Sum"(%2755, %dims2756) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %dims2757 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2757 = "tf.Min"(%2756, %dims2757) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?xi64>
  %2758 = "tf.Minimum"(%2602, %2757) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<?x?x?xi64>) -> tensor<?x?x?xi64>
  %dims2759 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2759 = "tf.Mean"(%2758, %dims2759) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?xi64>
  %dims2760 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2760 = "tf.Prod"(%2759, %dims2760) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<2xi32>) -> tensor<?xi64>
  %2761 = "tf.BiasAdd"(%2451, %2760) { data_format = "NHWC", device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?xi64>) -> tensor<?x?x?x?xi64>
  %dims2762 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2762 = "tf.Any"(%V__13, %dims2762) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<3xi32>) -> tensor<?xi1>
  %dims2763 = "tf.Const"() { value = dense<[1, 2, 0]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2763 = "tf.Transpose"(%V__6, %dims2763) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?xi64>
  %2764 = "tf.Select"(%2762, %2763, %V__82) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi1>, tensor<?x?x?xi64>, tensor<?x?x?xi64>) -> tensor<?x?x?xi64>
  %dims2765 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2765 = "tf.Max"(%2764, %dims2765) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?xi64>
  %dims2766 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2766 = "tf.Prod"(%2765, %dims2766) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?xi64>
  %2767 = "tf.FloorMod"(%V__31, %V__31) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %dims2768 = "tf.Const"() { value = dense<[2, 1, 0]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2768 = "tf.Transpose"(%V__8, %dims2768) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  %2769 = "tf.Shape"(%2768) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<3xi32>
  %2770 = "tf.BroadcastTo"(%2767, %2769) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  %2771 = "tf.Shape"(%2770) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<3xi32>
  %dims2773 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2773 = "tf.Prod"(%V__6, %dims2773) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?xi64>
  %dims2774 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2774 = "tf.Prod"(%V__81, %dims2774) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?xi64>
  %2775 = "tf.BitwiseOr"(%2773, %2774) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<?x?x?xi64>) -> tensor<?x?x?xi64>
  %dims2776 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2776 = "tf.Sum"(%V__83, %dims2776) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?xi64>
  %dims2777 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2777 = "tf.Sum"(%2776, %dims2777) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?xi64>
  %dims2778 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2778 = "tf.Min"(%2777, %dims2778) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?xi64>
  %dims2779 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2779 = "tf.Min"(%V__6, %dims2779) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?xi64>
  %2780 = "tf.AddV2"(%2779, %V__6) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<?x?x?xi64>) -> tensor<?x?x?xi64>
  %2781 = "tf.ClipByValue"(%2775, %2778, %2780) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<?x?x?xi64>, tensor<?x?x?xi64>) -> tensor<?x?x?xi64>
  %dims2782 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2782 = "tf.Max"(%2781, %dims2782) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?xi64>
  %dims2783 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2783 = "tf.Max"(%2782, %dims2783) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?xi64>
  %2784 = "tf.LeftShift"(%2766, %2783) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<?x?x?xi64>) -> tensor<?x?x?xi64>
  %dims2785 = "tf.Const"() { value = dense<[0, 2, 1, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2785 = "tf.Transpose"(%V__49, %dims2785) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims2786 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2786 = "tf.Sum"(%V__84, %dims2786) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %2787 = "tf.LeftShift"(%2785, %2786) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims2788 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2788 = "tf.Min"(%2787, %dims2788) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %2789 = "tf.Relu"(%2788) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims2790 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2790 = "tf.Min"(%2789, %dims2790) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?xi64>
  %dims2791 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2791 = "tf.Sum"(%V__82, %dims2791) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<1xi32>) -> tensor<?x?xi64>
  %2792 = "tf.Shape"(%V__2) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<4xi32>
  %2793 = "tf.BroadcastTo"(%2791, %2792) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims2794 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2794 = "tf.Min"(%2793, %dims2794) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims2795 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2795 = "tf.Mean"(%2794, %dims2795) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?xi64>
  %2796 = "tf.SquaredDifference"(%2790, %2795) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi64>, tensor<?x?xi64>) -> tensor<?x?xi64>
  %2797 = "tf.Const"() { value = dense<[36, 1, 1]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2798 = "tf.BroadcastTo"(%2796, %2797) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi64>, tensor<3xi32>) -> tensor<?x?x?xi64>
  %dims2799 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2799 = "tf.Min"(%2798, %dims2799) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?xi64>
  %dims2800 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2800 = "tf.Mean"(%V__83, %dims2800) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %dims2801 = "tf.Const"() { value = dense<[3, 2, 0, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2801 = "tf.Transpose"(%2800, %dims2801) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims2802 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2802 = "tf.Sum"(%2801, %dims2802) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %2803 = "tf.Neg"(%2802) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims2804 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2804 = "tf.Sum"(%V__55, %dims2804) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?x?xi64>
  %dims2805 = "tf.Const"() { value = dense<[0, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2805 = "tf.Mean"(%2804, %dims2805) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %dims2806 = "tf.Const"() { value = dense<[0, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2806 = "tf.Max"(%2805, %dims2806) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %2807 = "tf.Div"(%2803, %2806) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims2808 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2808 = "tf.Prod"(%2807, %dims2808) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?xi64>
  %dims2809 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2809 = "tf.Min"(%V__65, %dims2809) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims2810 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2810 = "tf.Mean"(%2809, %dims2810) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims2811 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2811 = "tf.Prod"(%2810, %dims2811) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims2812 = "tf.Const"() { value = dense<[1, 2, 0, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2812 = "tf.Transpose"(%V__65, %dims2812) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims2813 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2813 = "tf.Mean"(%2812, %dims2813) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims2814 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2814 = "tf.Mean"(%2813, %dims2814) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %2815 = "tf.Maximum"(%2811, %2814) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %2816 = "tf.Shape"(%2815) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<4xi32>
  %2817 = "tf.BroadcastTo"(%2808, %2816) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims2818 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2818 = "tf.Prod"(%2817, %dims2818) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?x?xi64>
  %2819 = "tf.OnesLike"(%2818) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims2820 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2820 = "tf.Prod"(%2819, %dims2820) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims2821 = "tf.Const"() { value = dense<[3, 1, 2, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2821 = "tf.Transpose"(%2820, %dims2821) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims2822 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2822 = "tf.Prod"(%2821, %dims2822) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims2823 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2823 = "tf.Sum"(%2822, %dims2823) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?x?xi64>
  %dims2824 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2824 = "tf.Prod"(%2823, %dims2824) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims2825 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2825 = "tf.Min"(%2824, %dims2825) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?x?xi64>
  %dims2826 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2826 = "tf.Mean"(%2825, %dims2826) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims2827 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2827 = "tf.Max"(%2826, %dims2827) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?x?xi64>
  %dims2828 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2828 = "tf.Min"(%2827, %dims2828) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?xi64>
  %2829 = "tf.ClipByValue"(%2784, %2799, %2828) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<?x?x?xi64>, tensor<?x?x?xi64>) -> tensor<?x?x?xi64>
  %2830 = "tf.Const"() { value = dense<[75, 1, 1, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %dims2832 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2832 = "tf.All"(%V__58, %dims2832) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<1xi32>) -> tensor<?x?x?x?xi1>
  %dims2833 = "tf.Const"() { value = dense<[0, 3, 2, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2833 = "tf.Transpose"(%2832, %dims2833) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<4xi32>) -> tensor<?x?x?x?xi1>
  %dims2834 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2834 = "tf.Min"(%V__78, %dims2834) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims2835 = "tf.Const"() { value = dense<[3, 2, 1, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2835 = "tf.Transpose"(%V__62, %dims2835) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %2836 = "tf.Select"(%2833, %2834, %2835) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims2837 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2837 = "tf.Mean"(%2836, %dims2837) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %dims2838 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2838 = "tf.All"(%V__4, %dims2838) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<2xi32>) -> tensor<?x?x?x?xi1>
  %2839 = "tf.Select"(%2838, %V__83, %V__84) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims2840 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2840 = "tf.Mean"(%2839, %dims2840) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims2841 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2841 = "tf.Min"(%2840, %dims2841) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims2842 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2842 = "tf.Sum"(%2841, %dims2842) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %2843 = "tf.Pow"(%2837, %2842) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims2844 = "tf.Const"() { value = dense<[0, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2844 = "tf.Sum"(%2843, %dims2844) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %2845 = "tf.Round"(%2844) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims2846 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2846 = "tf.Prod"(%2845, %dims2846) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims2847 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2847 = "tf.Min"(%2846, %dims2847) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %dims2848 = "tf.Const"() { value = dense<[0, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2848 = "tf.Prod"(%2847, %dims2848) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %dims2849 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2849 = "tf.Mean"(%2848, %dims2849) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %2850 = "tf.NotEqual"(%V__85, %V__23) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi1>
  %dims2851 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2851 = "tf.All"(%V__79, %dims2851) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<3xi32>) -> tensor<?x?x?x?xi1>
  %dims2852 = "tf.Const"() { value = dense<[2, 3, 1, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2852 = "tf.Transpose"(%V__79, %dims2852) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<4xi32>) -> tensor<?x?x?x?xi1>
  %2853 = "tf.Select"(%2850, %2851, %2852) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<?x?x?x?xi1>, tensor<?x?x?x?xi1>) -> tensor<?x?x?x?xi1>
  %dims2854 = "tf.Const"() { value = dense<[3, 2, 1, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2854 = "tf.Transpose"(%2853, %dims2854) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<4xi32>) -> tensor<?x?x?x?xi1>
  %dims2855 = "tf.Const"() { value = dense<[1, 0, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2855 = "tf.Transpose"(%2854, %dims2855) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<4xi32>) -> tensor<?x?x?x?xi1>
  %2856 = "tf.Squeeze"(%V__86) { squeeze_dims = [ 1 : i64, 2 : i64 ], device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi1>) -> tensor<?xi1>
  %dims2857 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2857 = "tf.Sum"(%V__64, %dims2857) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims2858 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2858 = "tf.Min"(%V__32, %dims2858) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %2859 = "tf.Select"(%2856, %2857, %2858) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi1>, tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims2860 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2860 = "tf.Sum"(%2859, %dims2860) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims2861 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2861 = "tf.Max"(%2860, %dims2861) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %2862 = "tf.Shape"(%2861) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<4xi32>
  %2863 = "tf.BroadcastTo"(%2855, %2862) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<4xi32>) -> tensor<?x?x?x?xi1>
  %dims2864 = "tf.Const"() { value = dense<[1, 2, 3, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2864 = "tf.Transpose"(%2863, %dims2864) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<4xi32>) -> tensor<?x?x?x?xi1>
  %2865 = "tf.Shape"(%2864) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>) -> tensor<4xi32>
  %2866 = "tf.BroadcastTo"(%2849, %2865) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims2867 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2867 = "tf.Prod"(%2866, %dims2867) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?xi64>
  %dims2868 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2868 = "tf.Sum"(%2867, %dims2868) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi64>, tensor<1xi32>) -> tensor<?x?xi64>
  %dims2869 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2869 = "tf.Mean"(%V__83, %dims2869) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?xi64>
  %2870 = "tf.RightShift"(%2869, %V__25) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi64>, tensor<?x?xi64>) -> tensor<?x?xi64>
  %dims2871 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2871 = "tf.Prod"(%V__2, %dims2871) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %2872 = "tf.Shape"(%2871) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<4xi32>
  %2873 = "tf.BroadcastTo"(%2870, %2872) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims2874 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2874 = "tf.Min"(%2873, %dims2874) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?x?xi64>
  %dims2875 = "tf.Const"() { value = dense<[2, 3, 1, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2875 = "tf.Transpose"(%2874, %dims2875) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims2876 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2876 = "tf.Min"(%2875, %dims2876) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %2877 = "tf.Sign"(%V__25) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi64>) -> tensor<?x?xi64>
  %2878 = "tf.Const"() { value = dense<[1, 1, 1, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %dims2880 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2880 = "tf.Mean"(%V__9, %dims2880) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims2881 = "tf.Const"() { value = dense<[0, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2881 = "tf.Sum"(%2880, %dims2881) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %2882 = "tf.Shape"(%2881) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<4xi32>
  %2883 = "tf.BroadcastTo"(%2877, %2882) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims2884 = "tf.Const"() { value = dense<[0, 1, 3, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2884 = "tf.Transpose"(%2883, %dims2884) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %2885 = "tf.BitwiseAnd"(%2876, %2884) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims2886 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2886 = "tf.Sum"(%2885, %dims2886) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %dims2887 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2887 = "tf.Min"(%V__78, %dims2887) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<i64>
  %2888 = "tf.Squeeze"(%V__25) { squeeze_dims = [ 0 : i64, 1 : i64 ], device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi64>) -> tensor<i64>
  %2889 = "tf.AddV2"(%2887, %2888) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %dims2890 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2890 = "tf.Prod"(%V__25, %dims2890) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi64>, tensor<1xi32>) -> tensor<?x?xi64>
  %dims2891 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2891 = "tf.Mean"(%2890, %dims2891) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi64>, tensor<1xi32>) -> tensor<?x?xi64>
  %dims2892 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2892 = "tf.Prod"(%2891, %dims2892) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi64>, tensor<2xi32>) -> tensor<i64>
  %2893 = "tf.Round"(%2892) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i64>) -> tensor<i64>
  %2894 = "tf.FloorDiv"(%2889, %2893) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %dims2895 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2895 = "tf.Min"(%V__87, %dims2895) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %dims2896 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2896 = "tf.Prod"(%2895, %dims2896) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %2897 = "tf.Shape"(%2896) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<4xi32>
  %dims2898 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2898 = "tf.Sum"(%V__5, %dims2898) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi64>, tensor<1xi32>) -> tensor<i64>
  %2899 = "tf.ZerosLike"(%2898) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i64>) -> tensor<i64>
  %2900 = "tf.Round"(%2899) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i64>) -> tensor<i64>
  %2901 = "tf.Fill"(%2897, %2900) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<4xi32>, tensor<i64>) -> tensor<?x?x?x?xi64>
  %2902 = "tf.Shape"(%2901) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<4xi32>
  %2903 = "tf.BroadcastTo"(%2894, %2902) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %2904 = "tf.Shape"(%2903) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<4xi32>
  %2905 = "tf.BroadcastTo"(%2886, %2904) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims2906 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2906 = "tf.Max"(%2905, %dims2906) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?x?xi64>
  %2907 = "tf.Shape"(%2906) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<4xi32>
  %2908 = "tf.BroadcastTo"(%2868, %2907) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %2909 = "tf.Squeeze"(%2908) { squeeze_dims = [ 2 : i64 ], device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<?x?x?xi64>
  %2910 = "tf.Const"() { value = dense<[75, 1, 1, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2911 = "tf.BroadcastTo"(%2909, %2910) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %2912 = "tf.Maximum"(%2829, %2911) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims2913 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2913 = "tf.Max"(%2912, %dims2913) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims2914 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2914 = "tf.Prod"(%2913, %dims2914) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims2915 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2915 = "tf.Min"(%2914, %dims2915) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?x?xi64>
  %dims2916 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2916 = "tf.Mean"(%2915, %dims2916) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims2917 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2917 = "tf.Any"(%V__4, %dims2917) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<4xi32>) -> tensor<?x?x?x?xi1>
  %dims2918 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2918 = "tf.All"(%2917, %dims2918) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<2xi32>) -> tensor<?x?x?x?xi1>
  %2919 = "tf.Cast"(%2918) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>) -> tensor<?x?x?x?xi64>
  %dims2920 = "tf.Const"() { value = dense<[2, 1, 0, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2920 = "tf.Transpose"(%2919, %dims2920) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims2921 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2921 = "tf.Prod"(%V__87, %dims2921) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %dims2922 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2922 = "tf.Sum"(%2921, %dims2922) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %dims2923 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2923 = "tf.Max"(%2922, %dims2923) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?x?xi64>
  %2924 = "tf.Add"(%2920, %2923) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims2925 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2925 = "tf.Mean"(%2924, %dims2925) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims2926 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2926 = "tf.Mean"(%2925, %dims2926) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %2927 = "tf.Invert"(%2926) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims2928 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2928 = "tf.Max"(%2927, %dims2928) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?x?xi64>
  %2929 = "tf.Invert"(%2928) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims2930 = "tf.Const"() { value = dense<[2, 0, 3, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2930 = "tf.Transpose"(%2929, %dims2930) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims2931 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2931 = "tf.Sum"(%2930, %dims2931) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims2932 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2932 = "tf.Min"(%2931, %dims2932) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?x?xi64>
  %2933 = "tf.IsNan"(%V__31) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>) -> tensor<?xi1>
  %dims2934 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2934 = "tf.Min"(%V__88, %dims2934) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?x?xi64>
  %2935 = "tf.Select"(%2933, %2934, %V__49) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi1>, tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %2936 = "tf.Const"() { value = dense<[26, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %dims2937 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2937 = "tf.Prod"(%V__87, %dims2937) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<i64>
  %2938 = "tf.Fill"(%2936, %2937) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<2xi32>, tensor<i64>) -> tensor<?x?xi64>
  %dims2939 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2939 = "tf.Max"(%2938, %dims2939) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi64>, tensor<1xi32>) -> tensor<?xi64>
  %2940 = "tf.BiasAdd"(%2935, %2939) { data_format = "NCHW", device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?xi64>) -> tensor<?x?x?x?xi64>
  %dims2941 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2941 = "tf.Sum"(%2940, %dims2941) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %dims2942 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2942 = "tf.Min"(%2941, %dims2942) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?x?xi64>
  %dims2943 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2943 = "tf.Prod"(%2942, %dims2943) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims2944 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2944 = "tf.Mean"(%2943, %dims2944) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims2945 = "tf.Const"() { value = dense<[1, 0, 3, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2945 = "tf.Transpose"(%2944, %dims2945) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %2946 = "tf.AddV2"(%2932, %2945) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims2947 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2947 = "tf.Max"(%2946, %dims2947) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims2948 = "tf.Const"() { value = dense<[0, 1, 3, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2948 = "tf.Transpose"(%2947, %dims2948) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims2949 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2949 = "tf.Mean"(%2948, %dims2949) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims2950 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2950 = "tf.Mean"(%2949, %dims2950) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %dims2951 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2951 = "tf.Prod"(%2950, %dims2951) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?x?xi64>
  %dims2952 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2952 = "tf.Mean"(%2951, %dims2952) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %dims2953 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2953 = "tf.Min"(%2952, %dims2953) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims2954 = "tf.Const"() { value = dense<[2, 1, 0, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2954 = "tf.Transpose"(%2953, %dims2954) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims2955 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2955 = "tf.Max"(%2954, %dims2955) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims2956 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2956 = "tf.Mean"(%2955, %dims2956) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %dims2957 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2957 = "tf.Min"(%2956, %dims2957) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?x?xi64>
  %dims2958 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2958 = "tf.Sum"(%V__78, %dims2958) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?xi64>
  %2959 = "tf.SquaredDifference"(%2958, %V__89) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<?x?x?xi64>) -> tensor<?x?x?xi64>
  %2960 = "tf.Squeeze"(%2959) { squeeze_dims = [ 2 : i64 ], device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>) -> tensor<?x?xi64>
  %dims2961 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2961 = "tf.Min"(%2960, %dims2961) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi64>, tensor<1xi32>) -> tensor<?xi64>
  %2962 = "tf.Relu6"(%2961) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi64>) -> tensor<?xi64>
  %2963 = "tf.Selu"(%V__46) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims2964 = "tf.Const"() { value = dense<[2, 3, 0, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2964 = "tf.Transpose"(%2963, %dims2964) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims2965 = "tf.Const"() { value = dense<[1, 3, 2, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2965 = "tf.Transpose"(%2964, %dims2965) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims2966 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2966 = "tf.Prod"(%2965, %dims2966) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %2967 = "tf.Shape"(%2966) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<4xi32>
  %2968 = "tf.BroadcastTo"(%2962, %2967) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims2969 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2969 = "tf.Sum"(%2968, %dims2969) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?xi64>
  %dims2970 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2970 = "tf.Mean"(%2969, %dims2970) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<2xi32>) -> tensor<?xi64>
  %dims2971 = "tf.Const"() { value = dense<[1, 0, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2971 = "tf.Transpose"(%V__90, %dims2971) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims2972 = "tf.Const"() { value = dense<[3, 0, 1, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2972 = "tf.Transpose"(%2971, %dims2972) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims2973 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2973 = "tf.Mean"(%2972, %dims2973) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims2974 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2974 = "tf.Min"(%2973, %dims2974) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims2975 = "tf.Const"() { value = dense<[0, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2975 = "tf.Prod"(%2974, %dims2975) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %dims2976 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2976 = "tf.Sum"(%V__62, %dims2976) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %2977 = "tf.Const"() { value = dense<[1, 1, 1, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2979 = "tf.FloorDiv"(%2975, %2976) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims2980 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2980 = "tf.Max"(%2979, %dims2980) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %dims2981 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2981 = "tf.Prod"(%2980, %dims2981) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?xi64>
  %dims2982 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2982 = "tf.Prod"(%2981, %dims2982) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<2xi32>) -> tensor<?xi64>
  %2983 = "tf.Maximum"(%V__5, %V__5) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi64>, tensor<?xi64>) -> tensor<?xi64>
  %dims2984 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2984 = "tf.Max"(%V__23, %dims2984) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?xi64>
  %dims2985 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2985 = "tf.Min"(%2984, %dims2985) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi64>, tensor<1xi32>) -> tensor<?xi64>
  %2986 = "tf.Maximum"(%2983, %2985) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi64>, tensor<?xi64>) -> tensor<?xi64>
  %2987 = "tf.Const"() { value = dense<[1, 1, 1]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %dims2989 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2989 = "tf.Prod"(%V__88, %dims2989) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?xi64>
  %2990 = "tf.Div"(%V__5, %2989) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi64>, tensor<?x?x?xi64>) -> tensor<?x?x?xi64>
  %2991 = "tf.Squeeze"(%2990) { squeeze_dims = [ 0 : i64, 2 : i64 ], device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>) -> tensor<?xi64>
  %2992 = "tf.BitwiseAnd"(%2986, %2991) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi64>, tensor<?xi64>) -> tensor<?xi64>
  %2993 = "tf.ClipByValue"(%2970, %2982, %2992) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi64>, tensor<?xi64>, tensor<?xi64>) -> tensor<?xi64>
  %2994 = "tf.BiasAdd"(%2957, %2993) { data_format = "NHWC", device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?xi64>) -> tensor<?x?x?x?xi64>
  %dims2995 = "tf.Const"() { value = dense<[3, 1, 0, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2995 = "tf.Transpose"(%2994, %dims2995) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims2996 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2996 = "tf.Sum"(%2995, %dims2996) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?x?xi64>
  %dims2997 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2997 = "tf.Max"(%2996, %dims2997) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims2998 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2998 = "tf.Mean"(%2997, %dims2998) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?x?xi64>
  %dims2999 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2999 = "tf.Min"(%2998, %dims2999) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %3000 = "tf.Const"() { value = dense<[73, 36, 1, 75]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3001 = "tf.Log"(%V__28) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %3002 = "tf.Floor"(%3001) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims3003 = "tf.Const"() { value = dense<[0, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3003 = "tf.Sum"(%3002, %dims3003) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %3004 = "tf.Atan"(%3003) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %3005 = "tf.Relu6"(%3004) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims3006 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3006 = "tf.Min"(%3005, %dims3006) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<f32>
  %3007 = "tf.Floor"(%3006) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>) -> tensor<f32>
  %3008 = "tf.Digamma"(%3007) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>) -> tensor<f32>
  %3009 = "tf.Fill"(%3000, %3008) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<4xi32>, tensor<f32>) -> tensor<?x?x?x?xf32>
  %dims3010 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3010 = "tf.Max"(%3009, %dims3010) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %3011 = "tf.Cast"(%3010) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xi64>
  %dims3012 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3012 = "tf.Max"(%3011, %dims3012) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?xi64>
  %3013 = "tf.Round"(%V__9) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims3014 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3014 = "tf.Mean"(%3013, %dims3014) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims3015 = "tf.Const"() { value = dense<[1, 2, 3, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3015 = "tf.Transpose"(%3014, %dims3015) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %3016 = "tf.Cast"(%3015) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi64>
  %dims3017 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3017 = "tf.Max"(%3016, %dims3017) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?x?xi64>
  %dims3018 = "tf.Const"() { value = dense<[0, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3018 = "tf.Min"(%3017, %dims3018) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %dims3019 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3019 = "tf.Mean"(%3018, %dims3019) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims3020 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3020 = "tf.Prod"(%3019, %dims3020) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?x?xi64>
  %dims3021 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3021 = "tf.Min"(%3020, %dims3021) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %3022 = "tf.Squeeze"(%3021) { squeeze_dims = [ 2 : i64 ], device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<?x?x?xi64>
  %dims3023 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3023 = "tf.Sum"(%V__91, %dims3023) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims3024 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3024 = "tf.Sum"(%3023, %dims3024) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %3025 = "tf.Squeeze"(%3024) { squeeze_dims = [ 2 : i64 ], device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<?x?x?xi64>
  %3026 = "tf.Relu"(%3025) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>) -> tensor<?x?x?xi64>
  %3027 = "tf.Acosh"(%V__31) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>) -> tensor<?xf32>
  %3028 = "tf.LeakyRelu"(%3027) { alpha = 3.000000e-01 : f32, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>) -> tensor<?xf32>
  %3029 = "tf.Cast"(%3028) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>) -> tensor<?xi64>
  %3030 = "tf.BiasAdd"(%3026, %3029) { data_format = "NCHW", device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<?xi64>) -> tensor<?x?x?xi64>
  %3031 = "tf.RightShift"(%3022, %3030) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<?x?x?xi64>) -> tensor<?x?x?xi64>
  %3032 = "tf.RightShift"(%3012, %3031) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<?x?x?xi64>) -> tensor<?x?x?xi64>
  %dims3033 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3033 = "tf.Mean"(%3032, %dims3033) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?xi64>
  %3034 = "tf.Squeeze"(%3033) { squeeze_dims = [ 0 : i64 ], device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>) -> tensor<?x?xi64>
  %dims3035 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3035 = "tf.Min"(%3034, %dims3035) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi64>, tensor<2xi32>) -> tensor<?x?xi64>
  %3036 = "tf.Const"() { value = dense<[1, 1, 1, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3037 = "tf.BroadcastTo"(%3035, %3036) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims3038 = "tf.Const"() { value = dense<[0, 3, 2, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3038 = "tf.Transpose"(%3037, %dims3038) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %3039 = "tf.Neg"(%3038) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims3040 = "tf.Const"() { value = dense<[2, 3, 1, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3040 = "tf.Transpose"(%3039, %dims3040) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %3041 = "tf.SquaredDifference"(%2999, %3040) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims3042 = "tf.Const"() { value = dense<[1, 2, 0, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3042 = "tf.Transpose"(%3041, %dims3042) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims3043 = "tf.Const"() { value = dense<[1, 0, 3, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3043 = "tf.Transpose"(%V__13, %dims3043) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<4xi32>) -> tensor<?x?x?x?xi1>
  %dims3044 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3044 = "tf.All"(%3043, %dims3044) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<4xi32>) -> tensor<i1>
  %dims3045 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3045 = "tf.Min"(%V__32, %dims3045) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %3046 = "tf.Square"(%V__2) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %3047 = "tf.Select"(%3044, %3045, %3046) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i1>, tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims3048 = "tf.Const"() { value = dense<[0, 2, 1, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3048 = "tf.Transpose"(%3047, %dims3048) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims3049 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3049 = "tf.Mean"(%3048, %dims3049) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims3050 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3050 = "tf.Min"(%3049, %dims3050) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %3051 = "tf.Shape"(%3050) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<4xi32>
  %3052 = "tf.SelectV2"(%V__44, %V__63, %V__92) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
  %3053 = "tf.Const"() { value = dense<[]> : tensor<0xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<0xi32>
  %3054 = "tf.Reshape"(%V__5, %3053) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi64>, tensor<0xi32>) -> tensor<i64>
  %dims3055 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3055 = "tf.Max"(%V__84, %dims3055) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?xi64>
  %3056 = "tf.Squeeze"(%3055) { squeeze_dims = [ 0 : i64, 1 : i64, 2 : i64 ], device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>) -> tensor<i64>
  %3057 = "tf.ClipByValue"(%3052, %3054, %3056) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<i64>
  %3058 = "tf.Fill"(%3051, %3057) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<4xi32>, tensor<i64>) -> tensor<?x?x?x?xi64>
  %3059 = "tf.Const"() { value = dense<[1, 1, 1, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3060 = "tf.BroadcastTo"(%3058, %3059) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims3061 = "tf.Const"() { value = dense<[3, 1, 2, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3061 = "tf.Transpose"(%3060, %dims3061) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %3062 = "tf.ZerosLike"(%V__67) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  %3063 = "tf.Mul"(%3062, %V__73) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  %dims3064 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3064 = "tf.Mean"(%3063, %dims3064) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?xi32>
  %dims3065 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3065 = "tf.Max"(%V__73, %dims3065) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?xi32>
  %dims3066 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3066 = "tf.Sum"(%3065, %dims3066) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?xi32>
  %dims3067 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3067 = "tf.Prod"(%3066, %dims3067) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?xi32>
  %3068 = "tf.Minimum"(%3064, %3067) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  %dims3069 = "tf.Const"() { value = dense<[2, 0, 1]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3069 = "tf.Transpose"(%3068, %dims3069) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?xi32>
  %dims3070 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3070 = "tf.Mean"(%V__7, %dims3070) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?xi32>
  %3071 = "tf.Squeeze"(%3070) { squeeze_dims = [ 0 : i64, 2 : i64 ], device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>) -> tensor<?xi32>
  %3072 = "tf.Squeeze"(%3071) { squeeze_dims = [ 0 : i64 ], device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi32>) -> tensor<i32>
  %dims3073 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3073 = "tf.Mean"(%V__40, %dims3073) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %3074 = "tf.Shape"(%3073) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<4xi32>
  %3075 = "tf.BroadcastTo"(%3072, %3074) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims3076 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3076 = "tf.Max"(%3075, %dims3076) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %3077 = "tf.Shape"(%3076) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<4xi32>
  %3078 = "tf.BroadcastTo"(%3069, %3077) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims3079 = "tf.Const"() { value = dense<[2, 3, 0, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3079 = "tf.Transpose"(%3078, %dims3079) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims3080 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3080 = "tf.Max"(%V__93, %dims3080) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims3081 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3081 = "tf.Min"(%3080, %dims3081) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims3082 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3082 = "tf.Min"(%3081, %dims3082) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims3083 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3083 = "tf.Min"(%V__65, %dims3083) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims3084 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3084 = "tf.Prod"(%3083, %dims3084) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %3085 = "tf.Div"(%3082, %3084) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims3086 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3086 = "tf.Mean"(%3085, %dims3086) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims3087 = "tf.Const"() { value = dense<[2, 0, 1]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3087 = "tf.Transpose"(%V__67, %dims3087) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?xi32>
  %dims3088 = "tf.Const"() { value = dense<[0, 2, 1]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3088 = "tf.Transpose"(%3087, %dims3088) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?xi32>
  %dims3089 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3089 = "tf.Mean"(%V__2, %dims3089) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?xi32>
  %3090 = "tf.FloorDiv"(%3088, %3089) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  %3091 = "tf.Squeeze"(%3090) { squeeze_dims = [ 1 : i64, 2 : i64 ], device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>) -> tensor<?xi32>
  %3092 = "tf.Round"(%3091) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi32>) -> tensor<?xi32>
  %3093 = "tf.BiasAdd"(%3086, %3092) { data_format = "NHWC", device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?xi32>) -> tensor<?x?x?x?xi32>
  %dims3094 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3094 = "tf.Prod"(%3093, %dims3094) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims3095 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3095 = "tf.Prod"(%3094, %dims3095) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims3096 = "tf.Const"() { value = dense<[2, 3, 0, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3096 = "tf.Transpose"(%3095, %dims3096) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims3097 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3097 = "tf.Prod"(%3096, %dims3097) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %3098 = "tf.Sub"(%3079, %3097) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims3099 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3099 = "tf.Min"(%3098, %dims3099) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %3100 = "tf.Shape"(%3099) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<4xi32>
  %3101 = "tf.BroadcastTo"(%3061, %3100) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims3102 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3102 = "tf.Prod"(%3101, %dims3102) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?xi64>
  %dims3103 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3103 = "tf.Mean"(%3102, %dims3103) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi64>, tensor<1xi32>) -> tensor<?xi64>
  %3104 = "tf.Cast"(%V__90) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %3105 = "tf.Maximum"(%3104, %V__81) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims3106 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3106 = "tf.Min"(%3105, %dims3106) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims3107 = "tf.Const"() { value = dense<[1, 0, 3, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3107 = "tf.Transpose"(%V__84, %dims3107) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims3108 = "tf.Const"() { value = dense<[1, 2, 0, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3108 = "tf.Transpose"(%V__62, %dims3108) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %3109 = "tf.Pow"(%3107, %3108) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %3110 = "tf.Minimum"(%3106, %3109) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims3111 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3111 = "tf.Min"(%V__43, %dims3111) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %3112 = "tf.Const"() { value = dense<[1, 1, 1, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %dims3114 = "tf.Const"() { value = dense<[1, 0, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3114 = "tf.Transpose"(%V__11, %dims3114) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims3115 = "tf.Const"() { value = dense<[0, 1, 3, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3115 = "tf.Transpose"(%3114, %dims3115) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %3116 = "tf.OnesLike"(%3115) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %3117 = "tf.Zeta"(%3111, %3116) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %3118 = "tf.Cast"(%3117) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xi64>
  %3119 = "tf.Add"(%3110, %3118) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims3120 = "tf.Const"() { value = dense<[1, 2, 3, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3120 = "tf.Transpose"(%3119, %dims3120) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims3121 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3121 = "tf.Min"(%3120, %dims3121) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims3122 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3122 = "tf.Mean"(%V__7, %dims3122) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?xi32>
  %3123 = "tf.Cast"(%3122) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>) -> tensor<?x?x?xi64>
  %dims3124 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3124 = "tf.Min"(%3123, %dims3124) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?xi64>
  %dims3125 = "tf.Const"() { value = dense<[1, 0, 3, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3125 = "tf.Transpose"(%V__4, %dims3125) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<4xi32>) -> tensor<?x?x?x?xi1>
  %dims3126 = "tf.Const"() { value = dense<[3, 1, 2, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3126 = "tf.Transpose"(%3125, %dims3126) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<4xi32>) -> tensor<?x?x?x?xi1>
  %3127 = "tf.Shape"(%3126) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>) -> tensor<4xi32>
  %3128 = "tf.BroadcastTo"(%3124, %3127) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims3129 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3129 = "tf.Mean"(%3128, %dims3129) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?xi64>
  %dims3130 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3130 = "tf.Max"(%3129, %dims3130) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi64>, tensor<1xi32>) -> tensor<?xi64>
  %dims3131 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3131 = "tf.Prod"(%V__59, %dims3131) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims3132 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3132 = "tf.Sum"(%3131, %dims3132) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?x?xi64>
  %dims3133 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3133 = "tf.Sum"(%3132, %dims3133) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims3134 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3134 = "tf.Mean"(%V__55, %dims3134) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?x?xi64>
  %dims3135 = "tf.Const"() { value = dense<[0, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3135 = "tf.Mean"(%3134, %dims3135) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %3136 = "tf.FloorDiv"(%3133, %3135) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims3137 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3137 = "tf.Min"(%3136, %dims3137) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims3138 = "tf.Const"() { value = dense<[1, 3, 2, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3138 = "tf.Transpose"(%3137, %dims3138) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %3139 = "tf.Shape"(%3138) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<4xi32>
  %3140 = "tf.BroadcastTo"(%3130, %3139) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %3141 = "tf.Relu"(%3140) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %3142 = "tf.Minimum"(%3121, %3141) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %3143 = "tf.Const"() { value = dense<[1, 1, 23, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3144 = "tf.BroadcastTo"(%3142, %3143) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %3145 = "tf.Shape"(%3144) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<4xi32>
  %3146 = "tf.BroadcastTo"(%3103, %3145) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims3147 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3147 = "tf.Max"(%3146, %dims3147) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?x?xi64>
  %dims3148 = "tf.Const"() { value = dense<[2, 0, 1, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3148 = "tf.Transpose"(%3147, %dims3148) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims3149 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3149 = "tf.Max"(%3148, %dims3149) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %3150 = "tf.BitwiseXor"(%3042, %3149) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims3151 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3151 = "tf.Mean"(%3150, %dims3151) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?x?xi64>
  %dims3152 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3152 = "tf.Max"(%3151, %dims3152) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %3153 = "tf.Pow"(%2916, %3152) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims3154 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3154 = "tf.Prod"(%3153, %dims3154) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?x?xi64>
  %dims3155 = "tf.Const"() { value = dense<[2, 1, 0, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3155 = "tf.Transpose"(%3154, %dims3155) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims3156 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3156 = "tf.Max"(%3155, %dims3156) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %dims3157 = "tf.Const"() { value = dense<[2, 1, 3, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3157 = "tf.Transpose"(%3156, %dims3157) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %3158 = "tf.ZerosLike"(%3157) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims3159 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3159 = "tf.Prod"(%3158, %dims3159) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %3160 = "tf.AddV2"(%2761, %3159) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims3161 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3161 = "tf.Mean"(%3160, %dims3161) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?xi64>
  %dims3162 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3162 = "tf.Max"(%3161, %dims3162) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi64>, tensor<1xi32>) -> tensor<?x?xi64>
  %dims3163 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3163 = "tf.Max"(%3162, %dims3163) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi64>, tensor<2xi32>) -> tensor<i64>
  %3164 = "tf.Invert"(%3163) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i64>) -> tensor<i64>
  %3165 = "tf.Const"() { value = dense<[1, 1, 1, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3166 = "tf.BroadcastTo"(%3164, %3165) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims3167 = "tf.Const"() { value = dense<[2, 3, 1, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3167 = "tf.Transpose"(%V__3, %dims3167) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims3168 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3168 = "tf.Mean"(%V__84, %dims3168) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?x?xi64>
  %3169 = "tf.AddV2"(%3167, %3168) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims3170 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3170 = "tf.Min"(%V__1, %dims3170) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<f32>
  %3171 = "tf.Const"() { value = dense<[]> : tensor<0xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<0xi32>
  %3172 = "tf.Reshape"(%3170, %3171) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>, tensor<0xi32>) -> tensor<f32>
  %3173 = "tf.Shape"(%3172) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>) -> tensor<0xi32>
  %3174 = "tf.Reshape"(%3169, %3173) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<0xi32>) -> tensor<i64>
  %3175 = "tf.Abs"(%3174) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i64>) -> tensor<i64>
  %dims3176 = "tf.Const"() { value = dense<[1, 0, 3, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3176 = "tf.Transpose"(%V__4, %dims3176) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<4xi32>) -> tensor<?x?x?x?xi1>
  %dims3177 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3177 = "tf.Any"(%3176, %dims3177) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<3xi32>) -> tensor<?x?x?x?xi1>
  %dims3178 = "tf.Const"() { value = dense<[3, 2, 0, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3178 = "tf.Transpose"(%V__4, %dims3178) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<4xi32>) -> tensor<?x?x?x?xi1>
  %3179 = "tf.LogicalOr"(%3177, %3178) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<?x?x?x?xi1>) -> tensor<?x?x?x?xi1>
  %dims3180 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3180 = "tf.Any"(%3179, %dims3180) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<3xi32>) -> tensor<?x?x?x?xi1>
  %3181 = "tf.Cast"(%3180) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>) -> tensor<?x?x?x?xi64>
  %dims3182 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3182 = "tf.Prod"(%3181, %dims3182) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims3183 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3183 = "tf.Min"(%3182, %dims3183) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?xi64>
  %3184 = "tf.Shape"(%3183) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>) -> tensor<3xi32>
  %3185 = "tf.Reshape"(%3175, %3184) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i64>, tensor<3xi32>) -> tensor<?x?x?xi64>
  %3186 = "tf.Cosh"(%V__50) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %3187 = "tf.Expm1"(%3186) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %3188 = "tf.Rint"(%3187) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %3189 = "tf.SelectV2"(%V__86, %V__71, %V__8) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi1>, tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %3190 = "tf.LessEqual"(%3188, %3189) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xi1>
  %dims3191 = "tf.Const"() { value = dense<[1, 2, 0, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3191 = "tf.Transpose"(%V__60, %dims3191) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %3192 = "tf.Pow"(%3191, %V__60) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims3193 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3193 = "tf.Min"(%3192, %dims3193) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims3194 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3194 = "tf.Max"(%3193, %dims3194) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?xi32>
  %3195 = "tf.Cast"(%3194) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>) -> tensor<?x?x?xi64>
  %dims3196 = "tf.Const"() { value = dense<[2, 1, 0]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3196 = "tf.Transpose"(%3195, %dims3196) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?xi64>
  %3197 = "tf.Shape"(%3196) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>) -> tensor<3xi32>
  %3198 = "tf.BroadcastTo"(%3190, %3197) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi1>, tensor<3xi32>) -> tensor<?x?x?xi1>
  %3199 = "tf.Cast"(%3198) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi1>) -> tensor<?x?x?xf32>
  %3200 = "tf.LeakyRelu"(%3199) { alpha = 3.000000e-01 : f32, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %3201 = "tf.ZerosLike"(%3200) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %3202 = "tf.Cast"(%3201) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xi64>
  %3203 = "tf.Minimum"(%3185, %3202) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<?x?x?xi64>) -> tensor<?x?x?xi64>
  %dims3204 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3204 = "tf.Max"(%3203, %dims3204) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<1xi32>) -> tensor<?x?xi64>
  %3205 = "tf.Neg"(%3204) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi64>) -> tensor<?x?xi64>
  %3206 = "tf.Const"() { value = dense<[1, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3207 = "tf.Expm1"(%V__74) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>) -> tensor<f32>
  %3208 = "tf.Rsqrt"(%V__48) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>) -> tensor<f32>
  %3209 = "tf.Zeta"(%3207, %3208) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %3210 = "tf.Abs"(%V__8) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %dims3211 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3211 = "tf.Sum"(%3210, %dims3211) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<3xi32>) -> tensor<f32>
  %3212 = "tf.Sqrt"(%3211) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>) -> tensor<f32>
  %3213 = "tf.Softplus"(%3212) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>) -> tensor<f32>
  %3214 = "tf.LessEqual"(%3209, %3213) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %3215 = "tf.Shape"(%3214) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i1>) -> tensor<0xi32>
  %3216 = "tf.BiasAdd"(%V__82, %V__94) { data_format = "NCHW", device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<?xi64>) -> tensor<?x?x?xi64>
  %dims3217 = "tf.Const"() { value = dense<[1, 0, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3217 = "tf.Transpose"(%3216, %dims3217) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?xi64>
  %3218 = "tf.FloorDiv"(%V__6, %V__6) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<?x?x?xi64>) -> tensor<?x?x?xi64>
  %3219 = "tf.FloorMod"(%3217, %3218) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<?x?x?xi64>) -> tensor<?x?x?xi64>
  %3220 = "tf.Square"(%3219) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>) -> tensor<?x?x?xi64>
  %dims3221 = "tf.Const"() { value = dense<[2, 0, 1]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3221 = "tf.Transpose"(%3220, %dims3221) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?xi64>
  %3222 = "tf.Squeeze"(%3221) { squeeze_dims = [ 0 : i64, 1 : i64, 2 : i64 ], device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>) -> tensor<i64>
  %3223 = "tf.Fill"(%3215, %3222) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<0xi32>, tensor<i64>) -> tensor<i64>
  %3224 = "tf.Fill"(%3206, %3223) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<2xi32>, tensor<i64>) -> tensor<?x?xi64>
  %3225 = "tf.Mod"(%3205, %3224) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi64>, tensor<?x?xi64>) -> tensor<?x?xi64>
  %dims3226 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3226 = "tf.Min"(%3225, %dims3226) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi64>, tensor<1xi32>) -> tensor<?xi64>
  %dims3227 = "tf.Const"() { value = dense<[0, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3227 = "tf.Prod"(%V__14, %dims3227) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?xi32>
  %3228 = "tf.Const"() { value = dense<[44, 73, 15, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3229 = "tf.BroadcastTo"(%3227, %3228) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims3230 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3230 = "tf.Mean"(%3229, %dims3230) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims3231 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3231 = "tf.Mean"(%3230, %dims3231) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?xi32>
  %dims3232 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3232 = "tf.Prod"(%3231, %dims3232) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?xi32>
  %dims3233 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3233 = "tf.Mean"(%3232, %dims3233) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?xi32>
  %3234 = "tf.Cast"(%3233) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>) -> tensor<?x?x?xi64>
  %dims3235 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3235 = "tf.Max"(%3234, %dims3235) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?xi64>
  %dims3236 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3236 = "tf.Max"(%V__80, %dims3236) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims3237 = "tf.Const"() { value = dense<[3, 2, 0, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3237 = "tf.Transpose"(%3236, %dims3237) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %3238 = "tf.Squeeze"(%3237) { squeeze_dims = [ 1 : i64 ], device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<?x?x?xi64>
  %3239 = "tf.Relu6"(%V__6) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>) -> tensor<?x?x?xi64>
  %3240 = "tf.Add"(%3239, %V__82) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<?x?x?xi64>) -> tensor<?x?x?xi64>
  %3241 = "tf.Mod"(%3238, %3240) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<?x?x?xi64>) -> tensor<?x?x?xi64>
  %3242 = "tf.Div"(%3235, %3241) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<?x?x?xi64>) -> tensor<?x?x?xi64>
  %dims3243 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3243 = "tf.Min"(%3242, %dims3243) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?xi64>
  %3244 = "tf.RightShift"(%V__67, %V__7) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  %dims3245 = "tf.Const"() { value = dense<[2, 0, 1]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3245 = "tf.Transpose"(%3244, %dims3245) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?xi32>
  %dims3246 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3246 = "tf.Min"(%V__20, %dims3246) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?xi32>
  %3247 = "tf.AddV2"(%3246, %V__0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  %3248 = "tf.BitwiseOr"(%3245, %3247) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  %3249 = "tf.LeakyRelu"(%V__35) { alpha = 2.000000e-01 : f32, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>) -> tensor<f32>
  %3250 = "tf.Exp"(%3249) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>) -> tensor<f32>
  %3251 = "tf.Shape"(%V__86) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi1>) -> tensor<3xi32>
  %3252 = "tf.Reshape"(%3250, %3251) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  %3253 = "tf.Cast"(%3252) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xi32>
  %dims3254 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3254 = "tf.Sum"(%3253, %dims3254) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?xi32>
  %3255 = "tf.ZerosLike"(%3254) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  %3256 = "tf.FloorMod"(%3248, %3255) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  %3257 = "tf.Cast"(%3256) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>) -> tensor<?x?x?xi1>
  %3258 = "tf.Shape"(%3257) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi1>) -> tensor<3xi32>
  %3259 = "tf.BroadcastTo"(%3243, %3258) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?xi64>
  %dims3260 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3260 = "tf.Max"(%3259, %dims3260) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?xi64>
  %3261 = "tf.Round"(%3260) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>) -> tensor<?x?x?xi64>
  %dims3262 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3262 = "tf.Sum"(%3261, %dims3262) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?xi64>
  %3263 = "tf.Neg"(%3262) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>) -> tensor<?x?x?xi64>
  %dims3264 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3264 = "tf.Prod"(%3263, %dims3264) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?xi64>
  %dims3265 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3265 = "tf.Max"(%3264, %dims3265) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<2xi32>) -> tensor<?xi64>
  %3266 = "tf.Relu"(%3265) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi64>) -> tensor<?xi64>
  %dims3267 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3267 = "tf.Min"(%V__49, %dims3267) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims3268 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3268 = "tf.Sum"(%3267, %dims3268) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %dims3269 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3269 = "tf.Sum"(%3268, %dims3269) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims3270 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3270 = "tf.Max"(%3269, %dims3270) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims3271 = "tf.Const"() { value = dense<[0, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3271 = "tf.Max"(%V__91, %dims3271) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %3272 = "tf.Square"(%3271) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims3273 = "tf.Const"() { value = dense<[0, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3273 = "tf.Sum"(%3272, %dims3273) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %3274 = "tf.Cast"(%3273) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %3275 = "tf.Sub"(%3270, %3274) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims3276 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3276 = "tf.Prod"(%V__16, %dims3276) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>, tensor<1xi32>) -> tensor<?xi32>
  %3277 = "tf.Cast"(%3276) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi32>) -> tensor<?xi1>
  %dims3278 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3278 = "tf.Prod"(%V__95, %dims3278) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims3279 = "tf.Const"() { value = dense<[2, 1, 0, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3279 = "tf.Transpose"(%3278, %dims3279) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %3280 = "tf.OnesLike"(%V__55) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %3281 = "tf.Select"(%3277, %3279, %3280) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi1>, tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims3282 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3282 = "tf.Min"(%3281, %dims3282) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %3283 = "tf.BitwiseAnd"(%3275, %3282) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %3284 = "tf.Const"() { value = dense<[1, 1, 1, 37]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3285 = "tf.BroadcastTo"(%3283, %3284) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims3286 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3286 = "tf.Mean"(%3285, %dims3286) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %dims3287 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3287 = "tf.Max"(%3286, %dims3287) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims3288 = "tf.Const"() { value = dense<[3, 1, 0, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3288 = "tf.Transpose"(%3287, %dims3288) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims3289 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3289 = "tf.Max"(%3288, %dims3289) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?x?xi64>
  %dims3290 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3290 = "tf.Mean"(%3289, %dims3290) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?x?xi64>
  %dims3291 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3291 = "tf.Max"(%3290, %dims3291) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?xi64>
  %3292 = "tf.Add"(%3266, %3291) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi64>, tensor<?xi64>) -> tensor<?xi64>
  %3293 = "tf.Cast"(%3292) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi64>) -> tensor<?xi64>
  %3294 = "tf.Minimum"(%3226, %3293) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi64>, tensor<?xi64>) -> tensor<?xi64>
  %3295 = "tf.Const"() { value = dense<[11, 61, 80, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3296 = "tf.BroadcastTo"(%3294, %3295) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims3297 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3297 = "tf.Min"(%3296, %dims3297) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims3298 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3298 = "tf.Mean"(%3297, %dims3298) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?x?xi64>
  %dims3299 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3299 = "tf.Sum"(%3298, %dims3299) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?x?xi64>
  %dims3300 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3300 = "tf.Sum"(%3299, %dims3300) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %3301 = "tf.Const"() { value = dense<[1, 1, 59, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3302 = "tf.BroadcastTo"(%3300, %3301) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims3303 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3303 = "tf.Sum"(%3302, %dims3303) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?x?xi64>
  %dims3304 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3304 = "tf.Prod"(%3303, %dims3304) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %3305 = "tf.Const"() { value = dense<[87, 47, 1, 32]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3306 = "tf.Cast"(%V__79) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>) -> tensor<?x?x?x?xi32>
  %dims3307 = "tf.Const"() { value = dense<[3, 0, 2, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3307 = "tf.Transpose"(%3306, %dims3307) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims3308 = "tf.Const"() { value = dense<[2, 0, 1, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3308 = "tf.Transpose"(%V__60, %dims3308) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %3309 = "tf.ZerosLike"(%3308) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %3310 = "tf.Mod"(%3307, %3309) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims3311 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3311 = "tf.Mean"(%3310, %dims3311) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %3312 = "tf.Neg"(%V__89) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>) -> tensor<?x?x?xi64>
  %dims3313 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3313 = "tf.Max"(%3312, %dims3313) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<3xi32>) -> tensor<i64>
  %dims3314 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3314 = "tf.Sum"(%V__89, %dims3314) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<3xi32>) -> tensor<i64>
  %3315 = "tf.TruncateDiv"(%3313, %3314) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %3316 = "tf.Shape"(%3315) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i64>) -> tensor<0xi32>
  %3317 = "tf.Reshape"(%3311, %3316) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<0xi32>) -> tensor<i32>
  %3318 = "tf.OnesLike"(%V__82) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>) -> tensor<?x?x?xi64>
  %3319 = "tf.Neg"(%V__96) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>) -> tensor<?x?x?xi64>
  %3320 = "tf.BitwiseOr"(%3318, %3319) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<?x?x?xi64>) -> tensor<?x?x?xi64>
  %3321 = "tf.OnesLike"(%3320) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>) -> tensor<?x?x?xi64>
  %3322 = "tf.BitwiseXor"(%V__23, %V__49) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims3323 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3323 = "tf.Max"(%3322, %dims3323) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %dims3324 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3324 = "tf.Mean"(%3323, %dims3324) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?x?xi64>
  %dims3325 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3325 = "tf.Sum"(%3324, %dims3325) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?xi64>
  %3326 = "tf.BitwiseOr"(%3321, %3325) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<?x?x?xi64>) -> tensor<?x?x?xi64>
  %dims3327 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3327 = "tf.Mean"(%3326, %dims3327) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?xi64>
  %3328 = "tf.Shape"(%3327) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>) -> tensor<3xi32>
  %3329 = "tf.BroadcastTo"(%3317, %3328) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i32>, tensor<3xi32>) -> tensor<?x?x?xi32>
  %dims3330 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3330 = "tf.Max"(%3329, %dims3330) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?xi32>
  %dims3331 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3331 = "tf.Mean"(%3330, %dims3331) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<3xi32>) -> tensor<i32>
  %dims3332 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3332 = "tf.Prod"(%V__64, %dims3332) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims3333 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3333 = "tf.Min"(%3332, %dims3333) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims3334 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3334 = "tf.All"(%V__58, %dims3334) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<3xi32>) -> tensor<?x?x?x?xi1>
  %3335 = "tf.Shape"(%3334) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>) -> tensor<4xi32>
  %3336 = "tf.BroadcastTo"(%3333, %3335) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims3337 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3337 = "tf.Max"(%3336, %dims3337) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?xi32>
  %3338 = "tf.Const"() { value = dense<[]> : tensor<0xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<0xi32>
  %3339 = "tf.Reshape"(%3337, %3338) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>, tensor<0xi32>) -> tensor<i32>
  %dims3340 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3340 = "tf.Prod"(%V__9, %dims3340) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims3341 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3341 = "tf.Min"(%V__20, %dims3341) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %3342 = "tf.FloorMod"(%3340, %3341) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims3343 = "tf.Const"() { value = dense<[0, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3343 = "tf.Mean"(%3342, %dims3343) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims3344 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3344 = "tf.Min"(%3343, %dims3344) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?xi32>
  %3345 = "tf.Round"(%V__9) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %3346 = "tf.Relu6"(%3345) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims3347 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3347 = "tf.Min"(%V__9, %dims3347) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %3348 = "tf.TruncateDiv"(%3346, %3347) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims3349 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3349 = "tf.Sum"(%3348, %dims3349) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?xi32>
  %3350 = "tf.FloorDiv"(%3344, %3349) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
  %dims3351 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3351 = "tf.Mean"(%3350, %dims3351) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>, tensor<2xi32>) -> tensor<i32>
  %3352 = "tf.LeftShift"(%3339, %3351) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %3353 = "tf.LessEqual"(%3331, %3352) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %dims3354 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3354 = "tf.Min"(%V__6, %dims3354) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?xi64>
  %3355 = "tf.Cast"(%3354) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>) -> tensor<?x?x?xi64>
  %dims3356 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3356 = "tf.Max"(%V__83, %dims3356) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?xi64>
  %3357 = "tf.Shape"(%3356) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>) -> tensor<3xi32>
  %3358 = "tf.BroadcastTo"(%3355, %3357) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?xi64>
  %dims3359 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3359 = "tf.Mean"(%3358, %dims3359) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<3xi32>) -> tensor<i64>
  %3360 = "tf.Const"() { value = dense<[1, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3362 = "tf.LeftShift"(%V__25, %V__25) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi64>, tensor<?x?xi64>) -> tensor<?x?xi64>
  %3363 = "tf.BitwiseAnd"(%V__25, %3362) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi64>, tensor<?x?xi64>) -> tensor<?x?xi64>
  %dims3364 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3364 = "tf.Prod"(%3363, %dims3364) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi64>, tensor<2xi32>) -> tensor<i64>
  %3365 = "tf.Minimum"(%3359, %3364) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %3366 = "tf.Sign"(%V__97) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims3367 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3367 = "tf.Sum"(%3366, %dims3367) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %dims3368 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3368 = "tf.Min"(%V__91, %dims3368) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %3369 = "tf.TruncateDiv"(%3367, %3368) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims3370 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3370 = "tf.Sum"(%3369, %dims3370) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %dims3371 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3371 = "tf.Prod"(%V__98, %dims3371) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?x?xi64>
  %dims3372 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3372 = "tf.Min"(%3371, %dims3372) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims3373 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3373 = "tf.Min"(%3372, %dims3373) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims3374 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3374 = "tf.Max"(%3373, %dims3374) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims3375 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3375 = "tf.Mean"(%3374, %dims3375) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %3376 = "tf.SquaredDifference"(%3370, %3375) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims3377 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3377 = "tf.Max"(%3376, %dims3377) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<i64>
  %3378 = "tf.Invert"(%3377) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i64>) -> tensor<i64>
  %3379 = "tf.LeftShift"(%3365, %3378) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %dims3380 = "tf.Const"() { value = dense<[2, 0, 1, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3380 = "tf.Transpose"(%V__11, %dims3380) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims3381 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3381 = "tf.Mean"(%3380, %dims3381) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims3382 = "tf.Const"() { value = dense<[2, 3, 1, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3382 = "tf.Transpose"(%V__53, %dims3382) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims3383 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3383 = "tf.Min"(%3382, %dims3383) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %3384 = "tf.Asin"(%V__40) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %3385 = "tf.ClipByValue"(%3381, %3383, %3384) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims3386 = "tf.Const"() { value = dense<[1, 2, 3, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3386 = "tf.Transpose"(%3385, %dims3386) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %3387 = "tf.Cast"(%3386) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xi32>
  %dims3388 = "tf.Const"() { value = dense<[2, 1, 0, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3388 = "tf.Transpose"(%3387, %dims3388) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims3389 = "tf.Const"() { value = dense<[2, 1, 0, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3389 = "tf.Transpose"(%3388, %dims3389) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims3390 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3390 = "tf.Mean"(%V__9, %dims3390) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims3391 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3391 = "tf.Min"(%3390, %dims3391) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %3392 = "tf.Shape"(%3391) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<4xi32>
  %3393 = "tf.Cast"(%V__99) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i64>) -> tensor<i32>
  %3394 = "tf.Mod"(%3393, %V__26) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %3395 = "tf.Fill"(%3392, %3394) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<4xi32>, tensor<i32>) -> tensor<?x?x?x?xi32>
  %dims3396 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3396 = "tf.Prod"(%3395, %dims3396) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims3397 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3397 = "tf.Max"(%3396, %dims3397) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims3398 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3398 = "tf.Max"(%3397, %dims3398) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %3399 = "tf.FloorDiv"(%3389, %3398) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims3400 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3400 = "tf.Prod"(%3399, %dims3400) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims3401 = "tf.Const"() { value = dense<[0, 1, 3, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3401 = "tf.Transpose"(%3400, %dims3401) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims3402 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3402 = "tf.Sum"(%3401, %dims3402) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims3403 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3403 = "tf.Sum"(%3402, %dims3403) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %3404 = "tf.Cast"(%3403) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi64>
  %dims3405 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3405 = "tf.Mean"(%3404, %dims3405) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?x?xi64>
  %dims3406 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3406 = "tf.Prod"(%3405, %dims3406) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<i64>
  %3407 = "tf.Pow"(%3379, %3406) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %3408 = "tf.Squeeze"(%V__58) { squeeze_dims = [ 2 : i64 ], device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>) -> tensor<?x?x?xi1>
  %3409 = "tf.Cast"(%3408) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi1>) -> tensor<?x?x?xi1>
  %3410 = "tf.Equal"(%V__6, %V__96) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<?x?x?xi64>) -> tensor<?x?x?xi1>
  %3411 = "tf.Const"() { value = dense<[1, 1, 1]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3412 = "tf.BroadcastTo"(%V__15, %3411) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi1>, tensor<3xi32>) -> tensor<?x?x?xi1>
  %3413 = "tf.Select"(%3409, %3410, %3412) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi1>, tensor<?x?x?xi1>, tensor<?x?x?xi1>) -> tensor<?x?x?xi1>
  %dims3414 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3414 = "tf.All"(%V__86, %dims3414) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi1>, tensor<1xi32>) -> tensor<?x?x?xi1>
  %3415 = "tf.SelectV2"(%3414, %V__86, %V__86) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi1>, tensor<?x?x?xi1>, tensor<?x?x?xi1>) -> tensor<?x?x?xi1>
  %dims3416 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3416 = "tf.Mean"(%V__96, %dims3416) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?xi64>
  %dims3417 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3417 = "tf.Min"(%3416, %dims3417) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?xi64>
  %3418 = "tf.Shape"(%3417) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>) -> tensor<3xi32>
  %3419 = "tf.BroadcastTo"(%3415, %3418) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi1>, tensor<3xi32>) -> tensor<?x?x?xi1>
  %3420 = "tf.Neg"(%V__90) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims3421 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3421 = "tf.Min"(%3420, %dims3421) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?xi64>
  %dims3422 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3422 = "tf.Min"(%3421, %dims3422) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?xi64>
  %3423 = "tf.Const"() { value = dense<[1, 1, 1, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3424 = "tf.BroadcastTo"(%V__91, %3423) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims3425 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3425 = "tf.Sum"(%3424, %dims3425) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?xi64>
  %3426 = "tf.Less"(%3422, %3425) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<?x?x?xi64>) -> tensor<?x?x?xi1>
  %3427 = "tf.Select"(%3413, %3419, %3426) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi1>, tensor<?x?x?xi1>, tensor<?x?x?xi1>) -> tensor<?x?x?xi1>
  %dims3428 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3428 = "tf.Any"(%3427, %dims3428) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi1>, tensor<2xi32>) -> tensor<?x?x?xi1>
  %dims3429 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3429 = "tf.All"(%3428, %dims3429) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi1>, tensor<1xi32>) -> tensor<?x?x?xi1>
  %dims3430 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3430 = "tf.All"(%3429, %dims3430) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi1>, tensor<3xi32>) -> tensor<i1>
  %3431 = "tf.Cast"(%3430) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i1>) -> tensor<i64>
  %3432 = "tf.Tanh"(%V__37) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %dims3433 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3433 = "tf.Sum"(%3432, %dims3433) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>, tensor<1xi32>) -> tensor<?xf32>
  %3434 = "tf.ZerosLike"(%3433) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>) -> tensor<?xf32>
  %3435 = "tf.Sign"(%3434) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>) -> tensor<?xf32>
  %3436 = "tf.Tanh"(%3435) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>) -> tensor<?xf32>
  %3437 = "tf.Div"(%V__72, %V__100) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims3438 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3438 = "tf.Min"(%3437, %dims3438) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims3439 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3439 = "tf.Prod"(%3438, %dims3439) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %dims3440 = "tf.Const"() { value = dense<[0, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3440 = "tf.Max"(%3439, %dims3440) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?xf32>
  %3441 = "tf.Zeta"(%3436, %3440) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %3442 = "tf.Digamma"(%3441) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>) -> tensor<?xf32>
  %dims3443 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3443 = "tf.Sum"(%3442, %dims3443) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>, tensor<1xi32>) -> tensor<?xf32>
  %3444 = "tf.Shape"(%3443) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>) -> tensor<1xi32>
  %dims3445 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3445 = "tf.Prod"(%V__96, %dims3445) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<3xi32>) -> tensor<i64>
  %3446 = "tf.Square"(%3445) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i64>) -> tensor<i64>
  %3447 = "tf.BitwiseOr"(%V__92, %V__101) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %3448 = "tf.Maximum"(%3446, %3447) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %dims3449 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3449 = "tf.Prod"(%V__96, %dims3449) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<3xi32>) -> tensor<i64>
  %3450 = "tf.Sign"(%3449) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i64>) -> tensor<i64>
  %3451 = "tf.Abs"(%V__101) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i64>) -> tensor<i64>
  %3452 = "tf.Sign"(%3451) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i64>) -> tensor<i64>
  %3453 = "tf.Add"(%3450, %3452) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %3454 = "tf.Mod"(%3448, %3453) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %3455 = "tf.Fill"(%3444, %3454) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<1xi32>, tensor<i64>) -> tensor<?xi64>
  %dims3456 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3456 = "tf.Max"(%3455, %dims3456) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi64>, tensor<1xi32>) -> tensor<i64>
  %3457 = "tf.LeftShift"(%3431, %3456) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %3458 = "tf.Select"(%3353, %3407, %3457) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
  %3459 = "tf.Fill"(%3305, %3458) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<4xi32>, tensor<i64>) -> tensor<?x?x?x?xi64>
  %3460 = "tf.Relu"(%3459) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims3461 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3461 = "tf.Sum"(%3460, %dims3461) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %3462 = "tf.Const"() { value = dense<[1, 1, 1, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3464 = "tf.BitwiseXor"(%3304, %3461) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims3465 = "tf.Const"() { value = dense<[0, 1, 3, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3465 = "tf.Transpose"(%3464, %dims3465) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims3466 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3466 = "tf.Mean"(%3465, %dims3466) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims3467 = "tf.Const"() { value = dense<[3, 1, 0, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3467 = "tf.Transpose"(%3466, %dims3467) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims3468 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3468 = "tf.Max"(%V__55, %dims3468) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?x?xi64>
  %3469 = "tf.Div"(%3468, %V__83) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %3470 = "tf.Sign"(%3469) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims3471 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3471 = "tf.Sum"(%3470, %dims3471) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<i64>
  %dims3472 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3472 = "tf.Min"(%V__11, %dims3472) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims3473 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3473 = "tf.Mean"(%3472, %dims3473) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %3474 = "tf.ZerosLike"(%3473) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %3475 = "tf.Shape"(%3474) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<4xi32>
  %3476 = "tf.Reshape"(%3471, %3475) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims3477 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3477 = "tf.Min"(%3476, %dims3477) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %dims3478 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3478 = "tf.Sum"(%3477, %dims3478) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %3479 = "tf.Cast"(%V__21) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>) -> tensor<i64>
  %3480 = "tf.Relu6"(%V__99) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i64>) -> tensor<i64>
  %3481 = "tf.TruncateDiv"(%3479, %3480) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %dims3482 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3482 = "tf.Prod"(%V__41, %dims3482) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %3483 = "tf.LeakyRelu"(%3482) { alpha = 2.000000e-01 : f32, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %3484 = "tf.Shape"(%3483) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<4xi32>
  %3485 = "tf.Reshape"(%3481, %3484) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims3486 = "tf.Const"() { value = dense<[0, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3486 = "tf.Prod"(%3485, %dims3486) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %dims3487 = "tf.Const"() { value = dense<[1, 0, 3, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3487 = "tf.Transpose"(%3486, %dims3487) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims3488 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3488 = "tf.Prod"(%3487, %dims3488) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %dims3489 = "tf.Const"() { value = dense<[1, 0, 3, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3489 = "tf.Transpose"(%3488, %dims3489) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %3490 = "tf.FloorDiv"(%3478, %3489) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims3491 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3491 = "tf.Max"(%3490, %dims3491) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %3492 = "tf.Const"() { value = dense<[]> : tensor<0xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<0xi32>
  %3493 = "tf.Reshape"(%V__54, %3492) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi1>, tensor<0xi32>) -> tensor<i1>
  %dims3494 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3494 = "tf.Max"(%V__42, %dims3494) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %dims3495 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3495 = "tf.Min"(%3494, %dims3495) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %3496 = "tf.Sign"(%V__45) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims3497 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3497 = "tf.Prod"(%3496, %dims3497) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %3498 = "tf.Select"(%3493, %3495, %3497) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i1>, tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %3499 = "tf.Minimum"(%V__70, %V__102) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %3500 = "tf.Acos"(%3499) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims3501 = "tf.Const"() { value = dense<[3, 1, 0, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3501 = "tf.Transpose"(%3500, %dims3501) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims3502 = "tf.Const"() { value = dense<[2, 0, 1, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3502 = "tf.Transpose"(%V__57, %dims3502) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims3503 = "tf.Const"() { value = dense<[3, 1, 0, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3503 = "tf.Transpose"(%3502, %dims3503) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims3504 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3504 = "tf.Min"(%3503, %dims3504) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %3505 = "tf.Add"(%3501, %3504) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %3506 = "tf.SquaredDifference"(%3498, %3505) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims3507 = "tf.Const"() { value = dense<[0, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3507 = "tf.Mean"(%3506, %dims3507) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %3508 = "tf.ZerosLike"(%3507) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %3509 = "tf.Acosh"(%3508) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %3510 = "tf.Acosh"(%3509) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims3511 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3511 = "tf.Min"(%3510, %dims3511) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %3512 = "tf.Log1p"(%3511) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims3513 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3513 = "tf.Mean"(%3512, %dims3513) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %dims3514 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3514 = "tf.Max"(%3513, %dims3514) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %3515 = "tf.Shape"(%3514) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<4xi32>
  %3516 = "tf.BroadcastTo"(%3491, %3515) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims3517 = "tf.Const"() { value = dense<[1, 3, 0, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3517 = "tf.Transpose"(%3516, %dims3517) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims3518 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3518 = "tf.Max"(%3517, %dims3518) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?x?xi64>
  %dims3519 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3519 = "tf.Prod"(%3518, %dims3519) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims3520 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3520 = "tf.Max"(%V__46, %dims3520) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %3521 = "tf.Erfc"(%3520) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims3522 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3522 = "tf.Min"(%3521, %dims3522) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %3523 = "tf.Ceil"(%V__100) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %3524 = "tf.Neg"(%3523) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims3525 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3525 = "tf.Min"(%3524, %dims3525) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %3526 = "tf.ApproximateEqual"(%3522, %3525) { tolerance = 1.000000e-05 : f32, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xi1>
  %dims3527 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3527 = "tf.Sum"(%V__103, %dims3527) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims3528 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3528 = "tf.Prod"(%3527, %dims3528) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %dims3529 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3529 = "tf.Mean"(%3528, %dims3529) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?xi64>
  %dims3530 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3530 = "tf.Max"(%V__32, %dims3530) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims3531 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3531 = "tf.Sum"(%3530, %dims3531) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %3532 = "tf.Shape"(%3531) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<4xi32>
  %3533 = "tf.BroadcastTo"(%3529, %3532) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %3534 = "tf.FloorDiv"(%V__104, %V__97) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims3535 = "tf.Const"() { value = dense<[1, 0, 3, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3535 = "tf.Transpose"(%V__105, %dims3535) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims3536 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3536 = "tf.Mean"(%V__104, %dims3536) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %3537 = "tf.ClipByValue"(%3534, %3535, %3536) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %3538 = "tf.Select"(%3526, %3533, %3537) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims3539 = "tf.Const"() { value = dense<[0, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3539 = "tf.Sum"(%3538, %dims3539) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %dims3540 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3540 = "tf.Min"(%3539, %dims3540) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims3541 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3541 = "tf.Min"(%3540, %dims3541) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims3542 = "tf.Const"() { value = dense<[2, 1, 3, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3542 = "tf.Transpose"(%3541, %dims3542) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %3543 = "tf.Digamma"(%V__70) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims3544 = "tf.Const"() { value = dense<[0, 3, 1, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3544 = "tf.Transpose"(%3543, %dims3544) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %3545 = "tf.IsInf"(%3544) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xi1>
  %3546 = "tf.Expm1"(%V__57) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %3547 = "tf.Sinh"(%3546) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %3548 = "tf.Neg"(%V__106) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %3549 = "tf.Expm1"(%3548) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %3550 = "tf.Select"(%3545, %3547, %3549) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %3551 = "tf.Softmax"(%3550) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %3552 = "tf.Elu"(%3551) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims3553 = "tf.Const"() { value = dense<[0, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3553 = "tf.Min"(%V__42, %dims3553) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?xf32>
  %3554 = "tf.Shape"(%V__46) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<4xi32>
  %3555 = "tf.BroadcastTo"(%3553, %3554) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %3556 = "tf.Elu"(%3555) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims3557 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3557 = "tf.Prod"(%V__28, %dims3557) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %dims3558 = "tf.Const"() { value = dense<[1, 0, 3, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3558 = "tf.Transpose"(%V__12, %dims3558) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %3559 = "tf.Add"(%3557, %3558) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims3560 = "tf.Const"() { value = dense<[1, 3, 0, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3560 = "tf.Transpose"(%3559, %dims3560) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %3561 = "tf.Zeta"(%3556, %3560) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %3562 = "tf.SquaredDifference"(%3552, %3561) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims3563 = "tf.Const"() { value = dense<[2, 1, 0, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3563 = "tf.Transpose"(%3562, %dims3563) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims3564 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3564 = "tf.Mean"(%3563, %dims3564) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %3565 = "tf.Shape"(%3564) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<4xi32>
  %3566 = "tf.BroadcastTo"(%3542, %3565) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims3567 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3567 = "tf.Mean"(%3566, %dims3567) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims3568 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3568 = "tf.Mean"(%3567, %dims3568) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %3569 = "tf.GreaterEqual"(%3519, %3568) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi1>
  %dims3570 = "tf.Const"() { value = dense<[3, 1, 2, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3570 = "tf.Transpose"(%3569, %dims3570) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<4xi32>) -> tensor<?x?x?x?xi1>
  %3571 = "tf.Cast"(%3570) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>) -> tensor<?x?x?x?xi1>
  %dims3572 = "tf.Const"() { value = dense<[2, 0, 1, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3572 = "tf.Transpose"(%3571, %dims3572) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<4xi32>) -> tensor<?x?x?x?xi1>
  %dims3573 = "tf.Const"() { value = dense<[2, 0, 1, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3573 = "tf.Transpose"(%3572, %dims3573) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<4xi32>) -> tensor<?x?x?x?xi1>
  %dims3574 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3574 = "tf.Any"(%3573, %dims3574) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<1xi32>) -> tensor<?x?x?x?xi1>
  %3575 = "tf.Cast"(%3574) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>) -> tensor<?x?x?x?xi32>
  %dims3576 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3576 = "tf.Prod"(%3575, %dims3576) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims3577 = "tf.Const"() { value = dense<[3, 2, 1, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3577 = "tf.Transpose"(%3576, %dims3577) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims3578 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3578 = "tf.Min"(%3577, %dims3578) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?xi32>
  %dims3579 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3579 = "tf.Mean"(%V__14, %dims3579) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims3580 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3580 = "tf.Max"(%3579, %dims3580) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims3581 = "tf.Const"() { value = dense<[2, 3, 0, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3581 = "tf.Transpose"(%3580, %dims3581) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims3582 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3582 = "tf.Max"(%3581, %dims3582) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims3583 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3583 = "tf.Mean"(%3582, %dims3583) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims3584 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3584 = "tf.Sum"(%3583, %dims3584) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?xi32>
  %dims3585 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3585 = "tf.Min"(%3584, %dims3585) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?xi32>
  %dims3586 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3586 = "tf.Prod"(%3585, %dims3586) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?xi32>
  %dims3587 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3587 = "tf.Min"(%3586, %dims3587) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?xi32>
  %dims3588 = "tf.Const"() { value = dense<[1, 3, 2, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3588 = "tf.Transpose"(%V__79, %dims3588) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<4xi32>) -> tensor<?x?x?x?xi1>
  %dims3589 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3589 = "tf.Any"(%3588, %dims3589) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<2xi32>) -> tensor<?x?x?x?xi1>
  %dims3590 = "tf.Const"() { value = dense<[1, 3, 0, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3590 = "tf.Transpose"(%V__4, %dims3590) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<4xi32>) -> tensor<?x?x?x?xi1>
  %dims3591 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3591 = "tf.All"(%3590, %dims3591) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<2xi32>) -> tensor<?x?x?x?xi1>
  %3592 = "tf.LogicalAnd"(%3589, %3591) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<?x?x?x?xi1>) -> tensor<?x?x?x?xi1>
  %dims3593 = "tf.Const"() { value = dense<[3, 1, 2, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3593 = "tf.Transpose"(%3592, %dims3593) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<4xi32>) -> tensor<?x?x?x?xi1>
  %3594 = "tf.Shape"(%3593) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>) -> tensor<4xi32>
  %3595 = "tf.BroadcastTo"(%3587, %3594) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims3596 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3596 = "tf.Sum"(%3595, %dims3596) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %3597 = "tf.ZerosLike"(%3596) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims3598 = "tf.Const"() { value = dense<[3, 2, 1, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3598 = "tf.Transpose"(%3597, %dims3598) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims3599 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3599 = "tf.Mean"(%3598, %dims3599) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims3600 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3600 = "tf.Sum"(%3599, %dims3600) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims3601 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3601 = "tf.Sum"(%3600, %dims3601) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims3602 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3602 = "tf.Sum"(%3601, %dims3602) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %3603 = "tf.OnesLike"(%3602) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims3604 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3604 = "tf.Min"(%3603, %dims3604) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims3605 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3605 = "tf.Prod"(%3604, %dims3605) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims3606 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3606 = "tf.Mean"(%3605, %dims3606) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims3607 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3607 = "tf.Max"(%3606, %dims3607) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims3608 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3608 = "tf.Min"(%V__65, %dims3608) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims3609 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3609 = "tf.Min"(%3608, %dims3609) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims3610 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3610 = "tf.Max"(%3609, %dims3610) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims3611 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3611 = "tf.Max"(%3610, %dims3611) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims3612 = "tf.Const"() { value = dense<[1, 2, 0, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3612 = "tf.Transpose"(%3611, %dims3612) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims3613 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3613 = "tf.Prod"(%3612, %dims3613) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?xi32>
  %dims3614 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3614 = "tf.Min"(%V__87, %dims3614) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?x?xi64>
  %3615 = "tf.Round"(%3614) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims3616 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3616 = "tf.Sum"(%V__55, %dims3616) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %3617 = "tf.Mod"(%3615, %3616) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %3618 = "tf.Shape"(%3617) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<4xi32>
  %3619 = "tf.BroadcastTo"(%3613, %3618) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims3620 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3620 = "tf.Prod"(%3619, %dims3620) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims3621 = "tf.Const"() { value = dense<[0, 3, 1, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3621 = "tf.Transpose"(%V__14, %dims3621) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %3622 = "tf.FloorDiv"(%3621, %V__20) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims3623 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3623 = "tf.Mean"(%3622, %dims3623) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims3624 = "tf.Const"() { value = dense<[3, 2, 0, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3624 = "tf.Transpose"(%3623, %dims3624) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims3625 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3625 = "tf.Sum"(%3624, %dims3625) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims3626 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3626 = "tf.Min"(%V__60, %dims3626) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims3627 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3627 = "tf.Sum"(%3626, %dims3627) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims3628 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3628 = "tf.Min"(%V__107, %dims3628) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %3629 = "tf.Minimum"(%3627, %3628) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims3630 = "tf.Const"() { value = dense<[2, 1, 0, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3630 = "tf.Transpose"(%3629, %dims3630) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %3631 = "tf.Pow"(%3625, %3630) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %3632 = "tf.LeftShift"(%3620, %3631) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %3633 = "tf.Maximum"(%3607, %3632) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims3634 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3634 = "tf.Max"(%3633, %dims3634) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims3635 = "tf.Const"() { value = dense<[0, 2, 1, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3635 = "tf.Transpose"(%3634, %dims3635) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims3636 = "tf.Const"() { value = dense<[3, 2, 0, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3636 = "tf.Transpose"(%3635, %dims3636) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims3637 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3637 = "tf.Min"(%3636, %dims3637) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?xi32>
  %dims3638 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3638 = "tf.Mean"(%V__9, %dims3638) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims3639 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3639 = "tf.Sum"(%3638, %dims3639) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims3640 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3640 = "tf.Sum"(%3639, %dims3640) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?xi32>
  %dims3641 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3641 = "tf.Max"(%V__73, %dims3641) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?xi32>
  %dims3642 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3642 = "tf.Prod"(%3641, %dims3642) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?xi32>
  %3643 = "tf.RightShift"(%3640, %3642) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  %dims3644 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3644 = "tf.Min"(%3643, %dims3644) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?xi32>
  %3645 = "tf.OnesLike"(%3644) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  %dims3646 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3646 = "tf.Sum"(%3645, %dims3646) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?xi32>
  %3647 = "tf.Invert"(%3646) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  %dims3648 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3648 = "tf.Max"(%3647, %dims3648) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<1xi32>) -> tensor<?x?xi32>
  %3649 = "tf.Relu6"(%3648) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>) -> tensor<?x?xi32>
  %dims3650 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3650 = "tf.Prod"(%3649, %dims3650) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>, tensor<1xi32>) -> tensor<?xi32>
  %dims3651 = "tf.Const"() { value = dense<[0, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3651 = "tf.Sum"(%V__1, %dims3651) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %dims3652 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3652 = "tf.Min"(%3651, %dims3652) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims3653 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3653 = "tf.Max"(%3652, %dims3653) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %dims3654 = "tf.Const"() { value = dense<[0, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3654 = "tf.Sum"(%3653, %dims3654) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %3655 = "tf.Squeeze"(%V__36) { squeeze_dims = [ 1 : i64, 2 : i64, 3 : i64 ], device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?xf32>
  %3656 = "tf.Polygamma"(%3655, %V__31) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %3657 = "tf.BiasAdd"(%3654, %3656) { data_format = "NCHW", device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?xf32>) -> tensor<?x?x?x?xf32>
  %3658 = "tf.Lgamma"(%3657) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %3659 = "tf.Selu"(%3658) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims3660 = "tf.Const"() { value = dense<[0, 2, 3, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3660 = "tf.Transpose"(%3659, %dims3660) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims3661 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3661 = "tf.Sum"(%3660, %dims3661) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?xf32>
  %3662 = "tf.Shape"(%3661) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<3xi32>
  %3663 = "tf.BroadcastTo"(%3650, %3662) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi32>, tensor<3xi32>) -> tensor<?x?x?xi32>
  %dims3664 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3664 = "tf.Max"(%3663, %dims3664) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?xi32>
  %dims3665 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3665 = "tf.Sum"(%3664, %dims3665) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?xi32>
  %dims3666 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3666 = "tf.Sum"(%3665, %dims3666) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<2xi32>) -> tensor<?xi32>
  %3667 = "tf.Sigmoid"(%V__40) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims3668 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3668 = "tf.Min"(%V__100, %dims3668) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %3669 = "tf.ApproximateEqual"(%3667, %3668) { tolerance = 1.000000e-04 : f32, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xi1>
  %dims3670 = "tf.Const"() { value = dense<[2, 1, 3, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3670 = "tf.Transpose"(%V__60, %dims3670) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims3671 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3671 = "tf.Prod"(%3670, %dims3671) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims3672 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3672 = "tf.Max"(%3671, %dims3672) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %3673 = "tf.Round"(%V__30) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi32>) -> tensor<?xi32>
  %3674 = "tf.Const"() { value = dense<[1, 1, 1, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3676 = "tf.SelectV2"(%3669, %3672, %3673) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<?x?x?x?xi32>, tensor<?xi32>) -> tensor<?x?x?x?xi32>
  %dims3677 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3677 = "tf.Sum"(%3676, %dims3677) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?xi32>
  %3678 = "tf.FloorMod"(%V__93, %V__93) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims3679 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3679 = "tf.Max"(%V__9, %dims3679) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %3680 = "tf.Mul"(%3678, %3679) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims3681 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3681 = "tf.Sum"(%3680, %dims3681) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims3682 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3682 = "tf.Prod"(%3681, %dims3682) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims3683 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3683 = "tf.Max"(%3682, %dims3683) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims3684 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3684 = "tf.Min"(%3683, %dims3684) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims3685 = "tf.Const"() { value = dense<[1, 2, 3, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3685 = "tf.Transpose"(%3684, %dims3685) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims3686 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3686 = "tf.Prod"(%3685, %dims3686) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims3687 = "tf.Const"() { value = dense<[3, 0, 1, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3687 = "tf.Transpose"(%3686, %dims3687) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %3688 = "tf.Shape"(%3687) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<4xi32>
  %3689 = "tf.BroadcastTo"(%3677, %3688) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims3690 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3690 = "tf.Sum"(%3689, %dims3690) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims3691 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3691 = "tf.Max"(%3690, %dims3691) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims3692 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3692 = "tf.Prod"(%3691, %dims3692) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?xi32>
  %3693 = "tf.RightShift"(%3666, %3692) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
  %3694 = "tf.BitwiseAnd"(%3637, %3693) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
  %dims3695 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3695 = "tf.Max"(%V__0, %dims3695) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<2xi32>) -> tensor<?xi32>
  %dims3696 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3696 = "tf.Prod"(%3695, %dims3696) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32>
  %3697 = "tf.Shape"(%V__15) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi1>) -> tensor<2xi32>
  %dims3699 = "tf.Const"() { value = dense<[2, 0, 1, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3699 = "tf.Transpose"(%V__79, %dims3699) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<4xi32>) -> tensor<?x?x?x?xi1>
  %3700 = "tf.Select"(%3699, %V__108, %V__57) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %3701 = "tf.Shape"(%3700) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<4xi32>
  %3702 = "tf.BroadcastTo"(%3696, %3701) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %3703 = "tf.Invert"(%V__16) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>) -> tensor<?x?xi32>
  %3704 = "tf.BitwiseAnd"(%3703, %V__16) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
  %3705 = "tf.OnesLike"(%3704) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>) -> tensor<?x?xi32>
  %3706 = "tf.Round"(%V__109) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %3707 = "tf.Squeeze"(%3706) { squeeze_dims = [ 2 : i64, 3 : i64 ], device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?xi32>
  %3708 = "tf.Neg"(%3707) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>) -> tensor<?x?xi32>
  %dims3709 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3709 = "tf.Min"(%3708, %dims3709) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>, tensor<2xi32>) -> tensor<?x?xi32>
  %3710 = "tf.AddV2"(%3705, %3709) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
  %dims3711 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3711 = "tf.Max"(%3710, %dims3711) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>, tensor<1xi32>) -> tensor<?xi32>
  %3712 = "tf.BiasAdd"(%3702, %3711) { data_format = "NCHW", device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?xi32>) -> tensor<?x?x?x?xi32>
  %3713 = "tf.OnesLike"(%3712) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims3714 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3714 = "tf.Mean"(%3713, %dims3714) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %3715 = "tf.Abs"(%3714) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %3716 = "tf.ZerosLike"(%3715) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims3717 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3717 = "tf.All"(%V__15, %dims3717) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi1>, tensor<1xi32>) -> tensor<?xi1>
  %dims3718 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3718 = "tf.All"(%3717, %dims3718) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi1>, tensor<1xi32>) -> tensor<?xi1>
  %3719 = "tf.Shape"(%V__72) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<4xi32>
  %dims3721 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3721 = "tf.Sum"(%V__107, %dims3721) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %3722 = "tf.SquaredDifference"(%3721, %V__32) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims3723 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3723 = "tf.Max"(%3722, %dims3723) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims3724 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3724 = "tf.Sum"(%3723, %dims3724) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims3725 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3725 = "tf.Min"(%V__2, %dims3725) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %3726 = "tf.Const"() { value = dense<[1, 1, 1, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3727 = "tf.BroadcastTo"(%3725, %3726) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims3728 = "tf.Const"() { value = dense<[1, 3, 0, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3728 = "tf.Transpose"(%3727, %dims3728) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %3729 = "tf.SelectV2"(%3718, %3724, %3728) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi1>, tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims3730 = "tf.Const"() { value = dense<[0, 2, 3, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3730 = "tf.Transpose"(%3729, %dims3730) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims3731 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3731 = "tf.Max"(%3730, %dims3731) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims3732 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3732 = "tf.Sum"(%3731, %dims3732) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims3733 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3733 = "tf.Min"(%3732, %dims3733) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims3734 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3734 = "tf.Max"(%3733, %dims3734) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims3735 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3735 = "tf.Min"(%3734, %dims3735) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims3736 = "tf.Const"() { value = dense<[1, 3, 0, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3736 = "tf.Transpose"(%3735, %dims3736) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims3737 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3737 = "tf.Prod"(%3736, %dims3737) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims3738 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3738 = "tf.Max"(%3737, %dims3738) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims3739 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3739 = "tf.Max"(%3738, %dims3739) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims3740 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3740 = "tf.Prod"(%3739, %dims3740) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims3741 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3741 = "tf.Min"(%3740, %dims3741) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %3742 = "tf.TruncateDiv"(%3716, %3741) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims3743 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3743 = "tf.Sum"(%3742, %dims3743) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims3744 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3744 = "tf.Max"(%3743, %dims3744) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims3745 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3745 = "tf.Prod"(%3744, %dims3745) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims3746 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3746 = "tf.Max"(%3745, %dims3746) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %3747 = "tf.Neg"(%3746) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %3748 = "tf.OnesLike"(%3747) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims3749 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3749 = "tf.Sum"(%3748, %dims3749) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims3750 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3750 = "tf.Max"(%V__41, %dims3750) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?xf32>
  %3751 = "tf.Shape"(%V__55) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<4xi32>
  %3753 = "tf.Softplus"(%V__37) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %3754 = "tf.Const"() { value = dense<[1, 1, 1, 56]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3755 = "tf.BroadcastTo"(%3753, %3754) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims3756 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3756 = "tf.Max"(%3755, %dims3756) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %3757 = "tf.Xlog1py"(%3750, %3756) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims3758 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3758 = "tf.Min"(%3757, %dims3758) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %3759 = "tf.Erf"(%3758) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims3760 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3760 = "tf.Sum"(%3759, %dims3760) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?xf32>
  %3761 = "tf.Shape"(%3760) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>) -> tensor<1xi32>
  %dims3762 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3762 = "tf.Prod"(%V__73, %dims3762) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<3xi32>) -> tensor<i32>
  %3763 = "tf.Neg"(%3762) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i32>) -> tensor<i32>
  %3764 = "tf.Shape"(%V__61) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i64>) -> tensor<0xi32>
  %3765 = "tf.Reshape"(%3763, %3764) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
  %3766 = "tf.OnesLike"(%3765) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i32>) -> tensor<i32>
  %dims3767 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3767 = "tf.Min"(%V__7, %dims3767) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<3xi32>) -> tensor<i32>
  %3768 = "tf.Shape"(%V__0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>) -> tensor<3xi32>
  %3769 = "tf.Reshape"(%3767, %3768) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i32>, tensor<3xi32>) -> tensor<?x?x?xi32>
  %dims3770 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3770 = "tf.Prod"(%3769, %dims3770) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?xi32>
  %dims3771 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3771 = "tf.Prod"(%3770, %dims3771) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<3xi32>) -> tensor<i32>
  %3772 = "tf.FloorDiv"(%3766, %3771) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %3773 = "tf.Fill"(%3761, %3772) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<1xi32>, tensor<i32>) -> tensor<?xi32>
  %3774 = "tf.Relu"(%V__12) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %3775 = "tf.Erfc"(%3774) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims3776 = "tf.Const"() { value = dense<[1, 2, 0, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3776 = "tf.Transpose"(%3775, %dims3776) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims3777 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3777 = "tf.Mean"(%3776, %dims3777) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %dims3778 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3778 = "tf.Max"(%3777, %dims3778) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims3779 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3779 = "tf.Sum"(%3778, %dims3779) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %3780 = "tf.Const"() { value = dense<[53, 1, 1, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3781 = "tf.BroadcastTo"(%V__27, %3780) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims3782 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3782 = "tf.Mean"(%3781, %dims3782) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %dims3783 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3783 = "tf.Prod"(%3782, %dims3783) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %3784 = "tf.Asinh"(%3783) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %3785 = "tf.Sub"(%3779, %3784) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %3786 = "tf.Digamma"(%V__69) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims3787 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3787 = "tf.Max"(%3786, %dims3787) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims3788 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3788 = "tf.Min"(%V__36, %dims3788) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?xf32>
  %3789 = "tf.BiasAdd"(%3787, %3788) { data_format = "NCHW", device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?xf32>) -> tensor<?x?x?x?xf32>
  %3790 = "tf.Const"() { value = dense<[25, 1, 1, 34]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3791 = "tf.BroadcastTo"(%V__38, %3790) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims3792 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3792 = "tf.Mean"(%3791, %dims3792) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims3793 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3793 = "tf.Prod"(%3792, %dims3793) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %3794 = "tf.Shape"(%3793) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<4xi32>
  %3796 = "tf.RealDiv"(%3785, %3789) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims3797 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3797 = "tf.Sum"(%3796, %dims3797) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %3798 = "tf.Shape"(%3797) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<4xi32>
  %3799 = "tf.BroadcastTo"(%3773, %3798) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims3800 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3800 = "tf.Prod"(%3799, %dims3800) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims3801 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3801 = "tf.Sum"(%3800, %dims3801) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims3802 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3802 = "tf.Min"(%3801, %dims3802) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims3803 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3803 = "tf.Sum"(%3802, %dims3803) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims3804 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3804 = "tf.Sum"(%3803, %dims3804) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %3805 = "tf.Maximum"(%3749, %3804) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims3806 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3806 = "tf.Min"(%3805, %dims3806) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %3807 = "tf.Squeeze"(%3806) { squeeze_dims = [ 1 : i64, 2 : i64, 3 : i64 ], device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?xi32>
  %3808 = "tf.ClipByValue"(%3578, %3694, %3807) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi32>, tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
  %3809 = "tf.LeakyRelu"(%V__37) { alpha = 1.000000e-01 : f32, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %3810 = "tf.Const"() { value = dense<[1, 1, 1]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3811 = "tf.BroadcastTo"(%3809, %3810) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  %dims3812 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3812 = "tf.Min"(%3811, %dims3812) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<1xi32>) -> tensor<?x?xf32>
  %dims3813 = "tf.Const"() { value = dense<[1, 3, 0, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3813 = "tf.Transpose"(%V__65, %dims3813) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims3814 = "tf.Const"() { value = dense<[0, 1, 3, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3814 = "tf.Transpose"(%3813, %dims3814) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims3815 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3815 = "tf.Prod"(%3814, %dims3815) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %3816 = "tf.Shape"(%3815) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<4xi32>
  %3818 = "tf.Pow"(%V__40, %V__53) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %3819 = "tf.Const"() { value = dense<[1, 1, 1, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3821 = "tf.Xlog1py"(%3818, %V__100) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %3822 = "tf.ZerosLike"(%3821) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims3823 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3823 = "tf.Prod"(%3822, %dims3823) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  %3824 = "tf.Rsqrt"(%3823) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %3825 = "tf.Equal"(%3812, %3824) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xi1>
  %dims3826 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3826 = "tf.Max"(%V__109, %dims3826) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims3827 = "tf.Const"() { value = dense<[3, 1, 2, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3827 = "tf.Transpose"(%3826, %dims3827) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims3828 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3828 = "tf.Mean"(%3827, %dims3828) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %3829 = "tf.Square"(%V__18) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %3830 = "tf.Mul"(%3829, %V__64) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %3831 = "tf.Mul"(%3828, %3830) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %3832 = "tf.Const"() { value = dense<[1, 1, 1, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3833 = "tf.Reshape"(%V__76, %3832) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %3834 = "tf.Relu"(%3833) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims3835 = "tf.Const"() { value = dense<[0, 3, 2, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3835 = "tf.Transpose"(%3834, %dims3835) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %3836 = "tf.Abs"(%3835) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims3837 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3837 = "tf.Sum"(%3836, %dims3837) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims3838 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3838 = "tf.Sum"(%3837, %dims3838) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %3839 = "tf.BitwiseOr"(%3831, %3838) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims3840 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3840 = "tf.Mean"(%3839, %dims3840) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %3841 = "tf.Cast"(%3840) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi64>
  %dims3842 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3842 = "tf.Prod"(%3841, %dims3842) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %3843 = "tf.Shape"(%3842) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<4xi32>
  %3844 = "tf.BroadcastTo"(%3825, %3843) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<4xi32>) -> tensor<?x?x?x?xi1>
  %3845 = "tf.Cast"(%3844) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>) -> tensor<?x?x?x?xi1>
  %dims3846 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3846 = "tf.All"(%3845, %dims3846) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<2xi32>) -> tensor<?x?x?x?xi1>
  %dims3847 = "tf.Const"() { value = dense<[1, 0, 3, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3847 = "tf.Transpose"(%3846, %dims3847) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<4xi32>) -> tensor<?x?x?x?xi1>
  %dims3848 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3848 = "tf.Mean"(%V__107, %dims3848) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %3849 = "tf.FloorDiv"(%3848, %V__2) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dims3850 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3850 = "tf.Prod"(%3849, %dims3850) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims3851 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3851 = "tf.Sum"(%3850, %dims3851) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims3852 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3852 = "tf.Mean"(%3851, %dims3852) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims3853 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3853 = "tf.Sum"(%V__7, %dims3853) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<2xi32>) -> tensor<?xi32>
  %3854 = "tf.Shape"(%V__103) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<4xi32>
  %3855 = "tf.BroadcastTo"(%3853, %3854) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims3856 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3856 = "tf.Min"(%3855, %dims3856) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %3857 = "tf.GreaterEqual"(%3852, %3856) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi1>
  %dims3858 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3858 = "tf.Any"(%3857, %dims3858) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<3xi32>) -> tensor<?xi1>
  %dims3859 = "tf.Const"() { value = dense<[1, 2, 0, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3859 = "tf.Transpose"(%V__83, %dims3859) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %3860 = "tf.Add"(%3859, %V__49) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims3861 = "tf.Const"() { value = dense<[2, 0, 1, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3861 = "tf.Transpose"(%3860, %dims3861) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims3862 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3862 = "tf.Mean"(%3861, %dims3862) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?xi64>
  %3863 = "tf.Relu6"(%3862) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>) -> tensor<?x?x?xi64>
  %dims3864 = "tf.Const"() { value = dense<[1, 0, 3, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3864 = "tf.Transpose"(%V__110, %dims3864) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<4xi32>) -> tensor<?x?x?x?xi1>
  %dims3865 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3865 = "tf.All"(%V__58, %dims3865) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<1xi32>) -> tensor<?x?x?x?xi1>
  %3866 = "tf.LogicalOr"(%3864, %3865) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<?x?x?x?xi1>) -> tensor<?x?x?x?xi1>
  %3867 = "tf.Shape"(%3866) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>) -> tensor<4xi32>
  %3868 = "tf.BroadcastTo"(%3863, %3867) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %3869 = "tf.Round"(%3868) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %3870 = "tf.OnesLike"(%V__23) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims3871 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3871 = "tf.Min"(%3870, %dims3871) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?x?xi64>
  %dims3872 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3872 = "tf.Max"(%3871, %dims3872) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims3873 = "tf.Const"() { value = dense<[0, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3873 = "tf.Max"(%3872, %dims3873) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %dims3874 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3874 = "tf.Sum"(%3873, %dims3874) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims3875 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3875 = "tf.Prod"(%3874, %dims3875) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %3876 = "tf.Squeeze"(%V__96) { squeeze_dims = [ 0 : i64 ], device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>) -> tensor<?x?xi64>
  %dims3877 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3877 = "tf.Max"(%3876, %dims3877) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi64>, tensor<1xi32>) -> tensor<?xi64>
  %3878 = "tf.Shape"(%V__111) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<4xi32>
  %3879 = "tf.BroadcastTo"(%3877, %3878) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims3880 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3880 = "tf.Min"(%3879, %dims3880) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?x?xi64>
  %3881 = "tf.Mod"(%3875, %3880) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %3882 = "tf.Select"(%3858, %3869, %3881) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi1>, tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %3883 = "tf.Invert"(%3882) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %3884 = "tf.Shape"(%3883) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<4xi32>
  %3885 = "tf.BroadcastTo"(%3847, %3884) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<4xi32>) -> tensor<?x?x?x?xi1>
  %3886 = "tf.Cast"(%3885) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>) -> tensor<?x?x?x?xi32>
  %dims3887 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3887 = "tf.Mean"(%3886, %dims3887) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %dims3888 = "tf.Const"() { value = dense<[3, 1, 0, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3888 = "tf.Transpose"(%3887, %dims3888) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims3889 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3889 = "tf.Min"(%V__90, %dims3889) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims3890 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3890 = "tf.Min"(%3889, %dims3890) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %3891 = "tf.ZerosLike"(%3890) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %3892 = "tf.Cast"(%3891) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xf32>
  %dims3893 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3893 = "tf.Mean"(%3892, %dims3893) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %dims3894 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3894 = "tf.Sum"(%V__70, %dims3894) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims3895 = "tf.Const"() { value = dense<[1, 2, 3, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3895 = "tf.Transpose"(%3894, %dims3895) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims3896 = "tf.Const"() { value = dense<[3, 1, 0, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3896 = "tf.Transpose"(%V__42, %dims3896) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %3897 = "tf.Mod"(%3895, %3896) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %3898 = "tf.GreaterEqual"(%3893, %3897) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xi1>
  %3899 = "tf.Squeeze"(%3898) { squeeze_dims = [ 0 : i64 ], device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>) -> tensor<?x?x?xi1>
  %dims3900 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3900 = "tf.Max"(%V__50, %dims3900) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?xf32>
  %3901 = "tf.Cast"(%3900) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xi64>
  %3902 = "tf.Round"(%3901) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>) -> tensor<?x?x?xi64>
  %dims3903 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3903 = "tf.Prod"(%V__89, %dims3903) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?xi64>
  %dims3904 = "tf.Const"() { value = dense<[1, 0, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3904 = "tf.Transpose"(%3903, %dims3904) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?xi64>
  %3905 = "tf.Abs"(%3904) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>) -> tensor<?x?x?xi64>
  %3906 = "tf.TruncateDiv"(%3902, %3905) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<?x?x?xi64>) -> tensor<?x?x?xi64>
  %dims3907 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3907 = "tf.Min"(%3906, %dims3907) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?xi64>
  %dims3908 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3908 = "tf.Max"(%3907, %dims3908) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?xi64>
  %dims3909 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3909 = "tf.Prod"(%3908, %dims3909) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?xi64>
  %dims3910 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3910 = "tf.Min"(%3909, %dims3910) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?xi64>
  %dims3911 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3911 = "tf.Prod"(%3910, %dims3911) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?xi64>
  %dims3912 = "tf.Const"() { value = dense<[0, 2, 1, 3]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3912 = "tf.Transpose"(%V__59, %dims3912) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims3913 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3913 = "tf.Prod"(%3912, %dims3913) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %dims3914 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3914 = "tf.Prod"(%3913, %dims3914) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?xi64>
  %3915 = "tf.OnesLike"(%3914) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>) -> tensor<?x?x?xi64>
  %dims3916 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3916 = "tf.Prod"(%3915, %dims3916) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?xi64>
  %dims3917 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3917 = "tf.Mean"(%V__87, %dims3917) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?x?xi64>
  %dims3918 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3918 = "tf.Max"(%3917, %dims3918) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?xi64>
  %dims3919 = "tf.Const"() { value = dense<[0, 2, 1]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3919 = "tf.Transpose"(%V__89, %dims3919) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?xi64>
  %3920 = "tf.RightShift"(%3918, %3919) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<?x?x?xi64>) -> tensor<?x?x?xi64>
  %3921 = "tf.Maximum"(%3916, %3920) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<?x?x?xi64>) -> tensor<?x?x?xi64>
  %dims3922 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3922 = "tf.Prod"(%3921, %dims3922) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?xi64>
  %3923 = "tf.SelectV2"(%3899, %3911, %3922) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi1>, tensor<?x?x?xi64>, tensor<?x?x?xi64>) -> tensor<?x?x?xi64>
  %dims3924 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3924 = "tf.Min"(%3923, %dims3924) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?xi64>
  %dims3925 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3925 = "tf.Max"(%3924, %dims3925) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?xi64>
  %dims3926 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3926 = "tf.Prod"(%3925, %dims3926) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?xi64>
  %dims3927 = "tf.Const"() { value = dense<[1, 2, 0]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3927 = "tf.Transpose"(%3926, %dims3927) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?xi64>
  %dims3928 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3928 = "tf.Sum"(%3927, %dims3928) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<2xi32>) -> tensor<?xi64>
  %3929 = "tf.Abs"(%3928) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi64>) -> tensor<?xi64>
  %3930 = "tf.Select"(%V__112, %V__37, %V__37) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i1>, tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %3931 = "tf.Sigmoid"(%3930) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %3932 = "tf.Const"() { value = dense<[1, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3933 = "tf.BroadcastTo"(%V__24, %3932) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi32>, tensor<2xi32>) -> tensor<?x?xi32>
  %3934 = "tf.Shape"(%3933) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>) -> tensor<2xi32>
  %3936 = "tf.Sign"(%V__0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  %dims3937 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3937 = "tf.Mean"(%3936, %dims3937) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?xi32>
  %dims3938 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3938 = "tf.Prod"(%3937, %dims3938) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?xi32>
  %3939 = "tf.Abs"(%3938) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  %3940 = "tf.Invert"(%3939) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  %3941 = "tf.Cast"(%3940) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>) -> tensor<?x?x?xi64>
  %dims3942 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3942 = "tf.Min"(%3941, %dims3942) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?xi64>
  %3943 = "tf.Shape"(%3942) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>) -> tensor<3xi32>
  %3944 = "tf.BroadcastTo"(%3931, %3943) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  %3945 = "tf.Sin"(%3944) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %dims3946 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3946 = "tf.Prod"(%3945, %dims3946) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?xf32>
  %dims3947 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3947 = "tf.Min"(%V__39, %dims3947) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %3948 = "tf.Atan"(%3947) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims3949 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3949 = "tf.Max"(%3948, %dims3949) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %dims3950 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3950 = "tf.Min"(%3949, %dims3950) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims3951 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3951 = "tf.Prod"(%3950, %dims3951) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?xf32>
  %3952 = "tf.LeakyRelu"(%3951) { alpha = 3.000000e-01 : f32, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %3953 = "tf.Sinh"(%3952) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %dims3954 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3954 = "tf.Sum"(%3953, %dims3954) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?xf32>
  %3955 = "tf.Relu6"(%3954) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %3956 = "tf.Sinh"(%3955) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %3957 = "tf.Cast"(%V__54) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi1>) -> tensor<?x?x?xi1>
  %dims3958 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3958 = "tf.Any"(%3957, %dims3958) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi1>, tensor<3xi32>) -> tensor<?x?x?xi1>
  %3959 = "tf.Cast"(%3958) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi1>) -> tensor<?x?x?xf32>
  %dims3960 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3960 = "tf.Sum"(%3959, %dims3960) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?xf32>
  %3961 = "tf.LeakyRelu"(%V__71) { alpha = 3.000000e-01 : f32, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %3962 = "tf.Neg"(%3961) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %dims3963 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3963 = "tf.Max"(%3962, %dims3963) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?xf32>
  %3964 = "tf.Sub"(%3960, %3963) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %3965 = "tf.Pow"(%3956, %3964) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %3966 = "tf.Xlogy"(%3946, %3965) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %dims3967 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3967 = "tf.Max"(%3966, %dims3967) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?xf32>
  %3968 = "tf.Squeeze"(%3967) { squeeze_dims = [ 2 : i64 ], device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?xf32>
  %3969 = "tf.Shape"(%3968) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>) -> tensor<2xi32>
  %3970 = "tf.BroadcastTo"(%3929, %3969) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi64>, tensor<2xi32>) -> tensor<?x?xi64>
  %dims3971 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3971 = "tf.Max"(%3970, %dims3971) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi64>, tensor<1xi32>) -> tensor<?xi64>
  %3972 = "tf.Cast"(%3971) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi64>) -> tensor<?xi32>
  %3973 = "tf.BiasAdd"(%3888, %3972) { data_format = "NHWC", device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<?xi32>) -> tensor<?x?x?x?xi32>
  %dims3974 = "tf.Const"() { value = dense<[0, 1, 3, 2]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3974 = "tf.Transpose"(%3973, %dims3974) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims3975 = "tf.Const"() { value = dense<[2, 3, 1, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3975 = "tf.Transpose"(%3974, %dims3975) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims3976 = "tf.Const"() { value = dense<[3, 1, 2, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3976 = "tf.Transpose"(%3975, %dims3976) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %3977 = "tf.Cast"(%3976) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %3978 = "tf.Const"() { value = dense<[1, 1, 34, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3979 = "tf.BroadcastTo"(%3977, %3978) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims3980 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3980 = "tf.Mean"(%3979, %dims3980) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?x?xi32>
  %dims3981 = "tf.Const"() { value = dense<[2, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3981 = "tf.Max"(%3980, %dims3981) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?x?xi32>
  %dims3982 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3982 = "tf.Min"(%3981, %dims3982) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?x?xi32>
  %3983 = "tf.Shape"(%3982) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<4xi32>
  %3984 = "tf.BroadcastTo"(%3808, %3983) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %dims3985 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3985 = "tf.Max"(%3984, %dims3985) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>, tensor<2xi32>) -> tensor<?x?xi32>
  %3986 = "tf.Const"() { value = dense<[1, 1, 84, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3987 = "tf.BroadcastTo"(%3985, %3986) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi32>
  %3988 = "tf.Shape"(%3987) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi32>) -> tensor<4xi32>
  %3989 = "tf.BroadcastTo"(%3467, %3988) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims3990 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3990 = "tf.Prod"(%3989, %dims3990) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %dims3991 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3991 = "tf.Mean"(%3990, %dims3991) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims3992 = "tf.Const"() { value = dense<[0, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %3992 = "tf.Max"(%3991, %dims3992) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %dims3993 = "tf.Const"() { value = dense<[0, 2, 3, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3993 = "tf.Transpose"(%3992, %dims3993) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %3994 = "tf.Square"(%3993) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims3995 = "tf.Const"() { value = dense<[1, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3995 = "tf.Min"(%3994, %dims3995) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %3996 = "tf.Select"(%2111, %3166, %3995) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi1>, tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  func.return %3996 : tensor<?x?x?x?xi64>
}