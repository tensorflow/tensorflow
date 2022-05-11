// RUN: tfg-transforms-opt -split-input-file -shape-inference=graph-version=1010 %s | FileCheck %s

module  {
  tfg.graph #tf_type.version<producer = 1010, min_consumer = 0> {
    %Const, %ctl = Const name("c1") {dtype = f32, value = dense<1.000000e+00> : tensor<1xf32>} : () -> (tensor<1xf32>)
    %Const_0, %ctl_1 = Const name("c2") {dtype = f32, value = dense<2.000000e+00> : tensor<2xf32>} : () -> (tensor<2xf32>)
    %Const_2, %ctl_3 = Const name("c3") {dtype = f32, value = dense<3.000000e+00> : tensor<2xf32>} : () -> (tensor<2xf32>)
    %Placeholder, %ctl_4 = Placeholder name("x") {dtype = f32, shape = #tf_type.shape<2x2>} : () -> (tensor<2x2xf32>)
    // CHECK: Add{{.*}} name("add_child") {{.*}} -> (tensor<2x2xf32>)
    %Add, %ctl_5 = Add(%Const_0, %Placeholder) name("add_child") {T = f32} : (tensor<2xf32>, tensor<2x2xf32>) -> (tensor<*xf32>)
    // CHECK: Add{{.*}} name("add_parent") {{.*}} -> (tensor<2x2xf32>)
    %Add_6, %ctl_7 = Add(%Const, %Add) name("add_parent") {T = f32} : (tensor<1xf32>, tensor<*xf32>) -> (tensor<*xf32>)
    %Const_8, %ctl_9 = Const name("c4") {dtype = f32, value = dense<4.000000e+00> : tensor<2xf32>} : () -> (tensor<2xf32>)
    %Const_10, %ctl_11 = Const name("c5") {dtype = f32, value = dense<5.000000e+00> : tensor<2xf32>} : () -> (tensor<2xf32>)
    %Const_12, %ctl_13 = Const name("c20") {dtype = f32, value = dense<2.000000e+01> : tensor<2xf32>} : () -> (tensor<2xf32>)
    %Placeholder_14, %ctl_15 = Placeholder name("y") {dtype = f32, shape = #tf_type.shape<2x2>} : () -> (tensor<2x2xf32>)
    // CHECK: Mul{{.*}} name("mul_child") {{.*}} -> (tensor<2x2xf32>)
    %Mul, %ctl_16 = Mul(%Const_8, %Placeholder_14) name("mul_child") {T = f32} : (tensor<2xf32>, tensor<2x2xf32>) -> (tensor<*xf32>)
    // CHECK: Mul{{.*}} name("mul_parent") {{.*}} -> (tensor<2x2xf32>)
    %Mul_17, %ctl_18 = Mul(%Const_10, %Mul) name("mul_parent") {T = f32} : (tensor<2xf32>, tensor<*xf32>) -> (tensor<*xf32>)
    // CHECK: Add{{.*}} name("addmul_child") {{.*}} -> (tensor<2x2xf32>)
    %Add_19, %ctl_20 = Add(%Const_8, %Placeholder) name("addmul_child") {T = f32} : (tensor<2xf32>, tensor<2x2xf32>) -> (tensor<*xf32>)
    // CHECK: Mul{{.*}} name("addmul_parent") {{.*}} -> (tensor<2x2xf32>)
    %Mul_21, %ctl_22 = Mul(%Const_10, %Add_19) name("addmul_parent") {T = f32} : (tensor<2xf32>, tensor<*xf32>) -> (tensor<*xf32>)
  }
}

// -----

module {
  tfg.graph #tf_type.version<producer = 1010, min_consumer = 0> {
    %Const, %ctl = Const device("/CPU:0") name("Const/Const") {dtype = i32, value = dense<[10, 1]> : tensor<2xi32>} : () -> (tensor<2xi32>)
    // CHECK: RandomStandardNormal{{.*}} -> ([[TENSOR:.*]])
    %RandomStandardNormal, %ctl_0 = RandomStandardNormal(%Const) device("/CPU:0") name("x") {T = i32, dtype = f32, seed = 0 : i64, seed2 = 0 : i64} : (tensor<2xi32>) -> (tensor<*xf32>)
    // CHECK: name("Sign") {{.*}} : ([[TENSOR]]) -> ([[TENSOR]])
    %Sign, %ctl_1 = Sign(%RandomStandardNormal) device("/job:localhost/replica:0/task:0/device:CPU:0") name("Sign") {T = f32} : (tensor<*xf32>) -> (tensor<*xf32>)
    // CHECK: name("Sign_1") {{.*}} : ([[TENSOR]]) -> ([[TENSOR]])
    %Sign_2, %ctl_3 = Sign(%Sign) device("/job:localhost/replica:0/task:0/device:CPU:0") name("Sign_1") {T = f32} : (tensor<*xf32>) -> (tensor<*xf32>)
    // CHECK: name("Sign_2") {{.*}} : ([[TENSOR]]) -> ([[TENSOR]])
    %Sign_4, %ctl_5 = Sign(%Sign_2) device("/job:localhost/replica:0/task:0/device:CPU:0") name("Sign_2") {T = f32} : (tensor<*xf32>) -> (tensor<*xf32>)
    // CHECK: name("Sign_3") {{.*}} : ([[TENSOR]]) -> ([[TENSOR]])
    %Sign_6, %ctl_7 = Sign(%Sign_4) device("/job:localhost/replica:0/task:0/device:CPU:0") name("Sign_3") {T = f32} : (tensor<*xf32>) -> (tensor<*xf32>)
    // CHECK: AddN{{.*}} name("y") {{.*}} : ([[TENSOR]]) -> ([[TENSOR]])
    %AddN, %ctl_8 = AddN(%Sign_6) device("/CPU:0") name("y") {N = 1 : i64, T = f32} : (tensor<*xf32>) -> (tensor<*xf32>)
  }
}

// -----

module {
  tfg.graph #tf_type.version<producer = 1010, min_consumer = 0> {
    // CHECK: Variable name("Var") {{.*}} #tf_type.shape<[[VAR_SHAPE:.*]]>, {{.*}} : () -> (tensor<[[VAR_SHAPE]]x!tf_type.f32ref>)
    %Variable, %ctl = Variable name("Var") {container = "", dtype = f32, shape = #tf_type.shape<3x7>, shared_name = ""} : () -> (tensor<*x!tf_type.f32ref>)
  }
}

// -----

module {
  tfg.graph #tf_type.version<producer = 0, min_consumer = 0> {
    // CHECK: VarHandleOp name("Var") {{.*}} #tf_type.shape<[[VAR_HANDLE_SHAPE:.*]]>, {{.*}} -> (tensor<!tf_type.resource<tensor<[[VAR_HANDLE_SHAPE]]
    %VarHandleOp, %ctl = VarHandleOp name("Var") {allowed_devices = [], container = "", dtype = f32, shape = #tf_type.shape<3x7>, shared_name = ""} : () -> (tensor<!tf_type.resource>)
    // CHECK: ReadVariableOp{{.*}} name("VarRead") {{.*}} -> (tensor<[[VAR_HANDLE_SHAPE]]xf32>)
    %ReadVariableOp, %ctl_0 = ReadVariableOp(%VarHandleOp) name("VarRead") {dtype = f32} : (tensor<!tf_type.resource>) -> (tensor<*xf32>)
  }
}

// -----

module {
  tfg.graph #tf_type.version<producer = 1070, min_consumer = 0> {
    %Const, %ctl = Const name("a") {dtype = i32, value = dense<[5, 7]> : tensor<2xi32>} : () -> (tensor<2xi32>)
    // CHECK: Identity{{.*}} name("a1") {{.*}} -> (tensor<2xi32>)
    %Identity, %ctl_0 = Identity(%Const) name("a1") {T = i32} : (tensor<2xi32>) -> (tensor<*xi32>)
    %Const_1, %ctl_2 = Const name("b") {dtype = i32, value = dense<99> : tensor<i32>} : () -> (tensor<i32>)
    // CHECK: Identity{{.*}} name("b1") {{.*}} -> (tensor<i32>)
    %Identity_3, %ctl_4 = Identity(%Const_1) name("b1") {T = i32} : (tensor<i32>) -> (tensor<*xi32>)
    %Const_5, %ctl_6 = Const name("c") {dtype = i32, value = dense<1> : tensor<4x4x4xi32>} : () -> (tensor<4x4x4xi32>)
    // CHECK: Identity{{.*}} name("c1") {{.*}} -> (tensor<4x4x4xi32>)
    %Identity_7, %ctl_8 = Identity(%Const_5) name("c1") {T = i32} : (tensor<4x4x4xi32>) -> (tensor<*xi32>)
  }
}

// -----

module {
  tfg.graph #tf_type.version<producer = 1070, min_consumer = 0> {
    // CHECK: Const name("identity_a") {{.*}} : () -> (tensor<[[CONST_DIM:.*]]xi32>)
    %Const, %ctl = Const name("identity_a") {dtype = i32, value = dense<5> : tensor<2xi32>} : () -> (tensor<2xi32>)
    // CHECK: Identity{{.*}} name("identity_b") {{.*}} -> (tensor<[[CONST_DIM]]xi32>)
    %Identity, %ctl_0 = Identity(%Const) name("identity_b") {T = i32} : (tensor<2xi32>) -> (tensor<*xi32>)
    %Const_1, %ctl_2 = Const name("const") {dtype = f32, value = dense<1.000000e-01> : tensor<f32>} : () -> (tensor<f32>)
    // CHECK: Fill{{.*}} name("fill") {{.*}} -> (tensor<5x5xf32>)
    %Fill, %ctl_3 = Fill(%Identity, %Const_1) name("fill") {T = f32, index_type = i32} : (tensor<*xi32>, tensor<f32>) -> (tensor<*xf32>)
    // CHECK: IdentityN{{.*}} name("identityn_a") {{.*}} -> (tensor<[[CONST_DIM]]xi32>)
    %IdentityN, %ctl_4 = IdentityN(%Const) name("identityn_a") {T = [i32]} : (tensor<2xi32>) -> (tensor<*xi32>)
    // CHECK: Fill{{.*}} name("fill_identityn_a") {{.*}} -> (tensor<5x5xf32>)
    %Fill_1, %ctl_5 = Fill(%IdentityN, %Const_1) name("fill_identityn_a") {T = f32, index_type = i32} : (tensor<*xi32>, tensor<f32>) -> (tensor<*xf32>)
    // CHECK: IdentityN{{.*}} name("identityn_b") {{.*}} -> (tensor<[[CONST_DIM]]xi32>, tensor<f32>)
    %IdentityN_1:2, %ctl_6 = IdentityN(%Const, %Const_1) name("identityn_b") {T = [i32, f32]} : (tensor<2xi32>, tensor<f32>) -> (tensor<*xi32>, tensor<f32>)
    // CHECK: Fill{{.*}} name("fill_identityn_b") {{.*}} -> (tensor<?x?xf32>)
    %Fill_2, %ctl_7 = Fill(%IdentityN_1#0, %Const_1) name("fill_identityn_b") {T = f32, index_type = i32} : (tensor<*xi32>, tensor<f32>) -> (tensor<*xf32>)
  }
}

// -----

module {
  tfg.graph #tf_type.version<producer = 1070, min_consumer = 0> {
    %Const, %ctl = Const name("Const") {dtype = i32, value = dense<1> : tensor<4x4x4xi32>} : () -> (tensor<4x4x4xi32>)
    // CHECK: Rank{{.*}} name("Rank") {{.*}} -> (tensor<i32>)
    %Rank, %ctl_0 = Rank(%Const) name("Rank") {T = i32} : (tensor<4x4x4xi32>) -> (tensor<*xi32>)
    // CHECK: Identity{{.*}} name("Identity_Rank") {{.*}} -> (tensor<i32>)
    %Identity, %ctl_1 = Identity(%Rank) name("Identity_Rank") {T = i32} : (tensor<*xi32>) -> (tensor<*xi32>)
    %PlaceHolder, %ctl_2 = Placeholder name("placeholder_with_rank") : () -> (tensor<?x?xi32>)
    // CHECK: Rank{{.*}} name("rank_on_placeholder") {{.*}} -> (tensor<i32>)
    %Rank_1, %ctl_3 = Rank(%PlaceHolder) name("rank_on_placeholder") {T = i32} : (tensor<?x?xi32>) -> (tensor<*xi32>)
  }
}

// -----

module {
  tfg.graph #tf_type.version<producer = 1070, min_consumer = 0> {
    %Const, %ctl = Const name("Const") {dtype = i32, value = dense<1> : tensor<1x2x3x4xi32>} : () -> (tensor<1x2x3x4xi32>)
    // CHECK: Size{{.*}} name("Size") {{.*}} -> (tensor<i32>)
    %Size, %ctl_0 = Size(%Const) name("Size") {T = i32, out_type = i32} : (tensor<1x2x3x4xi32>) -> (tensor<*xi32>)
    // CHECK: Identity{{.*}} name("Identity_Size") {{.*}} -> (tensor<i32>)
    %Identity, %ctl_1 = Identity(%Size) name("Identity_Size") {T = i32} : (tensor<*xi32>) -> (tensor<*xi32>)
  }
}

// -----

module {
  tfg.graph #tf_type.version<producer = 1070, min_consumer = 0> {
    %Const, %ctl = Const name("Const") {dtype = i32, value = dense<1> : tensor<1x2x3x4xi32>} : () -> (tensor<1x2x3x4xi32>)
    // CHECK: Shape{{.*}} name("Shape_32") {{.*}} -> (tensor<4xi32>)
    %Size, %ctl_0 = Shape(%Const) name("Shape_32") {T = i32, out_type = i32} : (tensor<1x2x3x4xi32>) -> (tensor<2x?xi32>)
    // CHECK: Shape{{.*}} name("Shape_64") {{.*}} -> (tensor<4xi64>)
    %Shape_64, %ctl_1 = Shape(%Const, %ctl, %ctl_0) name("Shape_64") {T = i64, out_type = i64} : (tensor<1x2x3x4xi32>, !tf_type.control, !tf_type.control) -> (tensor<*xi64>)
  }
}

// -----

module {
  tfg.func @update_function_arg_return_type(%arg0 : tensor<*xi32> {tfg.name = "input", tf._output_shapes = [#tf_type.shape<2x3>]}, %arg1 : tensor<*xf32> {tfg.name = "another_input"})
      -> (tensor<*xi32> {tfg.name = "result1"}) {
    %Const, %ctl = Const name("Const") {dtype = i32, value = dense<1> : tensor<1x2x3x4xi32>} : () -> (tensor<*xi32>)
    %Size, %ctl_0 = Size(%Const) name("Size") {T = i32, out_type = i32} : (tensor<*xi32>) -> (tensor<*xi32>)
    // CHECK: Shape{{.*}} name("Shape_1") {{.*}} -> (tensor<2xi64>)
    %Shape_1, %ctl_1 = Shape(%arg0) name("Shape_1") {T = i64, out_type = i64} : (tensor<*xi32>) -> (tensor<*xi64>)
    return(%Size) : tensor<*xi32>
  }
}

// -----

module {
  tfg.func @cant_infer_shape(%arg0 : tensor<*xf32> {tfg.name = "input"}, %arg1 : tensor<*xf32> {tfg.name = "another_input"})
      -> (tensor<*xf32> {tfg.name = "result1"}) {
    // CHECK: Add{{.*}} name("unranked_add") {{.*}} (tensor<*xf32>)
    %add, %ctl = Add(%arg0, %arg1) name("unranked_add") : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>)
    %add_1, %ctl_1 = Add(%arg0, %arg1) name("unranked_add2") : (tensor<*xf32>, tensor<*xf32>) -> (tensor<?x2xf32>)
    return(%add) : tensor<*xf32>
  }
}

// -----

module {
  tfg.func generic @cant_infer_opaque_tensor(%x: !tf_type.tensor {tfg.name = "x", tfg.type_attr = "T"})
       -> (!tf_type.tensor {tfg.name = "y", tfg.type_attr = "T"})
   attributes {is_stateful, tf._noinline = true, tfg.func_attrs = {T = {allowed_values = [f32, f64, i32, i64], function_type = "type"}}} {
    %XTimesFour, %ctl = XTimesFour(%x) name("x4") {T = #tf_type.placeholder<"T">} : (!tf_type.tensor) -> (!tf_type.tensor)
    %0 = get_result(%XTimesFour) "y" : 0
    %XTimesFour_0, %ctl_1 = XTimesFour(%0) name("y") {T = #tf_type.placeholder<"T">} : (!tf_type.tensor) -> (!tf_type.tensor)
    %1 = get_result(%XTimesFour_0) "y" : 0
    return(%1) : !tf_type.tensor
  }
}
