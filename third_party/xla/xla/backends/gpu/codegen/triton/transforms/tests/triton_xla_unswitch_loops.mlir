// RUN: xla-opt %s --triton-xla-unswitch-loops | FileCheck %s

// CHECK-LABEL: func @single_if
func.func @single_if(%arg0: tensor<2xf32>, %arg1: index) -> tensor<2xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %cst3 = arith.constant 3.0 : f32
    %cst4 = arith.constant 4.0 : f32
    %cond = arith.cmpi sle, %arg1, %c1 : index
    %for = scf.for %i = %c0 to %c2 step %c1 iter_args(%arg2 = %arg0) -> tensor<2xf32> {
      %result = scf.if %cond -> tensor<2xf32> {
        %set_3 = tensor.insert %cst3 into %arg2[%i] : tensor<2xf32>
        scf.yield %set_3 : tensor<2xf32>
      } else {
        %set_4 = tensor.insert %cst4 into %arg2[%i] : tensor<2xf32>
        scf.yield %set_4 : tensor<2xf32>
      }
      scf.yield %result : tensor<2xf32>
    }
    func.return %for : tensor<2xf32>
    // CHECK: scf.if
    // CHECK: scf.for
    // CHECK: scf.for
}

// CHECK-LABEL: func @iter_arg_in_if_cond_unmodified
func.func @iter_arg_in_if_cond_unmodified(%arg0: tensor<2xf32>, %arg1: tensor<2xi1>) -> tensor<2xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %cst3 = arith.constant 3.0 : f32
    %cst4 = arith.constant 4.0 : f32
    %for = scf.for %i = %c0 to %c2 step %c1 iter_args(%arg2 = %arg0) -> tensor<2xf32> {
      %cond = tensor.extract %arg1[%i] : tensor<2xi1>
      // We can't unswitch this loop because the iter arg is used in
      // the condition of the if.
      %result = scf.if %cond -> tensor<2xf32> {
        %set_3 = tensor.insert %cst3 into %arg2[%i] : tensor<2xf32>
        scf.yield %set_3 : tensor<2xf32>
      } else {
        %set_4 = tensor.insert %cst4 into %arg2[%i] : tensor<2xf32>
        scf.yield %set_4 : tensor<2xf32>
      }
      scf.yield %result : tensor<2xf32>
    }
    func.return %for : tensor<2xf32>
    // CHECK: scf.for
    // CHECK-NEXT: tensor.extract
    // CHECK-NEXT: scf.if
}

// CHECK-LABEL: func @outer_loop_condition
func.func @outer_loop_condition(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %cst10 = arith.constant 10.0 : f32
    %cst20 = arith.constant 20.0 : f32

    %outer = scf.for %i = %c0 to %c2 step %c1 iter_args(%iter_arg0 = %arg0) -> tensor<2x2xf32> {
      %cond = arith.cmpi eq, %i, %c1 : index
      %inner = scf.for %j = %c0 to %c2 step %c1 iter_args(%iter_arg1 = %iter_arg0) -> tensor<2x2xf32> {
        %current_val = tensor.extract %iter_arg1[%i, %j] : tensor<2x2xf32>
        %if_res = scf.if %cond -> tensor<2x2xf32> {
          %added = arith.addf %current_val, %cst10 : f32
          %set_10 = tensor.insert %added into %iter_arg1[%i, %j] : tensor<2x2xf32>
          scf.yield %set_10 : tensor<2x2xf32>
        } else {
          %subbed = arith.subf %current_val, %cst20 : f32
          %set_20 = tensor.insert %subbed into %iter_arg1[%i, %j] : tensor<2x2xf32>
          scf.yield %set_20 : tensor<2x2xf32>
        }
        scf.yield %if_res : tensor<2x2xf32>
      }
      scf.yield %inner : tensor<2x2xf32>
    }
    func.return %outer : tensor<2x2xf32>
    // CHECK: scf.for
    // CHECK-NEXT: arith.cmpi eq
    // CHECK-NEXT: scf.if
    // CHECK-NEXT: scf.for
    // CHECK: else
    // CHECK-NEXT: scf.for
}

// CHECK-LABEL: func @ops_before_and_after_if
func.func @ops_before_and_after_if(%arg0: tensor<2xf32>, %arg1: index) -> tensor<2xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %cst4 = arith.constant 4.0 : f32
    %cst5 = arith.constant 5.0 : f32
    %cond = arith.cmpi sle, %arg1, %c1 : index
    %for = scf.for %i = %c0 to %c2 step %c1 iter_args(%arg2 = %arg0) -> tensor<2xf32> {
      %extra_val = tensor.extract %arg2[%i] : tensor<2xf32>
      %before_if = arith.addf %extra_val, %cst5 : f32
      %if_res = scf.if %cond -> tensor<2xf32> {
        %set_3 = tensor.insert %before_if into %arg2[%i] : tensor<2xf32>
        scf.yield %set_3 : tensor<2xf32>
      } else {
        %set_4 = tensor.insert %cst4 into %arg2[%i] : tensor<2xf32>
        scf.yield %set_4 : tensor<2xf32>
      }
      %after_if = arith.addf %if_res, %if_res : tensor<2xf32>
      %set_5 = tensor.insert %cst5 into %after_if[%i] : tensor<2xf32>
      scf.yield %set_5 : tensor<2xf32>
    }
    func.return %for : tensor<2xf32>
    // CHECK: scf.if
    // CHECK: scf.for
    // CHECK-NEXT: tensor.extract
    // CHECK: arith.addf
    // CHECK: arith.addf
    // CHECK: scf.for
    // CHECK-NEXT: tensor.insert
    // CHECK: arith.addf
}

// CHECK-LABEL: func @loop_var_in_if_cond_unmodified
func.func @loop_var_in_if_cond_unmodified(%arg0: tensor<2xf32>) -> tensor<2xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %cst3 = arith.constant 3.0 : f32
    %cst4 = arith.constant 4.0 : f32
    %for = scf.for %i = %c0 to %c2 step %c1 iter_args(%arg2 = %arg0) -> tensor<2xf32> {
      %cond = arith.cmpi sle, %i, %c1 : index
      // We can't unswitch this loop because the induction variable is used in
      // the condition of the if.
      %result = scf.if %cond -> tensor<2xf32> {
        %set_3 = tensor.insert %cst3 into %arg2[%i] : tensor<2xf32>
        scf.yield %set_3 : tensor<2xf32>
      } else {
        %set_4 = tensor.insert %cst4 into %arg2[%i] : tensor<2xf32>
        scf.yield %set_4 : tensor<2xf32>
      }
      scf.yield %result : tensor<2xf32>
    }
    func.return %for : tensor<2xf32>
    // CHECK: scf.for
    // CHECK-NEXT: arith.cmpi sle
    // CHECK-NEXT: scf.if
}

// CHECK-LABEL: func @nested_loops
func.func @nested_loops(%arg0: tensor<2x2xf32>, %cond: i1) -> tensor<2x2xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %cst10 = arith.constant 10.0 : f32
    %cst20 = arith.constant 20.0 : f32

    %outer = scf.for %i = %c0 to %c2 step %c1 iter_args(%iter_arg0 = %arg0) -> tensor<2x2xf32> {
      %inner = scf.for %j = %c0 to %c2 step %c1 iter_args(%iter_arg1 = %iter_arg0) -> tensor<2x2xf32> {
        %current_val = tensor.extract %iter_arg1[%i, %j] : tensor<2x2xf32>
        %result = scf.if %cond -> tensor<2x2xf32> {
          %added = arith.addf %current_val, %cst10 : f32
          %set_10 = tensor.insert %added into %iter_arg1[%i, %j] : tensor<2x2xf32>
          scf.yield %set_10 : tensor<2x2xf32>
        } else {
          %subbed = arith.subf %current_val, %cst20 : f32
          %set_20 = tensor.insert %subbed into %iter_arg1[%i, %j] : tensor<2x2xf32>
          scf.yield %set_20 : tensor<2x2xf32>
        }
        scf.yield %result : tensor<2x2xf32>
      }
      scf.yield %inner : tensor<2x2xf32>
    }
    func.return %outer : tensor<2x2xf32>
    // CHECK: scf.if
    // CHECK: scf.for
    // CHECK: scf.for
    // CHECK: else
    // CHECK: scf.for
    // CHECK: scf.for
}

// CHECK-LABEL: func @two_ifs_different_conditions_unmodified
func.func @two_ifs_different_conditions_unmodified(%arg0: tensor<2xf32>, %cond1: i1, %cond2: i1) -> tensor<2xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %cst3 = arith.constant 3.0 : f32
    %cst4 = arith.constant 4.0 : f32
    %cst5 = arith.constant 5.0 : f32
    %cst6 = arith.constant 6.0 : f32

    %for = scf.for %i = %c0 to %c2 step %c1 iter_args(%arg2 = %arg0) -> tensor<2xf32> {
      %if1_res = scf.if %cond1 -> tensor<2xf32> {
        %set_3 = tensor.insert %cst3 into %arg2[%i] : tensor<2xf32>
        scf.yield %set_3 : tensor<2xf32>
      } else {
        %set_4 = tensor.insert %cst4 into %arg2[%i] : tensor<2xf32>
        scf.yield %set_4 : tensor<2xf32>
      }
      %if2_res = scf.if %cond2 -> tensor<2xf32> {
        %set_5 = tensor.insert %cst5 into %if1_res[%i] : tensor<2xf32>
        scf.yield %set_5 : tensor<2xf32>
      } else {
        %set_6 = tensor.insert %cst6 into %if1_res[%i] : tensor<2xf32>
        scf.yield %set_6 : tensor<2xf32>
      }
      scf.yield %if2_res : tensor<2xf32>
    }
    func.return %for : tensor<2xf32>
    // CHECK: scf.for
    // CHECK: scf.if
    // CHECK: scf.if
}


// CHECK-LABEL: func @two_ifs_same_condition
func.func @two_ifs_same_condition(%arg0: tensor<2xf32>, %cond1: i1) -> tensor<2xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %cst3 = arith.constant 3.0 : f32
    %cst4 = arith.constant 4.0 : f32
    %cst5 = arith.constant 5.0 : f32
    %cst6 = arith.constant 6.0 : f32

    %for = scf.for %i = %c0 to %c2 step %c1 iter_args(%arg2 = %arg0) -> tensor<2xf32> {
      %if1_res = scf.if %cond1 -> tensor<2xf32> {
        %set_3 = tensor.insert %cst3 into %arg2[%i] : tensor<2xf32>
        scf.yield %set_3 : tensor<2xf32>
      } else {
        %set_4 = tensor.insert %cst4 into %arg2[%i] : tensor<2xf32>
        scf.yield %set_4 : tensor<2xf32>
      }
      %if2_res = scf.if %cond1 -> tensor<2xf32> {
        %set_5 = tensor.insert %cst5 into %if1_res[%i] : tensor<2xf32>
        scf.yield %set_5 : tensor<2xf32>
      } else {
        %set_6 = tensor.insert %cst6 into %if1_res[%i] : tensor<2xf32>
        scf.yield %set_6 : tensor<2xf32>
      }
      scf.yield %if2_res : tensor<2xf32>
    }
    func.return %for : tensor<2xf32>
    // CHECK: scf.if
    // CHECK: scf.for
    // CHECK: else
    // CHECK: scf.for
}
