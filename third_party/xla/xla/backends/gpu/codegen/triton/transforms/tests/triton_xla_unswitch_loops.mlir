// RUN: xla-opt %s --triton-xla-unswitch-loops | FileCheck %s

// CHECK-LABEL: func @single_if
func.func @single_if(%arg0: i32, %cond: i1) -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %for = scf.for %i = %c0 to %c2 step %c1 iter_args(%iter = %arg0) -> i32 {
      %result = scf.if %cond -> i32 {
        %set_3 = arith.addi %iter, %arg0: i32
        scf.yield %set_3 : i32
      } else {
        %set_4 = arith.subi %iter, %arg0: i32
        scf.yield %set_4 : i32
      }
      scf.yield %result : i32
    }
    func.return %for : i32
    // CHECK: scf.if
    // CHECK: scf.for
    // CHECK: scf.for
}

// CHECK-LABEL: func @for_if_for_if
func.func @for_if_for_if(%arg0: i32, %condA: i1, %condB: i1) -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %cst10 = arith.constant 10 : i32
    %cst20 = arith.constant 20 : i32
    %cst30 = arith.constant 30 : i32
    %cst40 = arith.constant 40 : i32

    %for1 = scf.for %i = %c0 to %c2 step %c1 iter_args(%iter1 = %arg0) -> i32 {
      %if1_res = scf.if %condA -> i32 {
        %add1 = arith.addi %iter1, %cst10 : i32
        scf.yield %add1 : i32
      } else {
        %sub1 = arith.subi %iter1, %cst20 : i32
        scf.yield %sub1 : i32
      }

      %for2 = scf.for %j = %c0 to %c2 step %c1 iter_args(%iter2 = %if1_res) -> i32 {
        %if2_res = scf.if %condB -> i32 {
          %add2 = arith.addi %iter2, %cst30 : i32
          scf.yield %add2 : i32
        } else {
          %sub2 = arith.subi %iter2, %cst40 : i32
          scf.yield %sub2 : i32
        }
        scf.yield %if2_res : i32
      }
      scf.yield %for2 : i32
    }
    func.return %for1 : i32
    // CHECK: scf.if
    // CHECK: scf.for
    // CHECK: scf.if
    // CHECK: scf.for
}

// CHECK-LABEL: func @iter_arg_in_if_cond_unmodified
func.func @iter_arg_in_if_cond_unmodified(%arg0: i32, %arg1: i32) -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %cst3 = arith.constant 3 : i32
    %cst4 = arith.constant 4 : i32
    %for = scf.for %i = %c0 to %c2 step %c1 iter_args(%iter = %arg0) -> i32 {
      // The condition depends on the %iter, so we can't unswitch.
      %cond = arith.cmpi eq, %iter, %arg1 : i32
      %result = scf.if %cond -> i32 {
        %set_3 = arith.addi %iter, %cst3 : i32
        scf.yield %set_3 : i32
      } else {
        %set_4 = arith.subi %iter, %cst4 : i32
        scf.yield %set_4 : i32
      }
      scf.yield %result : i32
    }
    func.return %for : i32
    // CHECK: scf.for
    // CHECK-NEXT: arith.cmpi eq
    // CHECK-NEXT: scf.if
}

// CHECK-LABEL: func @outer_loop_condition
func.func @outer_loop_condition(%arg0: i32) -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %cst10 = arith.constant 10 : i32
    %cst20 = arith.constant 20 : i32

    %outer = scf.for %i = %c0 to %c2 step %c1 iter_args(%iter_arg0 = %arg0) -> i32 {
      %cond = arith.cmpi eq, %i, %c1 : index
      %inner = scf.for %j = %c0 to %c2 step %c1 iter_args(%iter_arg1 = %iter_arg0) -> i32 {
        %if_res = scf.if %cond -> i32 {
          %added = arith.addi %iter_arg1, %cst10 : i32
          scf.yield %added : i32
        } else {
          %subbed = arith.subi %iter_arg1, %cst20 : i32
          scf.yield %subbed : i32
        }
        scf.yield %if_res : i32
      }
      scf.yield %inner : i32
    }
    func.return %outer : i32
    // CHECK: scf.for
    // CHECK-NEXT: arith.cmpi eq
    // CHECK-NEXT: scf.if
    // CHECK-NEXT: scf.for
    // CHECK: else
    // CHECK-NEXT: scf.for
}

// CHECK-LABEL: func @ops_before_and_after_if
func.func @ops_before_and_after_if(%arg0: i32, %arg1: i32) -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %cst4 = arith.constant 4 : i32
    %cst5 = arith.constant 5 : i32
    %cond = arith.cmpi sle, %arg1, %cst4 : i32
    %for = scf.for %i = %c0 to %c2 step %c1 iter_args(%iter = %arg0) -> i32 {
      %before_if = arith.addi %iter, %cst5 : i32
      %if_res = scf.if %cond -> i32 {
        %set_3 = arith.addi %before_if, %cst4 : i32
        scf.yield %set_3 : i32
      } else {
        %set_4 = arith.subi %before_if, %cst4 : i32
        scf.yield %set_4 : i32
      }
      %after_if = arith.addi %if_res, %if_res : i32
      scf.yield %after_if : i32
    }
    func.return %for : i32
    // CHECK: scf.if
    // CHECK: scf.for
    // CHECK-NEXT: arith.addi
    // CHECK: arith.addi
    // CHECK: arith.addi
    // CHECK: else
    // CHECK: scf.for
    // CHECK-NEXT: arith.addi
    // CHECK: arith.subi
    // CHECK: arith.addi
}

// CHECK-LABEL: func @loop_var_in_if_cond_unmodified
func.func @loop_var_in_if_cond_unmodified(%arg0: i32) -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %cst3 = arith.constant 3 : i32
    %cst4 = arith.constant 4 : i32
    %for = scf.for %i = %c0 to %c2 step %c1 iter_args(%iter = %arg0) -> i32 {
      %cond = arith.cmpi sle, %i, %c1 : index
      // We can't unswitch this loop because the induction variable is used in
      // the condition of the if.
      %result = scf.if %cond -> i32 {
        %set_3 = arith.addi %iter, %cst3 : i32
        scf.yield %set_3 : i32
      } else {
        %set_4 = arith.subi %iter, %cst4 : i32
        scf.yield %set_4 : i32
      }
      scf.yield %result : i32
    }
    func.return %for : i32
    // CHECK: scf.for
    // CHECK-NEXT: arith.cmpi sle
    // CHECK-NEXT: scf.if
}

// CHECK-LABEL: func @nested_loops
func.func @nested_loops(%arg0: i32, %cond: i1) -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %cst10 = arith.constant 10 : i32
    %cst20 = arith.constant 20 : i32

    %outer = scf.for %i = %c0 to %c2 step %c1 iter_args(%iter_arg0 = %arg0) -> i32 {
      %inner = scf.for %j = %c0 to %c2 step %c1 iter_args(%iter_arg1 = %iter_arg0) -> i32 {
        %result = scf.if %cond -> i32 {
          %added = arith.addi %iter_arg1, %cst10 : i32
          scf.yield %added : i32
        } else {
          %subbed = arith.subi %iter_arg1, %cst20 : i32
          scf.yield %subbed : i32
        }
        scf.yield %result : i32
      }
      scf.yield %inner : i32
    }
    func.return %outer : i32
    // CHECK: scf.if
    // CHECK: scf.for
    // CHECK: scf.for
    // CHECK: else
    // CHECK: scf.for
    // CHECK: scf.for
}

// CHECK-LABEL: func @for_if_for_if_for_if
// As we only unswitch if there are less than 3 nested loops, we expect that
// outermost loop will be kept.
func.func @for_if_for_if_for_if(%arg0: i32, %condA: i1, %condB: i1, %condC: i1) -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %cst10 = arith.constant 10 : i32
    %cst20 = arith.constant 20 : i32
    %cst30 = arith.constant 30 : i32
    %cst40 = arith.constant 40 : i32
    %cst50 = arith.constant 50 : i32
    %cst60 = arith.constant 60 : i32

    %for1 = scf.for %i = %c0 to %c2 step %c1 iter_args(%iter1 = %arg0) -> i32 {
      %if1_res = scf.if %condA -> i32 {
        %add1 = arith.addi %iter1, %cst10 : i32
        scf.yield %add1 : i32
      } else {
        %sub1 = arith.subi %iter1, %cst20 : i32
        scf.yield %sub1 : i32
      }

      %for2 = scf.for %j = %c0 to %c2 step %c1 iter_args(%iter2 = %if1_res) -> i32 {
        %if2_res = scf.if %condB -> i32 {
          %add2 = arith.addi %iter2, %cst30 : i32
          scf.yield %add2 : i32
        } else {
          %sub2 = arith.subi %iter2, %cst40 : i32
          scf.yield %sub2 : i32
        }

        %for3 = scf.for %k = %c0 to %c2 step %c1 iter_args(%iter3 = %if2_res) -> i32 {
          %if3_res = scf.if %condC -> i32 {
            %add3 = arith.addi %iter3, %cst50 : i32
            scf.yield %add3 : i32
          } else {
            %sub3 = arith.subi %iter3, %cst60 : i32
            scf.yield %sub3 : i32
          }
          scf.yield %if3_res : i32
        }
        scf.yield %for3 : i32
      }
      scf.yield %for2 : i32
    }
    func.return %for1 : i32
    // CHECK-COUNT-9: scf.for
}
