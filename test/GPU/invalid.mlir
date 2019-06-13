// RUN: mlir-opt -split-input-file -verify %s

func @not_enough_sizes(%sz : index) {
  // expected-error@+1 {{expected 6 or more operands}}
  "gpu.launch"(%sz, %sz, %sz, %sz, %sz) ({
    gpu.return
  }) : (index, index, index, index, index) -> ()
  return
}

// -----

func @no_region_attrs(%sz : index) {
  // expected-error@+1 {{unexpected number of region arguments}}
 "gpu.launch"(%sz, %sz, %sz, %sz, %sz, %sz) ({
  ^bb1(%bx: index, %by: index, %bz: index,
       %tx: index, %ty: index, %tz: index):
    gpu.return
  }) : (index, index, index, index, index, index) -> ()
  return
}

// -----

func @isolation_arg(%sz : index) {
 // expected-note@+1 {{required by region isolation constraints}}
 "gpu.launch"(%sz, %sz, %sz, %sz, %sz, %sz) ({
  ^bb1(%bx: index, %by: index, %bz: index,
       %tx: index, %ty: index, %tz: index,
       %szbx: index, %szby: index, %szbz: index,
       %sztx: index, %szty: index, %sztz: index):
    // expected-error@+1 {{using value defined outside the region}}
    "use"(%sz) : (index) -> ()
    gpu.return
  }) : (index, index, index, index, index, index) -> ()
  return
}

// -----

func @isolation_op(%sz : index) {
 %val = "produce"() : () -> (index)
 // expected-note@+1 {{required by region isolation constraints}}
 "gpu.launch"(%sz, %sz, %sz, %sz, %sz, %sz) ({
  ^bb1(%bx: index, %by: index, %bz: index,
       %tx: index, %ty: index, %tz: index,
       %szbx: index, %szby: index, %szbz: index,
       %sztx: index, %szty: index, %sztz: index):
    // expected-error@+1 {{using value defined outside the region}}
    "use"(%val) : (index) -> ()
    gpu.return
  }) : (index, index, index, index, index, index) -> ()
  return
}

// -----

func @nested_isolation(%sz : index) {
  // expected-note@+1 {{required by region isolation constraints}}
  "gpu.launch"(%sz, %sz, %sz, %sz, %sz, %sz) ({
  ^bb1(%bx: index, %by: index, %bz: index,
       %tx: index, %ty: index, %tz: index,
       %szbx: index, %szby: index, %szbz: index,
       %sztx: index, %szty: index, %sztz: index):
    "region"() ({
      "region"() ({
        // expected-error@+1 {{using value defined outside the region}}
        "use"(%sz) : (index) -> ()
      }) : () -> ()
    }) : () -> ()
    gpu.return
  }) : (index, index, index, index, index, index) -> ()
  return
}

// -----

func @launch_requires_gpu_return(%sz : index) {
  // @expected-note@+1 {{in 'gpu.launch' body region}}
  gpu.launch blocks(%bx, %by, %bz) in (%sbx = %sz, %sby = %sz, %sbz = %sz)
             threads(%tx, %ty, %tz) in (%stx = %sz, %sty = %sz, %stz = %sz) {
    // @expected-error@+1 {{expected 'gpu.terminator' or a terminator with successors}}
    return
  }
  return
}

// -----

func @launch_func_too_few_operands(%sz : index) {
  // expected-error@+1 {{expected 6 or more operands}}
  "gpu.launch_func"(%sz, %sz, %sz, %sz, %sz)
      : (index, index, index, index, index) -> ()
  return
}

// -----

func @launch_func_missing_callee_attribute(%sz : index) {
  // expected-error@+1 {{attribute 'kernel' must be specified}}
  "gpu.launch_func"(%sz, %sz, %sz, %sz, %sz, %sz) {foo: "bar"}
      : (index, index, index, index, index, index) -> ()
  return
}

// -----

func @launch_func_no_function_attribute(%sz : index) {
  // expected-error@+1 {{attribute 'kernel' must be a function}}
  "gpu.launch_func"(%sz, %sz, %sz, %sz, %sz, %sz) {kernel: 10}
      : (index, index, index, index, index, index) -> ()
  return
}

// -----

func @launch_func_undefined_function(%sz : index) {
  // expected-error@+1 {{kernel function '@kernel_1' is undefined}}
  "gpu.launch_func"(%sz, %sz, %sz, %sz, %sz, %sz) { kernel: @kernel_1 }
      : (index, index, index, index, index, index) -> ()
  return
}

// -----

func @kernel_1(%arg1 : !llvm<"float*">) {
  return
}

func @launch_func_missing_kernel_attr(%sz : index, %arg : !llvm<"float*">) {
  // expected-error@+1 {{kernel function is missing the 'gpu.kernel' attribute}}
  "gpu.launch_func"(%sz, %sz, %sz, %sz, %sz, %sz, %arg) {kernel: @kernel_1}
      : (index, index, index, index, index, index, !llvm<"float*">) -> ()
  return
}

// -----

func @kernel_1(%arg1 : !llvm<"float*">) attributes { gpu.kernel } {
  return
}

func @launch_func_kernel_operand_size(%sz : index, %arg : !llvm<"float*">) {
  // expected-error@+1 {{got 2 kernel operands but expected 1}}
  "gpu.launch_func"(%sz, %sz, %sz, %sz, %sz, %sz, %arg, %arg)
      {kernel: @kernel_1}
      : (index, index, index, index, index, index, !llvm<"float*">,
         !llvm<"float*">) -> ()
  return
}

// -----

func @kernel_1(%arg1 : !llvm<"float*">) attributes { gpu.kernel } {
  return
}

func @launch_func_kernel_operand_types(%sz : index, %arg : f32) {
  // expected-error@+1 {{type of function argument 0 does not match}}
  "gpu.launch_func"(%sz, %sz, %sz, %sz, %sz, %sz, %arg)
      {kernel: @kernel_1}
      : (index, index, index, index, index, index, f32) -> ()
  return
}
