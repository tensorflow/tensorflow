// RUN: fusion_compiler_opt %s --xtile-cpu-elemental-tensor-to-vector -split-input-file | FileCheck %s

func.func @addf(%lhs : tensor<1024xf32>, %rhs : tensor<1024xf32>) -> tensor<1024xf32> {
  // CHECK: arith.addf %{{.*}}, %{{.*}} : vector<1024xf32>
  %add = arith.addf %lhs, %rhs : tensor<1024xf32>
  return %add : tensor<1024xf32>
}

// -----

func.func @addi(%lhs : tensor<1024xi32>, %rhs : tensor<1024xi32>) -> tensor<1024xi32> {
  // CHECK: arith.addi %{{.*}}, %{{.*}} : vector<1024xi32>
  %add = arith.addi %lhs, %rhs : tensor<1024xi32>
  return %add : tensor<1024xi32>
}

// -----

func.func @addiu(%lhs : tensor<1024xi32>, %rhs : tensor<1024xi32>) -> (tensor<1024xi32>, tensor<1024xi1>) {
  // CHECK: arith.addui_extended %{{.*}}, %{{.*}} : vector<1024xi32>, vector<1024xi1>
  %add, %carry = arith.addui_extended %lhs, %rhs : tensor<1024xi32>, tensor<1024xi1>
  return %add, %carry : tensor<1024xi32>, tensor<1024xi1>
}

// -----

func.func @andi(%lhs : tensor<1024xi32>, %rhs : tensor<1024xi32>) -> tensor<1024xi32> {
  // CHECK: arith.andi %{{.*}}, %{{.*}} : vector<1024xi32>
  %and = arith.andi %lhs, %rhs : tensor<1024xi32>
  return %and : tensor<1024xi32>
}

// -----

func.func @bitcast(%arg0 : tensor<1024xi32>) -> tensor<1024xf32> {
  // CHECK: arith.bitcast %{{.*}} : vector<1024xi32> to vector<1024xf32>
  %cast = arith.bitcast %arg0 : tensor<1024xi32> to tensor<1024xf32>
  return %cast : tensor<1024xf32>
}

// -----

func.func @ceildivsi(%lhs : tensor<1024xi32>, %rhs : tensor<1024xi32>) -> tensor<1024xi32> {
  // CHECK: arith.ceildivsi %{{.*}}, %{{.*}} : vector<1024xi32>
  %div = arith.ceildivsi %lhs, %rhs : tensor<1024xi32>
  return %div : tensor<1024xi32>
}

// -----

func.func @ceildivui(%lhs : tensor<1024xi32>, %rhs : tensor<1024xi32>) -> tensor<1024xi32> {
  // CHECK: arith.ceildivui %{{.*}}, %{{.*}} : vector<1024xi32>
  %div = arith.ceildivui %lhs, %rhs : tensor<1024xi32>
  return %div : tensor<1024xi32>
}

// -----

func.func @cmpf(%lhs : tensor<1024xf32>, %rhs : tensor<1024xf32>) -> tensor<1024xi1> {
  // CHECK: arith.cmpf oeq, %{{.*}}, %{{.*}} : vector<1024xf32>
  %cmp = arith.cmpf oeq, %lhs, %rhs : tensor<1024xf32>
  return %cmp : tensor<1024xi1>
}

// -----

func.func @cmpi(%lhs : tensor<1024xi32>, %rhs : tensor<1024xi32>) -> tensor<1024xi1> {
  // CHECK: arith.cmpi eq, %{{.*}}, %{{.*}} : vector<1024xi32>
  %cmp = arith.cmpi eq, %lhs, %rhs : tensor<1024xi32>
  return %cmp : tensor<1024xi1>
}

// -----

func.func @constant() -> tensor<1024xf32> {
  // CHECK: arith.constant dense<1.000000e+00> : vector<1024xf32>
  %const = arith.constant dense<1.0> : tensor<1024xf32>
  return %const : tensor<1024xf32>
}

// -----

func.func @divf(%lhs : tensor<1024xf32>, %rhs : tensor<1024xf32>) -> tensor<1024xf32> {
  // CHECK: arith.divf %{{.*}}, %{{.*}} : vector<1024xf32>
  %div = arith.divf %lhs, %rhs : tensor<1024xf32>
  return %div : tensor<1024xf32>
}

// -----

func.func @divsi(%lhs : tensor<1024xi32>, %rhs : tensor<1024xi32>) -> tensor<1024xi32> {
  // CHECK: arith.divsi %{{.*}}, %{{.*}} : vector<1024xi32>
  %div = arith.divsi %lhs, %rhs : tensor<1024xi32>
  return %div : tensor<1024xi32>
}

// -----

func.func @divui(%lhs : tensor<1024xi32>, %rhs : tensor<1024xi32>) -> tensor<1024xi32> {
  // CHECK: arith.divui %{{.*}}, %{{.*}} : vector<1024xi32>
  %div = arith.divui %lhs, %rhs : tensor<1024xi32>
  return %div : tensor<1024xi32>
}

// -----

func.func @extf(%arg0 : tensor<1024xf32>) -> tensor<1024xf64> {
  // CHECK: arith.extf %{{.*}} : vector<1024xf32> to vector<1024xf64>
  %ext = arith.extf %arg0 : tensor<1024xf32> to tensor<1024xf64>
  return %ext : tensor<1024xf64>
}

// -----

func.func @extsi(%arg0 : tensor<1024xi16>) -> tensor<1024xi32> {
  // CHECK: arith.extsi %{{.*}} : vector<1024xi16> to vector<1024xi32>
  %ext = arith.extsi %arg0 : tensor<1024xi16> to tensor<1024xi32>
  return %ext : tensor<1024xi32>
}

// -----

func.func @extui(%arg0 : tensor<1024xi16>) -> tensor<1024xi32> {
  // CHECK: arith.extui %{{.*}} : vector<1024xi16> to vector<1024xi32>
  %ext = arith.extui %arg0 : tensor<1024xi16> to tensor<1024xi32>
  return %ext : tensor<1024xi32>
}

// -----

func.func @fptosi(%arg0 : tensor<1024xf32>) -> tensor<1024xi32> {
  // CHECK: arith.fptosi %{{.*}} : vector<1024xf32> to vector<1024xi32>
  %cast = arith.fptosi %arg0 : tensor<1024xf32> to tensor<1024xi32>
  return %cast : tensor<1024xi32>
}

// -----

func.func @fptoui(%arg0 : tensor<1024xf32>) -> tensor<1024xi32> {
  // CHECK: arith.fptoui %{{.*}} : vector<1024xf32> to vector<1024xi32>
  %cast = arith.fptoui %arg0 : tensor<1024xf32> to tensor<1024xi32>
  return %cast : tensor<1024xi32>
}

// -----

func.func @floordivsi(%lhs : tensor<1024xi32>, %rhs : tensor<1024xi32>) -> tensor<1024xi32> {
  // CHECK: arith.floordivsi %{{.*}}, %{{.*}} : vector<1024xi32>
  %div = arith.floordivsi %lhs, %rhs : tensor<1024xi32>
  return %div : tensor<1024xi32>
}

// -----

func.func @index_cast(%arg0 : tensor<1024xi32>) -> tensor<1024xindex> {
  // CHECK: arith.index_cast %{{.*}} : vector<1024xi32> to vector<1024xindex>
  %cast = arith.index_cast %arg0 : tensor<1024xi32> to tensor<1024xindex>
  return %cast : tensor<1024xindex>
}

// -----

func.func @index_castui(%arg0 : tensor<1024xi32>) -> tensor<1024xindex> {
  // CHECK: arith.index_castui %{{.*}} : vector<1024xi32> to vector<1024xindex>
  %cast = arith.index_castui %arg0 : tensor<1024xi32> to tensor<1024xindex>
  return %cast : tensor<1024xindex>
}

// -----

func.func @maxnumf(%lhs : tensor<1024xf32>, %rhs : tensor<1024xf32>) -> tensor<1024xf32> {
  // CHECK: arith.maxnumf %{{.*}}, %{{.*}} : vector<1024xf32>
  %max = arith.maxnumf %lhs, %rhs : tensor<1024xf32>
  return %max : tensor<1024xf32>
}

// -----

func.func @maxsi(%lhs : tensor<1024xi32>, %rhs : tensor<1024xi32>) -> tensor<1024xi32> {
  // CHECK: arith.maxsi %{{.*}}, %{{.*}} : vector<1024xi32>
  %max = arith.maxsi %lhs, %rhs : tensor<1024xi32>
  return %max : tensor<1024xi32>
}

// -----

func.func @maxui(%lhs : tensor<1024xi32>, %rhs : tensor<1024xi32>) -> tensor<1024xi32> {
  // CHECK: arith.maxui %{{.*}}, %{{.*}} : vector<1024xi32>
  %max = arith.maxui %lhs, %rhs : tensor<1024xi32>
  return %max : tensor<1024xi32>
}

// -----

func.func @maximumf(%lhs : tensor<1024xf32>, %rhs : tensor<1024xf32>) -> tensor<1024xf32> {
  // CHECK: arith.maximumf %{{.*}}, %{{.*}} : vector<1024xf32>
  %max = arith.maximumf %lhs, %rhs : tensor<1024xf32>
  return %max : tensor<1024xf32>
}

// -----

func.func @minnumf(%lhs : tensor<1024xf32>, %rhs : tensor<1024xf32>) -> tensor<1024xf32> {
  // CHECK: arith.minnumf %{{.*}}, %{{.*}} : vector<1024xf32>
  %min = arith.minnumf %lhs, %rhs : tensor<1024xf32>
  return %min : tensor<1024xf32>
}

// -----

func.func @minsi(%lhs : tensor<1024xi32>, %rhs : tensor<1024xi32>) -> tensor<1024xi32> {
  // CHECK: arith.minsi %{{.*}}, %{{.*}} : vector<1024xi32>
  %min = arith.minsi %lhs, %rhs : tensor<1024xi32>
  return %min : tensor<1024xi32>
}

// -----

func.func @minui(%lhs : tensor<1024xi32>, %rhs : tensor<1024xi32>) -> tensor<1024xi32> {
  // CHECK: arith.minui %{{.*}}, %{{.*}} : vector<1024xi32>
  %min = arith.minui %lhs, %rhs : tensor<1024xi32>
  return %min : tensor<1024xi32>
}

// -----

func.func @minimumf(%lhs : tensor<1024xf32>, %rhs : tensor<1024xf32>) -> tensor<1024xf32> {
  // CHECK: arith.minimumf %{{.*}}, %{{.*}} : vector<1024xf32>
  %min = arith.minimumf %lhs, %rhs : tensor<1024xf32>
  return %min : tensor<1024xf32>
}

// -----

func.func @mulf(%lhs : tensor<1024xf32>, %rhs : tensor<1024xf32>) -> tensor<1024xf32> {
  // CHECK: arith.mulf %{{.*}}, %{{.*}} : vector<1024xf32>
  %mul = arith.mulf %lhs, %rhs : tensor<1024xf32>
  return %mul : tensor<1024xf32>
}

// -----

func.func @muli(%lhs : tensor<1024xi32>, %rhs : tensor<1024xi32>) -> tensor<1024xi32> {
  // CHECK: arith.muli %{{.*}}, %{{.*}} overflow<nsw, nuw> : vector<1024xi32>
  %mul = arith.muli %lhs, %rhs overflow<nsw, nuw> : tensor<1024xi32>
  return %mul : tensor<1024xi32>
}

// -----

func.func @mului_ext(%lhs : tensor<1024xi32>, %rhs : tensor<1024xi32>) -> (tensor<1024xi32>, tensor<1024xi32>) {
  // CHECK: arith.mulsi_extended %{{.*}}, %{{.*}} : vector<1024xi32>
  %low, %high = arith.mulsi_extended %lhs, %rhs : tensor<1024xi32>
  return %low, %high : tensor<1024xi32>, tensor<1024xi32>
}

// -----

func.func @negf(%arg0 : tensor<1024xf32>) -> tensor<1024xf32> {
  // CHECK: arith.negf %{{.*}} : vector<1024xf32>
  %neg = arith.negf %arg0 : tensor<1024xf32>
  return %neg : tensor<1024xf32>
}

// -----

func.func @ori(%lhs : tensor<1024xi32>, %rhs : tensor<1024xi32>) -> tensor<1024xi32> {
  // CHECK: arith.ori %{{.*}}, %{{.*}} : vector<1024xi32>
  %or = arith.ori %lhs, %rhs : tensor<1024xi32>
  return %or : tensor<1024xi32>
}

// -----

func.func @remf(%lhs : tensor<1024xf32>, %rhs : tensor<1024xf32>) -> tensor<1024xf32> {
  // CHECK: arith.remf %{{.*}}, %{{.*}} : vector<1024xf32>
  %rem = arith.remf %lhs, %rhs : tensor<1024xf32>
  return %rem : tensor<1024xf32>
}

// -----

func.func @remsi(%lhs : tensor<1024xi32>, %rhs : tensor<1024xi32>) -> tensor<1024xi32> {
  // CHECK: arith.remsi %{{.*}}, %{{.*}} : vector<1024xi32>
  %rem = arith.remsi %lhs, %rhs : tensor<1024xi32>
  return %rem : tensor<1024xi32>
}

// -----

func.func @remui(%lhs : tensor<1024xi32>, %rhs : tensor<1024xi32>) -> tensor<1024xi32> {
  // CHECK: arith.remui %{{.*}}, %{{.*}} : vector<1024xi32>
  %rem = arith.remui %lhs, %rhs : tensor<1024xi32>
  return %rem : tensor<1024xi32>
}

// -----

func.func @sitofp(%arg0 : tensor<1024xi32>) -> tensor<1024xf32> {
  // CHECK: arith.sitofp %{{.*}} : vector<1024xi32> to vector<1024xf32>
  %cast = arith.sitofp %arg0 : tensor<1024xi32> to tensor<1024xf32>
  return %cast : tensor<1024xf32>
}

// -----

func.func @select(%cond : tensor<1024xi1>, %true_val : tensor<1024xf32>, %false_val : tensor<1024xf32>) -> tensor<1024xf32> {
  // CHECK: arith.select %{{.*}}, %{{.*}}, %{{.*}} : vector<1024xi1>, vector<1024xf32>
  %sel = arith.select %cond, %true_val, %false_val : tensor<1024xi1>, tensor<1024xf32>
  return %sel : tensor<1024xf32>
}

// -----

func.func @shli(%lhs : tensor<1024xi32>, %rhs : tensor<1024xi32>) -> tensor<1024xi32> {
  // CHECK: arith.shli %{{.*}}, %{{.*}} : vector<1024xi32>
  %shl = arith.shli %lhs, %rhs : tensor<1024xi32>
  return %shl : tensor<1024xi32>
}

// -----

func.func @shrsi(%lhs : tensor<1024xi32>, %rhs : tensor<1024xi32>) -> tensor<1024xi32> {
  // CHECK: arith.shrsi %{{.*}}, %{{.*}} : vector<1024xi32>
  %shr = arith.shrsi %lhs, %rhs : tensor<1024xi32>
  return %shr : tensor<1024xi32>
}

// -----

func.func @shrui(%lhs : tensor<1024xi32>, %rhs : tensor<1024xi32>) -> tensor<1024xi32> {
  // CHECK: arith.shrui %{{.*}}, %{{.*}} : vector<1024xi32>
  %shr = arith.shrui %lhs, %rhs : tensor<1024xi32>
  return %shr : tensor<1024xi32>
}

// -----

func.func @subf(%lhs : tensor<1024xf32>, %rhs : tensor<1024xf32>) -> tensor<1024xf32> {
  // CHECK: arith.subf %{{.*}}, %{{.*}} : vector<1024xf32>
  %sub = arith.subf %lhs, %rhs : tensor<1024xf32>
  return %sub : tensor<1024xf32>
}

// -----

func.func @subi(%lhs : tensor<1024xi32>, %rhs : tensor<1024xi32>) -> tensor<1024xi32> {
  // CHECK: arith.subi %{{.*}}, %{{.*}} : vector<1024xi32>
  %sub = arith.subi %lhs, %rhs : tensor<1024xi32>
  return %sub : tensor<1024xi32>
}

// -----

func.func @truncf(%arg0 : tensor<1024xf64>) -> tensor<1024xf32> {
  // CHECK: arith.truncf %{{.*}} : vector<1024xf64> to vector<1024xf32>
  %trunc = arith.truncf %arg0 : tensor<1024xf64> to tensor<1024xf32>
  return %trunc : tensor<1024xf32>
}

// -----

func.func @trunci(%arg0 : tensor<1024xi32>) -> tensor<1024xi16> {
  // CHECK: arith.trunci %{{.*}} : vector<1024xi32> to vector<1024xi16>
  %trunc = arith.trunci %arg0 : tensor<1024xi32> to tensor<1024xi16>
  return %trunc : tensor<1024xi16>
}

// -----

func.func @uitofp(%arg0 : tensor<1024xi32>) -> tensor<1024xf32> {
  // CHECK: arith.uitofp %{{.*}} : vector<1024xi32> to vector<1024xf32>
  %cast = arith.uitofp %arg0 : tensor<1024xi32> to tensor<1024xf32>
  return %cast : tensor<1024xf32>
}

// -----

func.func @xori(%lhs : tensor<1024xi32>, %rhs : tensor<1024xi32>) -> tensor<1024xi32> {
  // CHECK: arith.xori %{{.*}}, %{{.*}} : vector<1024xi32>
  %xor = arith.xori %lhs, %rhs : tensor<1024xi32>
  return %xor : tensor<1024xi32>
}