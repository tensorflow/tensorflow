// RUN: mlir-opt -split-input-file -verify %s

func @not_enough_sizes(%sz : index) {
  // expected-error@+1 {{expected 6 or more operands}}
  "gpu.launch"(%sz, %sz, %sz, %sz, %sz) ({
    return
  }) : (index, index, index, index, index) -> ()
  return
}

// -----

func @no_region_attrs(%sz : index) {
  // expected-error@+1 {{unexpected number of region arguments}}
 "gpu.launch"(%sz, %sz, %sz, %sz, %sz, %sz) ({
  ^bb1(%bx: index, %by: index, %bz: index,
       %tx: index, %ty: index, %tz: index):
    return
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
    return
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
    return
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
  }) : (index, index, index, index, index, index) -> ()
  return
}
