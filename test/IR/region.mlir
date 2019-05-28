// RUN: mlir-test-opt %s -split-input-file -verify | FileCheck %s

func @correct_number_of_regions() {
    // CHECK: test.two_region_op
    "test.two_region_op"()(
      {"work"() : () -> ()},
      {"work"() : () -> ()}
    ) : () -> ()
    return
}

// -----

func @missingk_regions() {
    // expected-error@+1 {{op has incorrect number of regions: expected 2 but found 1}}
    "test.two_region_op"()(
      {"work"() : () -> ()}
    ) : () -> ()
    return
}

// -----

func @extra_regions() {
    // expected-error@+1 {{op has incorrect number of regions: expected 2 but found 3}}
    "test.two_region_op"()(
      {"work"() : () -> ()},
      {"work"() : () -> ()},
      {"work"() : () -> ()}
    ) : () -> ()
    return
}
