// RUN: mlir-opt %s -vectorizer-test -compose-maps |  FileCheck %s

// For all these cases, the test traverses the `test_affine_map` ops and
// composes them in order one-by-one.
// For instance, the pseudo-sequence:
//    "test_affine_map"() { affine_map: f } : () -> ()
//    "test_affine_map"() { affine_map: g } : () -> ()
//    "test_affine_map"() { affine_map: h } : () -> ()
// will produce the sequence of compositions: f, g(f), h(g(f)) and print the
// AffineMap h(g(f)), which is what FileCheck checks against.

func @simple1() {
  // CHECK: Composed map: (d0) -> (d0)
  "test_affine_map"() { affine_map: (d0) -> (d0 - 1) } : () -> ()
  "test_affine_map"() { affine_map: (d0) -> (d0 + 1) } : () -> ()
  return
}

func @simple2() {
  // CHECK: Composed map: (d0)[s0] -> (d0)
  "test_affine_map"() { affine_map: (d0)[s0] -> (d0 + s0 - 1) } : () -> ()
  "test_affine_map"() { affine_map: (d0)[s0] -> (d0 - s0 + 1) } : () -> ()
  return
}

func @simple3a() {
  // CHECK: Composed map: (d0, d1)[s0, s1] -> ((d0 ceildiv s0) * s0, (d1 ceildiv s1) * s1)
  "test_affine_map"() { affine_map: (d0, d1)[s0, s1] -> (d0 ceildiv s0, d1 ceildiv s1) } : () -> ()
  "test_affine_map"() { affine_map: (d0, d1)[s0, s1] -> (d0 * s0, d1 * s1) } : () -> ()
  return
}

func @simple3b() {
  // CHECK: Composed map: (d0, d1)[s0, s1] -> (d0 mod s0, d1 mod s1)
  "test_affine_map"() { affine_map: (d0, d1)[s0, s1] -> (d0 mod s0, d1 mod s1) } : () -> ()
  return
}

func @simple3c() {
  // CHECK: Composed map: (d0, d1)[s0, s1, s2, s3] -> ((d0 ceildiv s0) * s0 + d0 mod s2, (d1 ceildiv s1) * s1 + d1 mod s3)
  "test_affine_map"() { affine_map: (d0, d1)[s0, s1] -> ((d0 ceildiv s0) * s0, (d1 ceildiv s1) * s1, d0, d1) } : () -> ()
  "test_affine_map"() { affine_map: (d0, d1, d2, d3)[s0, s1, s2, s3] -> (d0 + d2 mod s2, d1 + d3 mod s3) } : () -> ()
  return
}

func @simple4() {
  // CHECK: Composed map: (d0, d1)[s0, s1] -> (d1 * s1, d0 ceildiv s0)
  "test_affine_map"() { affine_map: (d0, d1) -> (d1, d0) } : () -> ()
  "test_affine_map"() { affine_map: (d0, d1)[s0, s1] -> (d0 * s1, d1 ceildiv s0) } : () -> ()
  return
}

func @simple5a() {
  // CHECK: Composed map: (d0) -> (d0 * 3 + 18)
  "test_affine_map"() { affine_map: (d0) -> (d0 - 1) } : () -> ()
  "test_affine_map"() { affine_map: (d0) -> (d0 + 7) } : () -> ()
  "test_affine_map"() { affine_map: (d0) -> (d0 * 24) } : () -> ()
  "test_affine_map"() { affine_map: (d0) -> (d0 ceildiv 8) } : () -> ()
  return
}

func @simple5b() {
  // CHECK: Composed map: (d0) -> ((d0 + 6) ceildiv 2)
  "test_affine_map"() { affine_map: (d0) -> (d0 - 1) } : () -> ()
  "test_affine_map"() { affine_map: (d0) -> (d0 + 7) } : () -> ()
  "test_affine_map"() { affine_map: (d0) -> (d0 * 4) } : () -> ()
  "test_affine_map"() { affine_map: (d0) -> (d0 ceildiv 8) } : () -> ()
  return
}

func @simple5c() {
  // CHECK: Composed map: (d0) -> (d0 * 8 + 48)
  "test_affine_map"() { affine_map: (d0) -> (d0 - 1) } : () -> ()
  "test_affine_map"() { affine_map: (d0) -> (d0 + 7) } : () -> ()
  "test_affine_map"() { affine_map: (d0) -> (d0 * 24) } : () -> ()
  "test_affine_map"() { affine_map: (d0) -> (d0 floordiv 3) } : () -> ()
  return
}

func @simple5d() {
  // CHECK: Composed map: (d0) -> ((d0 * 4 + 24) floordiv 3)
  "test_affine_map"() { affine_map: (d0) -> (d0 - 1) } : () -> ()
  "test_affine_map"() { affine_map: (d0) -> (d0 + 7) } : () -> ()
  "test_affine_map"() { affine_map: (d0) -> (d0 * 4) } : () -> ()
  "test_affine_map"() { affine_map: (d0) -> (d0 floordiv 3) } : () -> ()
  return
}

func @simple5e() {
  // CHECK: Composed map: (d0) -> ((d0 + 6) ceildiv 8)
  "test_affine_map"() { affine_map: (d0) -> (d0 - 1) } : () -> ()
  "test_affine_map"() { affine_map: (d0) -> (d0 + 7) } : () -> ()
  "test_affine_map"() { affine_map: (d0) -> (d0 ceildiv 8) } : () -> ()
  return
}

func @simple5f() {
  // CHECK: Composed map: (d0) -> ((d0 * 4 - 4) floordiv 3)
  "test_affine_map"() { affine_map: (d0) -> (d0 - 1) } : () -> ()
  "test_affine_map"() { affine_map: (d0) -> (d0 * 4) } : () -> ()
  "test_affine_map"() { affine_map: (d0) -> (d0 floordiv 3) } : () -> ()
  return
}

func @perm_and_proj() {
  // CHECK: Composed map: (d0, d1, d2, d3) -> (d1, d3, d0)
  "test_affine_map"() { affine_map: (d0, d1, d2, d3) -> (d3, d1, d2, d0) } : () -> ()
  "test_affine_map"() { affine_map: (d0, d1, d2, d3) -> (d1, d0, d3) } : () -> ()
  return
}

func @symbols1() {
  // CHECK: Composed map: (d0)[s0] -> (d0 + s0 + 1, d0 - s0 - 1)
  "test_affine_map"() { affine_map: (d0)[s0] -> (d0 + s0, d0 - s0) } : () -> ()
  "test_affine_map"() { affine_map: (d0, d1) -> (d0 + 1, d1 - 1) } : () -> ()
  return
}