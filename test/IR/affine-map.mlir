// RUN: mlir-opt %s | FileCheck %s

// Identity maps used in trivial compositions in MemRefs are optimized away.
// CHECK-NOT: #map{{[0-9]+}} = (d0, d1) -> (d0, d1)
#map0 = (i, j) -> (i, j)

// CHECK-NOT: #map{{[0-9]+}} = (d0, d1)[s0] -> (d0, d1)
#map1 = (i, j)[s0] -> (i, j)

// CHECK: #map{{[0-9]+}} = () -> (0)
// A map may have 0 inputs.
// However, an affine_apply always takes at least one input.
#map2 = () -> (0)

// All the maps in the following block are equivalent and are unique'd as one
// map. Therefore there should be only one output and we explicitly CHECK-NOT
// for the others.
// CHECK: #map{{[0-9]+}} = (d0, d1) -> (d0 + 1, d1 * 4 + 2)
#map3  = (i, j) -> (i+1, 4*j + 2)
// CHECK-NOT: #map3{{[a-z]}}
#map3a = (i, j) -> (1+i, 4*j + 2)
#map3b = (i, j) -> (2 + 3 - 2*2 + i, 4*j + 2)
#map3c = (i, j) -> (i +1 + 0, 4*j + 2)
#map3d = (i, j) -> (i + 3 + 2 - 4, 4*j + 2)
#map3e = (i, j) -> (1*i+3*2-2*2-1, 4*j + 2)
#map3f = (i, j) -> (i + 1, 4*j*1 + 2)
#map3g = (i, j) -> (i + 1, 2*2*j + 2)
#map3h = (i, j) -> (i + 1, 2*j*2 + 2)
#map3i = (i, j) -> (i + 1, j*2*2 + 2)
#map3j = (i, j) -> (i + 1, j*1*4 + 2)
#map3k = (i, j) -> (i + 1, j*4*1 + 2)

// The following reduction should be unique'd out too but such expression
// simplification is not performed for IR parsing, but only through analyses
// and transforms.
// CHECK: #map{{[0-9]+}} = (d0, d1) -> (d1 - d0 + (d0 - d1 + 1) * 2 + d1 - 1, d1 + d1 + d1 + d1 + 2)
#map3l = (i, j) -> ((j - i) + 2*(i - j + 1) + j - 1 + 0, j + j + 1 + j + j + 1)

// CHECK: #map{{[0-9]+}} = (d0, d1) -> (d0 + 2, d1)
#map4  = (i, j) -> (3+3-2*2+i, j)

// CHECK: #map{{[0-9]+}} = (d0, d1)[s0] -> (d0 + s0, d1)
#map5 = (i, j)[s0] -> (i + s0, j)

// CHECK: #map{{[0-9]+}} = (d0, d1)[s0] -> (d0 + s0, d1 + 5)
#map6 = (i, j)[s0] -> (i + s0, j + 5)

// CHECK: #map{{[0-9]+}} = (d0, d1)[s0] -> (d0 + d1 + s0, d1)
#map7 = (i, j)[s0] -> (i + j + s0, j)

// CHECK: #map{{[0-9]+}} = (d0, d1)[s0] -> (d0 + d1 + s0 + 5, d1)
#map8 = (i, j)[s0] -> (5 + i + j + s0, j)

// CHECK: #map{{[0-9]+}} = (d0, d1)[s0] -> (d0 + d1 + 5, d1)
#map9 = (i, j)[s0] -> ((i + j) + 5, j)

// CHECK: #map{{[0-9]+}} = (d0, d1)[s0] -> (d0 + d1 + 5, d1)
#map10 = (i, j)[s0] -> (i + (j + 5), j)

// CHECK: #map{{[0-9]+}} = (d0, d1)[s0] -> (d0 * 2, d1 * 3)
#map11 = (i, j)[s0] -> (2*i, 3*j)

// CHECK: #map{{[0-9]+}} = (d0, d1)[s0] -> (d0 + (d1 + s0 * 3) * 5 + 12, d1)
#map12 = (i, j)[s0] -> (i + 2*6 + 5*(j+s0*3), j)

// CHECK: #map{{[0-9]+}} = (d0, d1)[s0] -> (d0 * 5 + d1, d1)
#map13 = (i, j)[s0] -> (5*i + j, j)

// CHECK: #map{{[0-9]+}} = (d0, d1)[s0] -> (d0 + d1, d1)
#map14 = (i, j)[s0] -> ((i + j), (j))

// CHECK: #map{{[0-9]+}} = (d0, d1)[s0] -> (d0 + d1 + 7, d1 + 3)
#map15 = (i, j)[s0] -> ((i + j + 2) + 5, (j)+3)

// CHECK: #map{{[0-9]+}} = (d0, d1)[s0] -> (d0, 0)
#map16 = (i, j)[s1] -> (i, 0)

// CHECK: #map{{[0-9]+}} = (d0, d1)[s0] -> (d0, d1 * s0)
#map17 = (i, j)[s0] -> (i, s0*j)

// CHECK: #map{{[0-9]+}} = (d0, d1) -> (d0, d0 * 3 + d1)
#map19 = (i, j) -> (i, 3*i + j)

// CHECK: #map{{[0-9]+}} = (d0, d1) -> (d0, d0 + d1 * 3)
#map20 = (i, j)  -> (i, i + 3*j)

// CHECK: #map{{[0-9]+}} = (d0, d1)[s0] -> (d0, d0 * ((s0 * s0) * 9) + 3)
#map18 = (i, j)[N] -> (i, 2 + N*N*9*i + 1)

// CHECK: #map{{[0-9]+}} = (d0, d1) -> (1, d0 + d1 * 3 + 5)
#map21 = (i, j)  -> (1, i + 3*j + 5)

// CHECK: #map{{[0-9]+}} = (d0, d1)[s0] -> (s0 * 5, d0 + d1 * 3 + d0 * 5)
#map22 = (i, j)[s0] -> (5*s0, i + 3*j + 5*i)

// CHECK: #map{{[0-9]+}} = (d0, d1)[s0, s1] -> (d0 * (s0 * s1), d1)
#map23 = (i, j)[s0, s1] -> (i*(s0*s1), j)

// CHECK: #map{{[0-9]+}} = (d0, d1)[s0, s1] -> (d0, d1 mod 5)
#map24 = (i, j)[s0, s1] -> (i, j mod 5)

// CHECK: #map{{[0-9]+}} = (d0, d1)[s0, s1] -> (d0, d1 floordiv 5)
#map25 = (i, j)[s0, s1] -> (i, j floordiv 5)

// CHECK: #map{{[0-9]+}} = (d0, d1)[s0, s1] -> (d0, d1 ceildiv 5)
#map26 = (i, j)[s0, s1] -> (i, j ceildiv 5)

// CHECK: #map{{[0-9]+}} = (d0, d1)[s0, s1] -> (d0, d0 - d1 - 5)
#map29 = (i, j)[s0, s1] -> (i, i - j - 5)

// CHECK: #map{{[0-9]+}} = (d0, d1)[s0, s1] -> (d0, d0 - d1 * s1 + 2)
#map30 = (i, j)[M, N] -> (i, i - N*j + 2)

// CHECK: #map{{[0-9]+}} = (d0, d1)[s0, s1] -> (d0 * -5, d1 * -3, -2, -(d0 + d1), -s0)
#map32 = (i, j)[s0, s1] -> (-5*i, -3*j, -2, -1*(i+j), -1*s0)

// CHECK: #map{{[0-9]+}} = (d0, d1) -> (-4, -d0)
#map33 = (i, j) -> (-2+-5-(-3), -1*i)

// CHECK: #map{{[0-9]+}} = (d0, d1)[s0, s1] -> (d0, d1 floordiv s0, d1 mod s0)
#map34 = (i, j)[s0, s1] -> (i, j floordiv s0, j mod s0)

// CHECK: #map{{[0-9]+}} = (d0, d1, d2)[s0, s1, s2] -> ((d0 * s1) * s2 + d1 * s1 + d2)
#map35 = (i, j, k)[s0, s1, s2] -> (i*s1*s2 + j*s1 + k)

// Constant folding.
// CHECK: #map{{[0-9]+}} = (d0, d1) -> (8, 4, 1, 3, 2, 4)
#map36 = (i, j) -> (5+3, 2*2, 8-7, 100 floordiv 32, 5 mod 3, 10 ceildiv 3)
// CHECK: #map{{[0-9]+}} = (d0, d1) -> (4, 11, 512, 15)
#map37 = (i, j) -> (5 mod 3 + 2, 5*3 - 4, 128 * (500 ceildiv 128), 40 floordiv 7 * 3)

// CHECK: #map{{[0-9]+}} = (d0, d1) -> (d0 * 2 + 1, d1 + 2)
#map38 = (i, j) -> (1 + i*2, 2 + j)

// CHECK: #map{{[0-9]+}} = (d0, d1)[s0, s1] -> (d0 * s0, d0 + s0, d0 + 2, d1 * 2, s1 * 2, s0 + 2)
#map39 = (i, j)[M, N] -> (i*M, M + i, 2+i, j*2, N*2, 2 + M)

// CHECK: #map{{[0-9]+}} = (d0, d1) -> (d0, d1) size (10, 20)
#map40 = (i, j) -> (i, j) size (10, 20)

// CHECK: #map{{[0-9]+}} = (d0, d1)[s0, s1] -> (d0, d1) size (s0, s1 + 10)
#map41 = (i, j)[N, M] -> (i, j) size (N, M+10)

// CHECK: #map{{[0-9]+}} = (d0, d1)[s0, s1] -> (d0, d1) size (128, s0 * 2 + s1 + 5)
#map42 = (i, j)[N, M] -> (i, j) size (64 + 64, 5 + 2*N + M)

// CHECK: #map{{[0-9]+}} = (d0, d1)[s0] -> ((d0 * 5) floordiv 4, (d1 ceildiv 7) mod s0)
#map43 = (i, j) [s0] -> ( i * 5 floordiv 4, j ceildiv 7 mod s0)

// CHECK: #map{{[0-9]+}} = (d0, d1) -> (d0 - d1 * 2, (d1 * 6) floordiv 4)
#map44 = (i, j) -> (i - 2*j, j * 6 floordiv 4)

// Simplifications
// CHECK: #map{{[0-9]+}} = (d0, d1, d2)[s0] -> (d0 + d1 + d2 + 1, d2 + d1, (d0 * s0) * 8)
#map45 = (i, j, k) [N] -> (1 + i + 3 + j - 3 + k, k + 5 + j - 5, 2*i*4*N)

// CHECK: #map{{[0-9]+}} = (d0, d1, d2) -> (0, d1, d0 * 2, 0)
#map46 = (i, j, k) -> (i*0, 1*j, i * 128 floordiv 64, j * 0 floordiv 64)

// CHECK: #map{{[0-9]+}} = (d0, d1, d2) -> (d0, d0 * 4, 0, 0, 0)
#map47 = (i, j, k) -> (i * 64 ceildiv 64, i * 512 ceildiv 128, 4 * j mod 4, 4*j*4 mod 8, k mod 1)

// floordiv should resolve similarly to ceildiv and be unique'd out.
// CHECK-NOT: #map48{{[a-z]}}
#map48 = (i, j, k) -> (i * 64 floordiv 64, i * 512 floordiv 128, 4 * j mod 4, 4*j*4 mod 8)

// Simplifications for mod using known GCD's of the LHS expr.
// CHECK: #map{{[0-9]+}} = (d0, d1)[s0] -> (0, 0, 0, (d0 * 4 + 3) mod 2)
#map49 = (i, j)[s0] -> ( (i * 4 + 8) mod 4, 32 * j * s0 * 8 mod 256, (4*i + (j * (s0 * 2))) mod 2, (4*i + 3) mod 2)

// Floordiv, ceildiv divide by one.
// CHECK: #map{{[0-9]+}} = (d0, d1)[s0] -> (d0 * 2 + 1, d1 + s0)
#map50 = (i, j)[s0] -> ( (i * 2 + 1) ceildiv 1, (j + s0) floordiv 1)

// floordiv, ceildiv, and mod where LHS is negative.
// CHECK: #map{{[0-9]+}} = (d0) -> (-2, 1, -1)
#map51 = (i) -> (-5 floordiv 3, -5 mod 3, -5 ceildiv 3)

// Parenthesis ellision.
// CHECK: #map{{[0-9]+}} = (d0) -> (d0 * 16 - (d0 + 1) + 15)
#map52 = (d0) -> (16*d0 + ((d0 + 1) * -1) + 15)

// Single identity maps are removed.
// CHECK: func @f0(memref<2x4xi8, 1>)
func @f0(memref<2x4xi8, #map0, 1>)

// Single identity maps are removed.
// CHECK: func @f1(memref<2x4xi8, 1>)
func @f1(memref<2x4xi8, #map1, 1>)

// CHECK: func @f2(memref<i8, #map{{[0-9]+}}, 1>)
func @f2(memref<i8, #map2, 1>)

// CHECK: func @f3(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func @f3(memref<2x4xi8, #map3, 1>)
// CHECK: func @f3a(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func @f3a(memref<2x4xi8, #map3a, 1>)
// CHECK: func @f3b(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func @f3b(memref<2x4xi8, #map3b, 1>)
// CHECK: func @f3c(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func @f3c(memref<2x4xi8, #map3c, 1>)
// CHECK: func @f3d(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func @f3d(memref<2x4xi8, #map3d, 1>)
// CHECK: func @f3e(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func @f3e(memref<2x4xi8, #map3e, 1>)
// CHECK: func @f3f(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func @f3f(memref<2x4xi8, #map3f, 1>)
// CHECK: func @f3g(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func @f3g(memref<2x4xi8, #map3g, 1>)
// CHECK: func @f3h(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func @f3h(memref<2x4xi8, #map3h, 1>)
// CHECK: func @f3i(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func @f3i(memref<2x4xi8, #map3i, 1>)
// CHECK: func @f3j(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func @f3j(memref<2x4xi8, #map3j, 1>)
// CHECK: func @f3k(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func @f3k(memref<2x4xi8, #map3k, 1>)
// CHECK: func @f3l(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func @f3l(memref<2x4xi8, #map3l, 1>)

// CHECK: func @f4(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func @f4(memref<2x4xi8, #map4, 1>)

// CHECK: func @f5(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func @f5(memref<2x4xi8, #map5, 1>)

// CHECK: func @f6(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func @f6(memref<2x4xi8, #map6, 1>)

// CHECK: func @f7(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func @f7(memref<2x4xi8, #map7, 1>)

// CHECK: func @f8(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func @f8(memref<2x4xi8, #map8, 1>)

// CHECK: func @f9(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func @f9(memref<2x4xi8, #map9, 1>)

// CHECK: func @f10(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func @f10(memref<2x4xi8, #map10, 1>)

// CHECK: func @f11(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func @f11(memref<2x4xi8, #map11, 1>)

// CHECK: func @f12(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func @f12(memref<2x4xi8, #map12, 1>)

// CHECK: func @f13(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func @f13(memref<2x4xi8, #map13, 1>)

// CHECK: func @f14(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func @f14(memref<2x4xi8, #map14, 1>)

// CHECK: func @f15(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func @f15(memref<2x4xi8, #map15, 1>)

// CHECK: func @f16(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func @f16(memref<2x4xi8, #map16, 1>)

// CHECK: func @f17(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func @f17(memref<2x4xi8, #map17, 1>)

// CHECK: func @f19(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func @f19(memref<2x4xi8, #map19, 1>)

// CHECK: func @f20(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func @f20(memref<2x4xi8, #map20, 1>)

// CHECK: func @f18(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func @f18(memref<2x4xi8, #map18, 1>)

// CHECK: func @f21(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func @f21(memref<2x4xi8, #map21, 1>)

// CHECK: func @f22(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func @f22(memref<2x4xi8, #map22, 1>)

// CHECK: func @f23(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func @f23(memref<2x4xi8, #map23, 1>)

// CHECK: func @f24(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func @f24(memref<2x4xi8, #map24, 1>)

// CHECK: func @f25(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func @f25(memref<2x4xi8, #map25, 1>)

// CHECK: func @f26(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func @f26(memref<2x4xi8, #map26, 1>)

// CHECK: func @f29(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func @f29(memref<2x4xi8, #map29, 1>)

// CHECK: func @f30(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func @f30(memref<2x4xi8, #map30, 1>)

// CHECK: func @f32(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func @f32(memref<2x4xi8, #map32, 1>)

// CHECK: func @f33(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func @f33(memref<2x4xi8, #map33, 1>)

// CHECK: func @f34(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func @f34(memref<2x4xi8, #map34, 1>)

// CHECK: func @f35(memref<2x4x4xi8, #map{{[0-9]+}}, 1>)
func @f35(memref<2x4x4xi8, #map35, 1>)

// CHECK: func @f36(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func @f36(memref<2x4xi8, #map36, 1>)

// CHECK: func @f37(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func @f37(memref<2x4xi8, #map37, 1>)

// CHECK: func @f38(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func @f38(memref<2x4xi8, #map38, 1>)

// CHECK: func @f39(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func @f39(memref<2x4xi8, #map39, 1>)

// CHECK: func @f40(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func @f40(memref<2x4xi8, #map40, 1>)

// CHECK: func @f41(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func @f41(memref<2x4xi8, #map41, 1>)

// CHECK: func @f42(memref<2x4xi8, #map{{[0-9]+}}, 1>)
func @f42(memref<2x4xi8, #map42, 1>)

// CHECK: func @f43(memref<2x4xi8, #map{{[0-9]+}}>)
func @f43(memref<2x4xi8, #map43>)

// CHECK: func @f44(memref<2x4xi8, #map{{[0-9]+}}>)
func @f44(memref<2x4xi8, #map44>)

// CHECK: func @f45(memref<100x100x100xi8, #map{{[0-9]+}}>)
func @f45(memref<100x100x100xi8, #map45>)

// CHECK: func @f46(memref<100x100x100xi8, #map{{[0-9]+}}>)
func @f46(memref<100x100x100xi8, #map46>)

// CHECK: func @f47(memref<100x100x100xi8, #map{{[0-9]+}}>)
func @f47(memref<100x100x100xi8, #map47>)

// CHECK: func @f48(memref<100x100x100xi8, #map{{[0-9]+}}>)
func @f48(memref<100x100x100xi8, #map48>)

// CHECK: func @f49(memref<100x100xi8, #map{{[0-9]+}}>)
func @f49(memref<100x100xi8, #map49>)

// CHECK: func @f50(memref<100x100xi8, #map{{[0-9]+}}>)
func @f50(memref<100x100xi8, #map50>)

// CHECK: func @f51(memref<1xi8, #map{{[0-9]+}}>)
func @f51(memref<1xi8, #map51>)

// CHECK: func @f52(memref<1xi8, #map{{[0-9]+}}>)
func @f52(memref<1xi8, #map52>)
