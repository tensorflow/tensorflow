; RUN: %S/../../mlir-opt %s -o - | FileCheck %s

; CHECK: #{{[0-9]+}} = (d0, d1) -> (d0, d1)
#hello_world0 = (i, j) -> (i, j)

; CHECK: #{{[0-9]+}} = (d0, d1) [s0] -> (d0, d1)
#hello_world1 = (i, j) [s0] -> (i, j)

; CHECK: #{{[0-9]+}} = () -> (0)
#hello_world2 = () -> (0)

; CHECK: #{{[0-9]+}} = (d0, d1) -> ((d0 + 1), d1)
#hello_world3 = (i, j) -> (i+1, j)

; CHECK: #{{[0-9]+}} = (d0, d1) [s0] -> ((d0 + s0), d1)
#hello_world4 = (i, j) [s0] -> (i + s0, j)

; CHECK: #{{[0-9]+}} = (d0, d1) -> ((d0 + 1), d1)
#hello_world5 = (i, j) -> (1+i, j)

; CHECK: #{{[0-9]+}} = (d0, d1) [s0] -> ((d0 + s0), (d1 + 5))
#hello_world6 = (i, j) [s0] -> (i + s0, j + 5)

; CHECK: #{{[0-9]+}} = (d0, d1) [s0] -> (((d0 + d1) + s0), d1)
#hello_world7 = (i, j) [s0] -> (i + j + s0, j)

; CHECK: #{{[0-9]+}} = (d0, d1) [s0] -> ((((d0 + 5) + d1) + s0), d1)
#hello_world8 = (i, j) [s0] -> (5 + i + j + s0, j)

; CHECK: #{{[0-9]+}} = (d0, d1) [s0] -> (((d0 + d1) + 5), d1)
#hello_world9 = (i, j) [s0] -> ((i + j) + 5, j)

; CHECK: #{{[0-9]+}} = (d0, d1) [s0] -> ((d0 + (d1 + 5)), d1)
#hello_world10 = (i, j) [s0] -> (i + (j + 5), j)

; CHECK: #{{[0-9]+}} = (d0, d1) [s0] -> ((d0 * 2), (d1 * 3))
#hello_world11 = (i, j) [s0] -> (2*i, 3*j)

; CHECK: #{{[0-9]+}} = (d0, d1) [s0] -> (((d0 + 12) + ((d1 + (s0 * 3)) * 5)), d1)
#hello_world12 = (i, j) [s0] -> (i + 2*6 + 5*(j+s0*3), j)

; CHECK: #{{[0-9]+}} = (d0, d1) [s0] -> (((d0 * 5) + d1), d1)
#hello_world13 = (i, j) [s0] -> (5*i + j, j)

; CHECK: #{{[0-9]+}} = (d0, d1) [s0] -> ((d0 + d1), d1)
#hello_world14 = (i, j) [s0] -> ((i + j), (j))

; CHECK: #{{[0-9]+}} = (d0, d1) [s0] -> (((d0 + d1) + 5), (d1 + 3))
#hello_world15 = (i, j) [s0] -> ((i + j)+5, (j)+3)

; CHECK: #{{[0-9]+}} = (d0, d1) [s0] -> (d0, 0)
#hello_world16 = (i, j) [s1] -> (i, 0)

; CHECK: #{{[0-9]+}} = (d0, d1) [s0] -> (d0, (d1 * s0))
#hello_world17 = (i, j) [s0] -> (i, s0*j)

; CHECK: #{{[0-9]+}} = (d0, d1) -> (d0, ((d0 * 3) + d1))
#hello_world19 = (i, j) -> (i, 3*i + j)

; CHECK: #{{[0-9]+}} = (d0, d1) -> (d0, (d0 + (d1 * 3)))
#hello_world20 = (i, j)  -> (i, i + 3*j)

; CHECK: #{{[0-9]+}} = (d0, d1) [s0] -> (d0, (((d0 * ((s0 * s0) * 9)) + 2) + 1))
#hello_world18 = (i, j) [N] -> (i, 2 + N*N*9*i + 1)

; CHECK: #{{[0-9]+}} = (d0, d1) -> (1, ((d0 + (d1 * 3)) + 5))
#hello_world21 = (i, j)  -> (1, i + 3*j + 5)

; CHECK: #{{[0-9]+}} = (d0, d1) [s0] -> ((s0 * 5), ((d0 + (d1 * 3)) + (d0 * 5)))
#hello_world22 = (i, j) [s0] -> (5*s0, i + 3*j + 5*i)

; CHECK: #{{[0-9]+}} = (d0, d1) [s0, s1] -> ((d0 * (s0 * s1)), d1)
#hello_world23 = (i, j) [s0, s1] -> (i*(s0*s1), j)

; CHECK: #{{[0-9]+}} = (d0, d1) [s0, s1] -> (d0, (d1 mod 5))
#hello_world24 = (i, j) [s0, s1] -> (i, j mod 5)

; CHECK: #{{[0-9]+}} = (d0, d1) [s0, s1] -> (d0, (d1 floordiv 5))
#hello_world25 = (i, j) [s0, s1] -> (i, j floordiv 5)

; CHECK: #{{[0-9]+}} = (d0, d1) [s0, s1] -> (d0, (d1 ceildiv 5))
#hello_world26 = (i, j) [s0, s1] -> (i, j ceildiv 5)

; CHECK: #{{[0-9]+}} = (d0, d1) [s0, s1] -> (d0, ((d0 - d1) - 5))
#hello_world29 = (i, j) [s0, s1] -> (i, i - j - 5)

; CHECK: #{{[0-9]+}} = (d0, d1) [s0, s1] -> (d0, ((d0 - (d1 * s1)) + 2))
#hello_world30 = (i, j) [M, N] -> (i, i - N*j + 2)

; CHECK: #{{[0-9]+}} = (d0, d1) [s0, s1] -> ((d0 * -5), (d1 * -3), -2, ((d0 + d1) * -1), (s0 * -1))
#hello_world32 = (i, j) [s0, s1] -> (-5*i, -3*j, -2, -1*(i+j), -1*s0)

; CHECK: #{{[0-9]+}} = (d0, d1) -> (-4, (d0 * -1))
#hello_world33 = (i, j) -> (-2+-5-(-3), -1*i)

; CHECK: #{{[0-9]+}} = (d0, d1) [s0, s1] -> (d0, (d1 floordiv s0), (d1 mod s0))
#hello_world34 = (i, j) [s0, s1] -> (i, j floordiv s0, j mod s0)

; CHECK: #{{[0-9]+}} = (d0, d1, d2) [s0, s1, s2] -> (((((d0 * s1) * s2) + (d1 * s1)) + d2))
#hello_world35 = (i, j, k) [s0, s1, s2] -> (i*s1*s2 + j*s1 + k)

; CHECK: #{{[0-9]+}} = (d0, d1) -> (8, 4, 1, 3, 2, 4)
#hello_world36 = (i, j) -> (5+3, 2*2, 8-7, 100 floordiv 32, 5 mod 3, 10 ceildiv 3)

; CHECK: #{{[0-9]+}} = (d0, d1) -> (4, 11, 512, 15)
#hello_world37 = (i, j) -> (5 mod 3 + 2, 5*3 - 4, 128 * (500 ceildiv 128), 40 floordiv 7 * 3)

; CHECK: #{{[0-9]+}} = (d0, d1) -> (((d0 * 2) + 1), (d1 + 2))
#hello_world38 = (i, j) -> (1 + i*2, 2 + j)

; CHECK: #{{[0-9]+}} = (d0, d1) [s0, s1] -> ((d0 * s0), (d0 + s0), (d0 + 2), (d1 * 2), (s1 * 2), (s0 + 2))
#hello_world39 = (i, j) [M, N] -> (i*M, M + i, 2+i, j*2, N*2, 2 + M)

; CHECK: #{{[0-9]+}} = (d0, d1) -> (d0, d1) size (10, 20)
#hello_world40 = (i, j) -> (i, j) size (10, 20)

; CHECK: #{{[0-9]+}} = (d0, d1) [s0, s1] -> (d0, d1) size (s0, (s1 + 10))
#hello_world41 = (i, j) [N, M] -> (i, j) size (N, M+10)

; CHECK: #{{[0-9]+}} = (d0, d1) [s0, s1] -> (d0, d1) size (128, (((s0 * 2) + 5) + s1))
#hello_world42 = (i, j) [N, M] -> (i, j) size (64 + 64, 5 + 2*N + M)
