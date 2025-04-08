// RUN: triton-opt %s -split-input-file -verify-diagnostics

// expected-error@+2 {{ttg.dot_op opIdx parameter can be 0 or 1, got: 2}}
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#dot_op = #ttg.dot_op<{opIdx = 2, parent = #blocked, kWidth = 2}>

// -----

// expected-error@+2 {{ttg.dot_op kWidth parameter is not supported when the parent is a blocked layout}}
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#dot_op = #ttg.dot_op<{opIdx = 1, parent = #blocked, kWidth = 8}>

// -----

// expected-error@+2 {{ttg.dot_op kWidth parameter can only be non-zero for Ampere or Hopper MMA parent}}
#mma = #ttg.nvidia_mma<{versionMajor = 1, warpsPerCTA = [1, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], instrShape = [16, 8]}>
#dot_op = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>

// -----

// expected-error@+2 {{ttg.dot_op kWidth parameter is mandatory for Ampere or Hopper MMA parent}}
#mma = #ttg.nvidia_mma<{versionMajor = 2, warpsPerCTA = [1, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], instrShape = [16, 8]}>
#dot_op = #ttg.dot_op<{opIdx = 0, parent = #mma}>

// -----

// expected-error@+2 {{ttg.dot_op kWidth parameter is mandatory for Ampere or Hopper MMA parent}}
#mma = #ttg.nvidia_mma<{versionMajor = 3, warpsPerCTA = [1, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], instrShape = [16, 8]}>
#dot_op = #ttg.dot_op<{opIdx = 0, parent = #mma}>

// -----

// expected-error@+2 {{ttg.dot_op opIdx parameter must be 0 for Hopper MMA parent, since Hopper WGMMA only allows first operand to be in registers}}
#mma = #ttg.nvidia_mma<{versionMajor = 3, warpsPerCTA = [1, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], instrShape = [16, 8]}>
#dot_op = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>

// -----

// expected-error@+2 {{ttg.dot_op kWidth parameter is mandatory for MFMA parent}}
#mfma = #ttg.amd_mfma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 1, 1], instrShape = [32, 32], isTransposed = false}>
#dot_op = #ttg.dot_op<{opIdx = 1, parent = #mfma}>

// -----

// expected-error@+2 {{ttg.dot_op kWidth parameter must be 16 for gfx11 and 8/16 for gfx12}}
#wmma = #ttg.amd_wmma<{version = 1, warpsPerCTA = [1, 4]}>
#dot_op = #ttg.dot_op<{opIdx = 1, parent = #wmma}>

// -----

// expected-error@+2 {{ttg.dot_op kWidth parameter must be 16 for gfx11 and 8/16 for gfx12}}
#wmma = #ttg.amd_wmma<{version = 1, warpsPerCTA = [1, 4]}>
#dot_op = #ttg.dot_op<{opIdx = 1, parent = #wmma, kWidth = 8}>

// -----
// expected-error@+2 {{ttg.dot_op kWidth parameter must be 16 for gfx11 and 8/16 for gfx12}}
#wmma = #ttg.amd_wmma<{version = 2, warpsPerCTA = [1, 4]}>
#dot_op = #ttg.dot_op<{opIdx = 1, parent = #wmma, kWidth = 32}>

// -----
// expected-error@+2 {{ttg.dot_op kWidth parameter must be 16 for gfx11 and 8/16 for gfx12}}
#wmma = #ttg.amd_wmma<{version = 2, warpsPerCTA = [1, 4]}>
#dot_op = #ttg.dot_op<{opIdx = 1, parent = #wmma, kWidth = 4}>

// -----

// expected-error@+1 {{major version must be in the [0, 4] range}}
#mfma = #ttg.amd_mfma<{versionMajor = 10, versionMinor = 0, warpsPerCTA = [1, 1, 1], instrShape = [32, 32], isTransposed = false}>

// -----

// expected-error@+1 {{minor version must be 0}}
#mfma = #ttg.amd_mfma<{versionMajor = 2, versionMinor = 5, warpsPerCTA = [1, 1, 1], instrShape = [32, 32], isTransposed = false}>

// -----

// expected-error@+1 {{(M, N) cases other than (32, 32) or (16, 16) unimplemented}}
#mfma = #ttg.amd_mfma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 1, 1], instrShape = [16, 8], isTransposed = false}>
