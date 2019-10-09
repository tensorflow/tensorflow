// RUN: not mlir-opt %s -pass-pipeline='module(test-module-pass{)' 2>&1 | FileCheck --check-prefix=CHECK_ERROR_1 %s
// RUN: not mlir-opt %s -pass-pipeline='module(test-module-pass{test-option=3})' 2>&1 | FileCheck --check-prefix=CHECK_ERROR_2 %s
// RUN: not mlir-opt %s -pass-pipeline='module(test-options-pass{list=3}, test-module-pass{invalid-option=3})' 2>&1 | FileCheck --check-prefix=CHECK_ERROR_3 %s
// RUN: not mlir-opt %s -pass-pipeline='test-options-pass{list=3 list=notaninteger}' 2>&1 | FileCheck --check-prefix=CHECK_ERROR_4 %s
// RUN: not mlir-opt %s -pass-pipeline='func(test-options-pass{list=1,2,3,4 list=5 string=value1 string=value2})' 2>&1 | FileCheck --check-prefix=CHECK_ERROR_5 %s
// RUN: mlir-opt %s -pass-pipeline='func(test-options-pass{string-list=a list=1,2,3,4 string-list=b,c list=5 string-list=d string=some_value})' 2>&1 | FileCheck --check-prefix=CHECK_1 %s
// RUN: mlir-opt %s -test-options-pass-pipeline='list=1 string-list=a,b' 2>&1 | FileCheck --check-prefix=CHECK_2 %s
// RUN: mlir-opt %s -pass-pipeline='module(test-options-pass{list=3}, test-options-pass{list=1,2,3,4})' 2>&1 | FileCheck --check-prefix=CHECK_3 %s

// CHECK_ERROR_1: missing closing '}' while processing pass options
// CHECK_ERROR_2: no such option test-option
// CHECK_ERROR_3: no such option invalid-option
// CHECK_ERROR_4: 'notaninteger' value invalid for integer argument
// CHECK_ERROR_5: string option: may only occur zero or one times

// CHECK_1: test-options-pass{list=1,2,3,4,5 string-list=a,b,c,d string=some_value}
// CHECK_2: test-options-pass{list=1 string-list=a,b}
// CHECK_3: test-options-pass{list=3}
// CHECK_3-NEXT: test-options-pass{list=1,2,3,4}
