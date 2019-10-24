// RUN: mlir-opt %s -disable-pass-threading=true -verify-each=true -pass-pipeline='func(cse,canonicalize,cse)' -pass-timing -pass-timing-display=list 2>&1 | FileCheck -check-prefix=LIST %s
// RUN: mlir-opt %s -disable-pass-threading=true -verify-each=true -pass-pipeline='func(cse,canonicalize,cse)' -pass-timing -pass-timing-display=pipeline 2>&1 | FileCheck -check-prefix=PIPELINE %s
// RUN: mlir-opt %s -disable-pass-threading=false -verify-each=true -pass-pipeline='func(cse,canonicalize,cse)' -pass-timing -pass-timing-display=list 2>&1 | FileCheck -check-prefix=MT_LIST %s
// RUN: mlir-opt %s -disable-pass-threading=false -verify-each=true -pass-pipeline='func(cse,canonicalize,cse)' -pass-timing -pass-timing-display=pipeline 2>&1 | FileCheck -check-prefix=MT_PIPELINE %s
// RUN: mlir-opt %s -disable-pass-threading=false -verify-each=false -test-pm-nested-pipeline -pass-timing -pass-timing-display=pipeline 2>&1 | FileCheck -check-prefix=NESTED_MT_PIPELINE %s

// LIST: Pass execution timing report
// LIST: Total Execution Time:
// LIST: Name
// LIST-DAG: Canonicalizer
// LIST-DAG: Verifier
// LIST-DAG: CSE
// LIST-DAG: DominanceInfo
// LIST: Total

// PIPELINE: Pass execution timing report
// PIPELINE: Total Execution Time:
// PIPELINE: Name
// PIPELINE-NEXT: 'func' Pipeline
// PIPELINE-NEXT:   CSE
// PIPELINE-NEXT:     (A) DominanceInfo
// PIPELINE-NEXT:   Verifier
// PIPELINE-NEXT:   Canonicalizer
// PIPELINE-NEXT:   Verifier
// PIPELINE-NEXT:   CSE
// PIPELINE-NEXT:     (A) DominanceInfo
// PIPELINE-NEXT:   Verifier
// PIPELINE-NEXT: Verifier
// PIPELINE-NEXT: Total

// MT_LIST: Pass execution timing report
// MT_LIST: Total Execution Time:
// MT_LIST: Name
// MT_LIST-DAG: Canonicalizer
// MT_LIST-DAG: Verifier
// MT_LIST-DAG: CSE
// MT_LIST-DAG: DominanceInfo
// MT_LIST: Total

// MT_PIPELINE: Pass execution timing report
// MT_PIPELINE: Total Execution Time:
// MT_PIPELINE: Name
// MT_PIPELINE-NEXT: 'func' Pipeline
// MT_PIPELINE-NEXT:   CSE
// MT_PIPELINE-NEXT:     (A) DominanceInfo
// MT_PIPELINE-NEXT:   Verifier
// MT_PIPELINE-NEXT:   Canonicalizer
// MT_PIPELINE-NEXT:   Verifier
// MT_PIPELINE-NEXT:   CSE
// MT_PIPELINE-NEXT:     (A) DominanceInfo
// MT_PIPELINE-NEXT:   Verifier
// MT_PIPELINE-NEXT: Verifier
// MT_PIPELINE-NEXT: Total

// NESTED_MT_PIPELINE: Pass execution timing report
// NESTED_MT_PIPELINE: Total Execution Time:
// NESTED_MT_PIPELINE: Name
// NESTED_MT_PIPELINE-NEXT: Pipeline Collection : ['func', 'module']
// NESTED_MT_PIPELINE-NEXT:   'func' Pipeline
// NESTED_MT_PIPELINE-NEXT:     TestFunctionPass
// NESTED_MT_PIPELINE-NEXT:   'module' Pipeline
// NESTED_MT_PIPELINE-NEXT:     TestModulePass
// NESTED_MT_PIPELINE-NEXT:     'func' Pipeline
// NESTED_MT_PIPELINE-NEXT:       TestFunctionPass
// NESTED_MT_PIPELINE-NEXT: Total

func @foo() {
  return
}

func @bar() {
  return
}

func @baz() {
  return
}

func @foobar() {
  return
}

module {
  func @baz() {
    return
  }

  func @foobar() {
    return
  }
}
