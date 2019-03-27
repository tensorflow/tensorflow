// RUN: mlir-opt %s -verify-each=true -cse -canonicalize -cse -pass-timing -pass-timing-display=list 2>&1 | FileCheck -check-prefix=LIST %s
// RUN: mlir-opt %s -verify-each=true -cse -canonicalize -cse -pass-timing -pass-timing-display=pipeline 2>&1 | FileCheck -check-prefix=PIPELINE %s
// RUN: mlir-opt %s -experimental-mt-pm=true -verify-each=true -cse -canonicalize -cse -pass-timing -pass-timing-display=list 2>&1 | FileCheck -check-prefix=MT_LIST %s
// RUN: mlir-opt %s -experimental-mt-pm=true -verify-each=true -cse -canonicalize -cse -pass-timing -pass-timing-display=pipeline 2>&1 | FileCheck -check-prefix=MT_PIPELINE %s

// LIST: Pass execution timing report
// LIST: Total Execution Time:
// LIST: Name
// LIST-DAG: Canonicalizer
// LIST-DAG: FunctionVerifier
// LIST-DAG: CSE
// LIST-DAG: ModuleVerifier
// LIST-DAG: DominanceInfo
// LIST: Total

// PIPELINE: Pass execution timing report
// PIPELINE: Total Execution Time:
// PIPELINE: Name
// PIPELINE-NEXT: Function Pipeline
// PIPELINE-NEXT:   CSE
// PIPELINE-NEXT:     (A) DominanceInfo
// PIPELINE-NEXT:   FunctionVerifier
// PIPELINE-NEXT:   Canonicalizer
// PIPELINE-NEXT:   FunctionVerifier
// PIPELINE-NEXT:   CSE
// PIPELINE-NEXT:     (A) DominanceInfo
// PIPELINE-NEXT:   FunctionVerifier
// PIPELINE-NEXT: ModuleVerifier
// PIPELINE-NEXT: Total

// MT_LIST: Pass execution timing report
// MT_LIST: Total Execution Time:
// MT_LIST:    ---User Time---   ---Wall Time---  --- Name ---
// MT_LIST-DAG: Canonicalizer
// MT_LIST-DAG: FunctionVerifier
// MT_LIST-DAG: CSE
// MT_LIST-DAG: ModuleVerifier
// MT_LIST-DAG: DominanceInfo
// MT_LIST: Total

// MT_PIPELINE: Pass execution timing report
// MT_PIPELINE: Total Execution Time:
// MT_PIPELINE:    ---User Time---   ---Wall Time---  --- Name ---
// MT_PIPELINE-NEXT: Function Pipeline
// MT_PIPELINE-NEXT:   CSE
// MT_PIPELINE-NEXT:     (A) DominanceInfo
// MT_PIPELINE-NEXT:   FunctionVerifier
// MT_PIPELINE-NEXT:   Canonicalizer
// MT_PIPELINE-NEXT:   FunctionVerifier
// MT_PIPELINE-NEXT:   CSE
// MT_PIPELINE-NEXT:     (A) DominanceInfo
// MT_PIPELINE-NEXT:   FunctionVerifier
// MT_PIPELINE-NEXT: ModuleVerifier
// MT_PIPELINE-NEXT: Total

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
