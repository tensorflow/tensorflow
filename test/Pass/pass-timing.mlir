// RUN: mlir-opt %s -verify-each=true -cse -canonicalize -cse -pass-timing -pass-timing-display=list 2>&1 | FileCheck -check-prefix=LIST %s
// RUN: mlir-opt %s -verify-each=true -cse -canonicalize -cse -pass-timing -pass-timing-display=pipeline 2>&1 | FileCheck -check-prefix=PIPELINE %s

func @foo() {
  return
}

// LIST: Pass execution timing report
// LIST: Total Execution Time:
// LIST: Name
// LIST-DAG: Canonicalizer
// LIST-DAG: FunctionVerifier
// LIST-DAG: CSE
// LIST-DAG: ModuleVerifier
// LIST-DAG: DominanceInfo
// LIST: {{.*}} Total

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
