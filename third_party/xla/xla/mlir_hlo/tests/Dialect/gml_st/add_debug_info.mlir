// RUN: mlir-hlo-opt %s --add-debug-info --mlir-print-debuginfo | FileCheck %s

builtin.module {
  func.func @foo() {
    return
  }
}

// CHECK: module
// CHECK:   func.func @[[SUBPROGRAM_NAME:.*]]() {
// CHECK:     return loc(#[[RET_LOC:.*]])
// CHECK:   } loc(#[[FUSED_SUBPROGRAM_LOC:.*]])
// CHECK: } loc(#[[MODULE_LOC:.*]])
// CHECK: #di_basic_type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "void", encoding = DW_ATE_address>
// CHECK: #di_file = #llvm.di_file<"[[FILE_NAME:.*]]" in "[[DIR_NAME:.*]]">
// CHECK: #[[MODULE_LOC]] = loc("[[DIR_NAME]]/[[FILE_NAME]]":[[#MODULE_LINE:]]:1)
// CHECK: #[[SUBPROGRAM_LOC:.*]] = loc("[[DIR_NAME]]/[[FILE_NAME]]":[[#MODULE_LINE+1]]:3)
// CHECK: #[[RET_LOC]] = loc("[[DIR_NAME]]/[[FILE_NAME]]":[[#MODULE_LINE+2]]:5)
// CHECK: #di_compile_unit = #llvm.di_compile_unit<sourceLanguage = DW_LANG_C_plus_plus_17, file = #di_file, producer = "XLA CPU", isOptimized = false, emissionKind = LineTablesOnly>
// CHECK: #di_subroutine_type = #llvm.di_subroutine_type<callingConvention = DW_CC_normal, types = #di_basic_type>
// CHECK: #di_subprogram = #llvm.di_subprogram<compileUnit = #di_compile_unit, scope = #di_file, name = "[[SUBPROGRAM_NAME]]", linkageName = "[[SUBPROGRAM_NAME]]", file = #di_file, line = 1, scopeLine = 1, subprogramFlags = Definition, type = #di_subroutine_type>
// CHECK: #[[FUSED_SUBPROGRAM_LOC]] = loc(fused<#di_subprogram>[#[[SUBPROGRAM_LOC]]])