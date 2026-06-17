; RUN: ir-compiler-opt %s | FileCheck %s
; Basic test to verify the tool can read and output LLVM IR

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @simple_function(i32 %x) {
; CHECK: define i32 @simple_function
; CHECK: %x
entry:
  ret i32 %x
; CHECK: ret i32
}