// RUN: xla-runtime-opt %s --split-input-file --xla-rt-ordinal-assignment \
// RUN:   | FileCheck %s

// CHECK: rt.export @exported.0 ordinal 0
// CHECK: rt.export @exported.1 ordinal 1
rt.export @exported.0
rt.export @exported.1

func.func @exported.0() { return }
func.func @exported.1() { return }

// -----

// CHECK: rt.export @exported.0 ordinal 0
// CHECK: rt.export @exported.1 ordinal 1
rt.export @exported.0 ordinal 0
rt.export @exported.1

func.func @exported.0() { return }
func.func @exported.1() { return }

// -----

// CHECK: rt.export @exported.0 ordinal 1
// CHECK: rt.export @exported.1 ordinal 0
rt.export @exported.0 ordinal 1
rt.export @exported.1

func.func @exported.0() { return }
func.func @exported.1() { return }

// -----

// CHECK: rt.export @exported.0 ordinal 0
// CHECK: rt.export @exported.1 ordinal 1
// CHECK: rt.export @exported.2 ordinal 2
rt.export @exported.0
rt.export @exported.1
rt.export @exported.2 ordinal 2

func.func @exported.0() { return }
func.func @exported.1() { return }
func.func @exported.2() { return }

// -----

// CHECK: rt.export @exported.0 ordinal 0
// CHECK: rt.export @exported.1 ordinal 2
// CHECK: rt.export @exported.2 ordinal 1
rt.export @exported.0
rt.export @exported.1
rt.export @exported.2 ordinal 1

func.func @exported.0() { return }
func.func @exported.1() { return }
func.func @exported.2() { return }
