// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package weakdeps exists to add weak module dependencies.
//
// We want to ensure that this module is used with a minimum
// version of certain other modules, without actually importing
// those modules in normal builds. We do that by adding an
// import of a package in the module under a build constraint
// that is never satisfied in normal usage.
package weakdeps
