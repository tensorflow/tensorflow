// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package version records versioning information about this module.
package version

import (
	"fmt"
	"strings"
)

// These constants determine the current version of this module.
//
// For our release process, we enforce the following rules:
//   - Tagged releases use a tag that is identical to String.
//   - Tagged releases never reference a commit where the String
//     contains "devel".
//   - The set of all commits in this repository where String
//     does not contain "devel" must have a unique String.
//
// Steps for tagging a new release:
//
//  1. Create a new CL.
//
//  2. Update Minor, Patch, and/or PreRelease as necessary.
//     PreRelease must not contain the string "devel".
//
//  3. Since the last released minor version, have there been any changes to
//     generator that relies on new functionality in the runtime?
//     If yes, then increment RequiredGenerated.
//
//  4. Since the last released minor version, have there been any changes to
//     the runtime that removes support for old .pb.go source code?
//     If yes, then increment SupportMinimum.
//
//  5. Send out the CL for review and submit it.
//     Note that the next CL in step 8 must be submitted after this CL
//     without any other CLs in-between.
//
//  6. Tag a new version, where the tag is is the current String.
//
//  7. Write release notes for all notable changes
//     between this release and the last release.
//
//  8. Create a new CL.
//
//  9. Update PreRelease to include the string "devel".
//     For example: "" -> "devel" or "rc.1" -> "rc.1.devel"
//
//  10. Send out the CL for review and submit it.
const (
	Major      = 1
	Minor      = 32
	Patch      = 0
	PreRelease = "devel"
)

// String formats the version string for this module in semver format.
//
// Examples:
//
//	v1.20.1
//	v1.21.0-rc.1
func String() string {
	v := fmt.Sprintf("v%d.%d.%d", Major, Minor, Patch)
	if PreRelease != "" {
		v += "-" + PreRelease

		// TODO: Add metadata about the commit or build hash.
		// See https://golang.org/issue/29814
		// See https://golang.org/issue/33533
		var metadata string
		if strings.Contains(PreRelease, "devel") && metadata != "" {
			v += "+" + metadata
		}
	}
	return v
}
