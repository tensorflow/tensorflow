#!/bin/bash
# Copyright 2019 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

cd "$(git rev-parse --show-toplevel)"

read -p "What is the next release version (e.g., 'v1.26.0')?  " VERSION
SEMVER_REGEX='^v\([0-9]*\)[.]\([0-9]*\)[.]\([0-9]*\)\([.a-zA-Z0-9A-Z-]*\)$'
if ! [[ -z $(echo $VERSION | sed -e "s/$SEMVER_REGEX//") ]]; then
	echo; echo "invalid: must be a semver string"; exit 1
fi
VERSION_MAJOR=$(echo $VERSION | sed -e "s/$SEMVER_REGEX/\1/")
VERSION_MINOR=$(echo $VERSION | sed -e "s/$SEMVER_REGEX/\2/")
VERSION_PATCH=$(echo $VERSION | sed -e "s/$SEMVER_REGEX/\3/")
VERSION_PRERELEASE=$(echo $VERSION | sed -e "s/$SEMVER_REGEX/\4/")
if ! [[ "$VERSION_MAJOR" =~ ^1$ ]]; then
	echo; echo "invalid: major version must be 1"; exit 1
fi
if ! [[ -z $VERSION_PRERELEASE ]] && ! [[ "$VERSION_PRERELEASE" =~ ^-rc[.][0-9]+$ ]]; then
	echo; echo "invalid: pre-release suffix must be empty or '-rc.X'"; exit 1
fi
VERSION_PRERELEASE=${VERSION_PRERELEASE#"-"} # trim possible leading dash

function version_string() {
	VERSION_STRING="v${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}"
	if ! [[ -z $VERSION_PRERELEASE ]]; then
		VERSION_STRING="${VERSION_STRING}-${VERSION_PRERELEASE}"
	fi
	echo $VERSION_STRING
}

read -p "Were there any changes to the generator that relies on new runtime functionality?  " YN
case $YN in
[Yy]* )
	read -p "  What minor version of the runtime is required now?  " GEN_VERSION
	if ! [[ "$GEN_VERSION" =~ ^[0-9]+$ ]]; then echo; echo "invalid: must be an integer"; exit 1; fi;;
[Nn]* ) ;;
* ) echo; echo "invalid: must be 'yes' or 'no'"; exit 1;;
esac

read -p "Were there any dropped functionality in the runtime for old generated code?  " YN
case $YN in
[Yy]* )
	read -p "  What minor version of the runtime is required now?  " MIN_VERSION
	if ! [[ "$MIN_VERSION" =~ ^[0-9]+$ ]]; then echo; echo "invalid: must be an integer"; exit 1; fi;;
[Nn]* ) ;;
* ) echo; echo "invalid: must be 'yes' or 'no'"; exit 1;;
esac


echo
echo "Preparing changes to release $(version_string)."
echo

set -e

# Create a new branch to contain the release changes.
if [[ $(git branch --list release) ]]; then
	echo "error: release branch already exists"; exit 1
fi
git codereview change release
git codereview sync

# Create commit for actual release.
INPLACE='-i ""' # BSD version of sed expects argument after -i
if [[ "$(sed --version)" == *"GNU"* ]]; then
	INPLACE="-i" # GNU version of sed does not expect argument after -i
fi
sed $INPLACE -e "s/\(Minor *= *\)[0-9]*/\1$VERSION_MINOR/" internal/version/version.go
sed $INPLACE -e "s/\(Patch *= *\)[0-9]*/\1$VERSION_PATCH/" internal/version/version.go
sed $INPLACE -e "s/\(PreRelease *= *\)\"[^\"]*\"/\1\"$VERSION_PRERELEASE\"/" internal/version/version.go
if ! [[ -z $GEN_VERSION ]]; then
	sed $INPLACE -e "s/\(GenVersion *= *\)[0-9]*/\1$GEN_VERSION/" runtime/protoimpl/version.go
fi
if ! [[ -z $MIN_VERSION ]]; then
	sed $INPLACE -e "s/\(MinVersion *= *\)[0-9]*/\1$MIN_VERSION/" runtime/protoimpl/version.go
fi
git commit -a -m "all: release $(version_string)"

# Build release binaries.
go test google.golang.org/protobuf -run='^TestIntegration$' -timeout=60m -count=1 "$@" -buildRelease

# Create commit to start development after release.
VERSION_PRERELEASE="${VERSION_PRERELEASE}.devel" # append ".devel"
VERSION_PRERELEASE="${VERSION_PRERELEASE#"."}"   # trim possible leading "."
sed $INPLACE -e "s/\(PreRelease *= *\)\"[^\"]*\"/\1\"$VERSION_PRERELEASE\"/" internal/version/version.go
git commit -a -m "all: start $(version_string)"

echo
echo "Release changes prepared. Additional steps:"
echo "  1) Submit the changes:"
echo "    a. Mail out the changes: git mail HEAD"
echo "    b. Request a review on the changes and merge them."
echo "  2) Tag a new release on GitHub:"
echo "    a. Write release notes highlighting notable changes."
echo "    b. Attach pre-compiled binaries as assets to the release."
echo
