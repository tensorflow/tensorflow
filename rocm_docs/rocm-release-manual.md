How to synchonize TensorFlow ROCm port with upstream TensorFlow
===============================================================

This document articulates the step to create tensorflow rocm releases from upstream.


Useful github remote repositories (For reference)
-------------------------------------------------

git locations of repositories used in the merge process are:

- TensorFlow ROCm port:
  - URL: <https://github.com/ROCm/tensorflow-upstream.git>
  - branch: develop-upstream
- TensorFlow upstream:
  - URL: <https://github.com/tensorflow/tensorflow.git>
  - branch: master

# Step-by-step Merge Process
--------------------------

## Get TensorFlow ROCm port prior to the release branching on upstream
Check the upstream when the new release was updated by checking the verison bump history in
[Release.md](https://github.com/tensorflow/tensorflow/blob/master/RELEASE.md)
For r2.19 it is this PR https://github.com/tensorflow/tensorflow/pull/86867 merged on 2025-02-10.
Checkout verified weekly-sync prior to this date.
(for r2.19 it is https://github.com/ROCm/tensorflow-upstream/tree/develop-upstream-sync-20250204)

```
git clone git@github.com:ROCm/tensorflow-upstream.git
git checkout merge-yy(yy)mmdd
git pull
```

## Add remotes
```
git remote add upstream https://github.com/tensorflow/tensorflow.git
git fetch upstream
```

## Merge release branch upstream/rX.xx

While creating this document it is r2.19

```
git merge upstream/rX.xx --no-edit
```

- Make first commit to record the merge state before resolving conflicts
  - Record all files with merge conflict locally
  - Mark all conflict files as resolved by staging them
  - Commit all files as-is. Note that at this stage the build should be broken
    because of the "<<<<<<======>>>>>>" symbols in source files
- Resolve any merge conflicts encountered here
- Make the second commit to record the merge state after resolving conflicts
  - When all merge conflict resolved, do a ```grep -rn "<<<<<<"``` to make sure
    no diff symbols exist in the source

# Build merged TensorFlow

- Build with either `bazel build` rule or `build_rocm_python3` script. Make sure everything
  builds fine and Python PIP whl package can be built.

## Push the release branch

- git checkout -b rX.xx-rocm-enhanced
- git push --set-upstream origin rX.xx-rocm-enhanced
- Create a CI for new release branch at http://ml-ci.amd.com:21096/job/tensorflow/


