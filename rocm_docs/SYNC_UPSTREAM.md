How to synchonize TensorFlow ROCm port with upstream TensorFlow
===============================================================

This article shows the step to synchronize TensorFlow with upstream. The
process is currently carried out manually but it could and should be automated.

**Important:** Synchronization should be performed on a weekly basis. Even if it has taken more than 2 weeks (>2) to merge the PR, the next weekly sync should first sync in the middle week between the last one and current one, and then sync the current week.
For example, if the last weekly-sync PR was 3 weeks ago (e.g. 2024-04-29), this weekly-sync (supposed to be 2024-05-13) should start from 2024-05-06, and then do another weekly-sync at 2024-05-13.

Useful github remote repositories (For reference)
-------------------------------------------------

git locations of repositories used in the merge process are:

- TensorFlow ROCm port:
  - URL: <https://github.com/ROCmSoftwarePlatform/tensorflow-upstream.git>
  - branch: develop-upstream
- TensorFlow upstream:
  - URL: <https://github.com/tensorflow/tensorflow.git>
  - branch: master

# Step-by-step Merge Process
--------------------------

## Get TensorFlow ROCm port

```
git clone git@github.com:ROCmSoftwarePlatform/tensorflow-upstream.git
git checkout develop-upstream
git pull
```

## Add remotes
```
git remote add upstream https://github.com/tensorflow/tensorflow.git
git fetch upstream
```

## Create a working branch to track merge process

```
git checkout -b develop-upstream-sync-YYMMDD
git merge upstream/master --no-edit
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

## Notice on common merge conflicts

- bazel / terminology
  - Rename cuda_ to gpu_ . Python unit tests usually run into this issue.

- Eigen
  - In case Eigen commit hash has been changed, contact:
    Deven.Desai@amd.com to get ROCm version of Eigen patched. The next version
    of this process would cover steps to get Eigen properly synchronized with
    Eigen upstream.

- XLA
  - Logic in XLA (tensorflow/compiler) changes relatively frequently. Expect
    API signature changes either within XLA itself or from LLVM.

- StreamExecutor / GPU common runtime
  - Apart from terminology (CUDA v GPU) changes, pay attention to interface
    changes especially in StreamExecutor. Study CUDA version to figure out what
    the latest interface looks like and adjust ROCm version accordingly.

- operator implementation
  - Check for new operators in `tensorflow/core/kernels` directly. Study the
    source code and try enable it adding `|| TENSORFLOW_WITH_ROCM` after
    `#if GOOGLE_CUDA`. Apply necessary changes to kernel call sites follow
    conventions set in other operators.

- upstream failures
  - When a ROCm reason cannot be determined for a newly failing test, attempt
    to roll back any source files related to the failing test to known working
    commits, e.g., the previous weekly merge.  Look for any differences.  By
    the end of the week, look at the most recent commits in case the bug was
    found and fixed upstream.  Cherry-pick and verify the fix.  In the meantime,
    whitelist the failure and indicate it is an upstream failure so that it can
    be restored as soon as possible.

# Build merged TensorFlow

- Build with either `build` or `build_python3` script. Make sure everything
  builds fine and Python PIP whl package can be built.

## Push and Create Pull Request for TensorFlow ROCm port

- git push --set-upstream origin develop-upstream-sync-YYMMDD
- Create a pull request to merge `develop-upstream-sync-YYMMDD` into
  <https://github.com/ROCmSoftwarePlatform/tensorflow-upstream.git> on branch
  `develop-upstream`.
- Wait until the pull request has finished validation on Jenkins CI.

## Update unit test whitelist if necessary

- The first Jenkins CI job might result in failure due to additional failed
  unit tests introduced from new commits upstream. Examine the reason behind
  those failed cases.
- In case those cases can't be easily fixed, modify the bazel target for that
  test to add one more of the`no_rocm`, `no_rocm_v2`, `no_cuda` tags to it.

  For example:\
  for the test `//tensorflow/python/kernel_tests:conv_ops_test`\
  the definition for the test-target `conv_ops_test` will be in the file
  `tensorflow/python/kernel_tests/BUILD`.
  - Adding `tags = ["no_rocm",]` to that target, will result in removing this
    test from rocm* CI runs.
  - Adding `tags = ["no_rocm_v2",]` to that target, will result in removing this
    test from rocm-v2 CI run.
  - Adding `tags = ["no_cuda",]` to that target, will result in removing this
    test from cuda* CI runs.
  - Adding `tags = ["no_rocm","no_cuda",]` to that target, will result in
    removing this test from both the rocm* and cuda* CI runs.

  grep for "tags" in the tensorflow/.../BUILD files for a concrete example of
  how to add tags to a target

  Note that
  - `no_rocm_v2` tag is for tests that pass when run with TF 1.X, but fail when run with TF 2.X
  - `no_cuda` tag is for "our" consumption only, it should not be upstreamed.
  - `no_gpu` tag is used to indicate which tests are excluded from GPU CI in the
    upstream repo. We should not add this tag to any tests.

  If you skip tests in the weekly sync, leave a comment in the format `TODO(rocm): weekly-sync YY-MM-DD` so we can enable it later once it's fixed upstream.

- Document the list of excluded tests amending the commit.
  Also update [this  Excel spreadsheet](https://amdcloud-my.sharepoint.com/:x:/r/personal/deven_amd_com/Documents/TF%20CI%20Unit%20Test%20Status.xlsx?d=w42bd3e2e76534209bd0438aa92857fa6&csf=1&e=5zpGPh)

- Push to the working branch once again to let the pull request be tested
  again. Repeat the process until we see a green check mark on the PR.

# Apply tags and merge

```
git checkout develop-upstream
git tag merge-YYMMDD-prev
git push --tags
```

- Go to the pull request on github. Hit merge.

```
git pull
git tag merge-YYMMDD
git push --tags
```

## Sync the upstream branch

```
git checkout upstream
git merge upstream/master
git push origin upstream
```

Upon reaching here, the process is now complete.

# Tips for debugging

Bisecting between two weekly syncs is often necessary for debugging. The process should look like this:
* Check out `merge-YYMMDD-prev` tag
* Select a commit in the middle of the sync (between `merge-YYMMDD-prev` and `merge-YYMMDD`)
* Merge to that commit point (i.e. perform a "mini sync")
* Resolve merge conflicts based on what was done in that weekly sync. In case of build braking, cherry-pick neccessary hot-fixes listed here: [XLA hotfix track](https://confluence.amd.com/display/~cchen104/XLA+hotfix+track)
* Test
* Repeat until the required commit is found
