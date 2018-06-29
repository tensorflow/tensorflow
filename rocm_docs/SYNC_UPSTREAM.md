How to synchonize TensorFlow ROCm port with upstream TensorFlow
===============================================================
This article shows the step to synchronize TensorFlow with upstream. The
process is currently carried out manually but it could and should be automated.

Useful github remote repositories (For reference)
-------------------------------------------------
git locations of repositories used in the merge process are:

- TensorFlow ROCm port:
  - URL: https://github.com/ROCmSoftwarePlatform/tensorflow-upstream.git
  - branch: develop-upstream
- TensorFlow upstream:
  - URL: https://github.com/tensorflow/tensorflow.git
  - branch: master

Step-by-step Merge Process
--------------------------

### Get TensorFlow ROCm port
- git clone git@github.com:ROCmSoftwarePlatform/tensorflow-upstream.git
- git checkout develop-upstream
- git pull

### Add remotes
- git remote add upstream git@github.com:tensorflow/tensorflow.git
- git fetch upstream

### Create a working branch to track merge process
- git checkout -b develop-upstream-sync-YYMMDD
- git merge upstream/master --no-edit
- Resolve any merge conflicts encountered here

### Notice on common merge conflicts

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

### Build merged TensorFlow
- Build with either `build` or `build_python3` script. Make sure everything
  builds fine and Python PIP whl package can be built.

### Push and Create Pull Request for TensorFlow ROCm port
- git push --set-upstream origin develop-upstream-sync-YYMMDD
- Create a pull request to merge `develop-upstream-sync-YYMMDD` into
  https://github.com/ROCmSoftwarePlatform/tensorflow-upstream.git on branch
  `develop-upstream`.
- Wait until the pull request has finished validation on Jenkins CI.

### Update unit test whitelist if necessary
- The first Jenkins CI job might result in failure due to additional failed
  unit tests introduced from new commits upstream. Examine the reason behind
  those failed cases.
- In case those cases can't be easily fixed, modify:

  `tensorflow/tools/ci_build/linux/rocm/run_py3_core.sh`

  and append those failed cases to the test script, prefixed with "-" so they
  will be skipped in Jenkins CI process.

- Document the list of excluded tests amending the commit. Also update:

  [URL to be supplied by JeffP]

- Push to the working branch once again to let the pull request be tested
  again. Repeat the process until we see a green check mark on the PR.

### Apply tags and merge

- git checkout develop-upstream
- git tag merge-YYMMDD-prev
- git push --tags

- Go to the pull request on github. Hit merge.

- git pull
- git tag merge-YYMMDD
- git push --tags

Upon reaching here, the process is now complete.
