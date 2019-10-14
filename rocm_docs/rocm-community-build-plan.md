# Overview

Our goal is to add ROCm to TensorFlow's list of Community Supported Builds.  To accomplish this, we need to provide a plan on how to set up the builds, how to run the tests, and how we'll support the process.  

# Build infrastructure

We currently use Jenkins CI to build and test all of the commits to our ROCm fork of TensorFlow.  

Going forward, we'll also use it to build and test the upstream TensorFlow repo.  We have significantly increased our CI worker node pool in order to be able to handle the increase in the number of jobs associated with the upstream repo.  

# Build+test plan

We will run three types of builds:

- Continuous master branch build+test cycles:  watching for regressions in new commits
- Nightly artifact builds:  building and testing WHL files every night
- Release-specific artifact builds:  building and testing WHL files every release

For the master branch builds, we'll use a webhook to notify our CI setup when a new commit exists, and that will trigger a new build+test cycle.  Again, we have multiple CI worker nodes, so we should be able to minimize the number of commits per build.  For testing, we typically run ci_build's `rocm/run_py3_core.sh` script.

Nightly WHL builds will be scheduled every night using Jenkins.  Release-specific WHL builds will be manually scheduled whenever a new TensorFlow release occurs.  Each of these builds will also be checked with the `run_py3_core.sh` script.  

# Ongoing support

The community supported builds for ROCm will be maintained by the following team members:  

- Jeff Poznanovic (`parallelo`, <jeffrey.poznanovic@amd.com>)
- Peng Sun (`sunway513`, <peng.sun@amd.com>)
- Jack Chung (`whchung`, <whchung@amd.com>)

Our installation docs can currently be found here:  

- [ROCm details](https://rocm.github.io/ROCmInstall.html)
- [TensorFlow details](https://github.com/ROCmSoftwarePlatform/tensorflow-upstream/blob/develop-upstream/rocm_docs/tensorflow-install-basic.md)