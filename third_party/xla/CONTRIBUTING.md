# How to Contribute

We'd love to accept your patches and contributions to this project!

## Before you begin

### Sign our Contributor License Agreement

Contributions to this project must be accompanied by a
[Contributor License Agreement](https://cla.developers.google.com/about) (CLA).
You (or your employer) retain the copyright to your contribution; this simply
gives us permission to use and redistribute your contributions as part of the
project.

If you or your current employer have already signed the Google CLA (even if it
was for a different project), you probably don't need to do it again.

Visit <https://cla.developers.google.com/> to see your current agreements or to
sign a new one.

### Review our Community Guidelines

This project follows
[Tensorflow's Open Source Community Guidelines](https://github.com/tensorflow/tensorflow/blob/master/CODE_OF_CONDUCT.md).

## Contribution process

### Developer Guide

For a guide on how to setup a dev environment for XLA, please refer to the
[XLA developer guide](https://github.com/openxla/xla/blob/main/docs/developer_guide.md).

### Code Reviews

All submissions, including submissions by project members, require review. We
use GitHub pull requests for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more
information on using pull requests. Please ensure that your PR passes CI before
requesting review!

### Notes on Testing and CI

Before a PR is merged, it will undergo internal testing that uses code internal
to Google. This can potentially add extra steps to the review process if there
are failures on internal tests that our public CI doesn't catch. The Googler
sheparding your PR will communicate any internal test failures and describe
what needs to be fixed.

We are actively working on increasing the number of tests run on Github!

### Copybara quirks
There are some oddities you may see while contributing, please see [this file](docs/copybara.md).
