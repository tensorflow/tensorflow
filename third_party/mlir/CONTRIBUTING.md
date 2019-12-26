# How to Contribute

Everyone is welcome to contribute to MLIR. There are several ways of getting involved and contributing including reporting bugs, improving documentation, writing models or tutorials. 

Please read our [Code of Conduct](https://github.com/tensorflow/tensorflow/blob/master/CODE_OF_CONDUCT.md) before participating.

## Community Guidelines

This project follows [Google's Open Source Community
Guidelines](https://opensource.google.com/conduct/).

## How to become a contributor and submit your own code

### Contributor License Agreements

We'd love to accept your patches! Before we can take them, please fill out either the individual or corporate Contributor License Agreement (CLA).

* If you are an individual writing original source code and you're sure you own the intellectual property, then you'll need to sign an [individual CLA](https://code.google.com/legal/individual-cla-v1.0.html).
  * If you work for a company that wants to allow you to contribute your work, then you'll need to sign a [corporate CLA](https://code.google.com/legal/corporate-cla-v1.0.html).

Follow either of the two links above to access the appropriate CLA and instructions for how to sign and return it. Once we receive it, we'll be able to accept your pull requests.

***NOTE***: Only original source code from you and other people that have signed the CLA can be accepted into the main repository.

### Contributing code

If you have improvements to MLIR, send us your pull requests! For those
just getting started, GitHub has a [howto](https://help.github.com/articles/using-pull-requests/).

MLIR team members will be assigned to review your pull requests. Once the pull requests are approved and pass continuous integration checks, a team member will merge your pull request submitted to our internal repository. After the change has been submitted internally, your pull request will be merged automatically on GitHub.

If you want to contribute, start working through the MLIR codebase, navigate to [Github "issues" tab](https://github.com/tensorflow/mlir/issues) and start looking through interesting issues. If you decide to start on an issue, leave a comment so that other people know that you're working on it. If you want to help out, but not alone, use the issue comment thread to coordinate.

### Contribution guidelines and standards

*   Read the [developer guide](g3doc/DeveloperGuide.md).
*   Ensure that you use the correct license. Examples are provided below.
*   Include tests when you contribute new features, as they help to a)
    prove that your code works correctly, and b) guard against future breaking
    changes to lower the maintenance cost.
*   Bug fixes also generally require tests, because the presence of bugs
    usually indicates insufficient test coverage.

#### License

Include a license at the top of new files.

* [C/C++ license example](https://github.com/tensorflow/mlir/blob/master/examples/toy/Ch1/toyc.cpp)
