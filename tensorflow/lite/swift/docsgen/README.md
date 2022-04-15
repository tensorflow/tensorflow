# Generating docs for TensorFlowLiteSwift

Documentation is generated via [Jazzy](https://github.com/realm/jazzy) (Googlers
see cr/363774564 for more details).

To browse the Swift reference documentation visit
https://www.tensorflow.org/lite/api_docs/swift.

This directory contains a dummy Xcode project for generating documentation for
TensorFlowLiteSwift via Jazzy, an open-source tool that hooks into Xcode's
build tooling to parse doc comments. Unfortunately, TensorFlowLiteSwift is not
primarily developed via xcodebuild, so the docs build can potentially become
decoupled from upstream TensorFlowLiteSwift development.

Known issues:
- Every new file added to TensorFlowLiteSwift's BUILD must also manually be
  added to this Xcode project.
- This project (and the resulting documentation) does not split types  by
  module, so there's no way to tell from looking at the generated documentation
  which modules must be included in order to access a specific type.
- The TensorFlowLiteC dependency is included in binary form, contributing
  significant bloat to the git repository since each binary contains unused
  architecture slices.

To generate documentation outside of Google, run jazzy as you would on any other
Swift module:

```
jazzy \
  --swift-build-tool xcodebuild \
  --module "TensorFlowLiteSwift" \
  --author "The TensorFlow Authors" \
  --sdk iphoneos \
```
