#!/bin/bash

set -x  # print commands as they are executed
set -e  # fail and exit on any command erroring

./oss_scripts/configure.sh

# Verify correct version of Bazel
installed_bazel_version=$(bazel version | grep label | sed -e 's/.*: //')
tf_bazel_version=$(cat .bazelversion)
if [ "$installed_bazel_version" != "$tf_bazel_version" ]; then
  echo "Incorrect version of Bazel installed."
  echo "Version $tf_bazel_version should be installed, but found version ${installed_bazel_version}."
  echo "Run oss_scripts/install_bazel.sh or manually install the correct version."
  exit 1
fi

# Set tensorflow version
if [[ $osname != "Darwin" ]] || [[ ! $(sysctl -n machdep.cpu.brand_string) =~ "Apple" ]]; then
  source oss_scripts/prepare_tf_dep.sh
fi

bazel test --test_output=errors --keep_going --jobs=1 tensorflow_text:all
