#!/bin/bash
set -e  # fail and exit on any command erroring

if (which python3) | grep -q "python3"; then
  installed_python="python3"
elif (which python) | grep -q "python"; then
  installed_python="python"
fi

ext=""
osname="$(uname -s | tr 'A-Z' 'a-z')"
if [[ "${osname}" == "darwin" ]]; then
  ext='""'
fi

# Update setup.nightly.py with current tf version.
tf_version=$(bazel run //oss_scripts/pip_package:tensorflow_build_info -- version)
echo "Updating setup.nightly.py to version $tf_version"
sed -i $ext "s/project_version = '.*'/project_version = '${tf_version}'/" oss_scripts/pip_package/setup.nightly.py
# Update __version__.
echo "Updating __init__.py to version $tf_version"
sed -i $ext "s/__version__ = .*\$/__version__ = \"${tf_version}\"/" tensorflow_text/__init__.py

# Get git commit sha of installed tensorflow.
echo "Querying commit SHA"
short_commit_sha=$(bazel run  //oss_scripts/pip_package:tensorflow_build_info -- git_version)
if [[ "$short_commit_sha" == "unknown" ]]; then
  # Some nightly builds report "unknown" for tf.__git_version.
  echo 'TF git version "unknown", assuming nightly.'
  commit_slug='nightly'
else
  if [[ "${osname}" == "darwin" ]]; then
    short_commit_sha=$(echo $short_commit_sha | perl -nle 'print $& while m{(?<=-g)[0-9a-f]*$}g')
  else
    short_commit_sha=$(echo $short_commit_sha | grep -oP '(?<=-g)[0-9a-f]*$')
  fi
  echo "Found tensorflow commit sha: $short_commit_sha"
  commit_slug=$(curl -s "https://api.github.com/repos/tensorflow/tensorflow/commits/$short_commit_sha" | grep "sha" | head -n 1 | cut -d '"' -f 4)
fi

# Update TF dependency to installed tensorflow.
echo "Updating WORKSPACE file to use TensorFlow commit $commit_slug"
sed -E -i $ext "s/strip_prefix = \"tensorflow-.+\",/strip_prefix = \"tensorflow-${commit_slug}\",/" WORKSPACE
sed -E -i $ext "s|\"https://github.com/tensorflow/tensorflow/archive/.+\.zip\"|\"https://github.com/tensorflow/tensorflow/archive/${commit_slug}.zip\"|" WORKSPACE
prev_shasum=$(grep -B 1 -e "strip_prefix.*tensorflow-" WORKSPACE | grep "sha256" | cut -d'"' -f2)
sed -i $ext "s/sha256 = \"${prev_shasum}\",//" WORKSPACE
