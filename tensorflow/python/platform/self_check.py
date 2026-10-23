# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Platform-specific code for checking the integrity of the TensorFlow build."""

import os
import struct

MSVCP_DLL_NAMES = "msvcp_dll_names"

# Distributions that all unpack into the same `site-packages/tensorflow`
# directory. On Windows `tensorflow` is a meta-package that requires
# `tensorflow-intel` at an identical version, so their coexistence is expected;
# only a version mismatch between any two of these indicates that files from
# different releases have been mixed in a single directory.
_CONFLICTING_DISTRIBUTIONS = (
    "tensorflow",
    "tensorflow-cpu",
    "tensorflow-gpu",
    "tensorflow-intel",
    "tf-nightly",
    "tf-nightly-cpu",
    "tf-nightly-gpu",
    "tf-nightly-intel",
)

try:
  from tensorflow.python.platform import build_info
except ImportError as exc:
  raise ImportError(
      "Could not import tensorflow. Do not import"
      " tensorflow from its source directory; change directory to outside the"
      " TensorFlow source tree, and relaunch your Python interpreter from"
      " there."
  ) from exc


def _is_windows():
  """Returns True if running on Windows."""
  return os.name == "nt"


def _installed_tensorflow_distributions():
  """Returns a {distribution name: version} map of installed TensorFlow dists.

  Only the distributions in `_CONFLICTING_DISTRIBUTIONS` are considered, and
  distributions that are not installed are omitted.
  """
  import importlib.metadata  # pylint: disable=g-import-not-at-top

  installed = {}
  for name in _CONFLICTING_DISTRIBUTIONS:
    try:
      installed[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
      pass
    except Exception:  # pylint: disable=broad-except
      # Package metadata can be missing or malformed in unusual environments.
      # This is only a diagnostic aid, so never let it mask the real error.
      pass
  return installed


def _check_conflicting_installs():
  """Raises if TensorFlow distributions of differing versions are installed.

  These distributions all unpack into the same `site-packages/tensorflow`
  directory, so installing more than one at differing versions leaves shared
  libraries from mismatched releases on disk. Loading the native runtime then
  fails with an error that gives no indication of the real cause.

  Raises:
    ImportError: If two or more of the distributions are installed at differing
      versions.
  """
  installed = _installed_tensorflow_distributions()
  if len(set(installed.values())) > 1:
    raise ImportError(
        "Found conflicting TensorFlow installations in this environment: %s. "
        "These distributions install into the same directory, so mixing "
        "versions leaves shared libraries from different releases on disk and "
        "the native TensorFlow runtime will fail to load. Uninstall all of "
        "them and reinstall exactly one. Note that some packages (for example "
        "tf-keras) depend on `tensorflow`, which can pull it in alongside a "
        "`tensorflow-cpu` that you installed directly."
        % ", ".join(
            "%s %s" % (name, version)
            for name, version in sorted(installed.items())
        )
    )


def _python_bitness():
  """Returns the pointer width of the running interpreter, in bits."""
  return struct.calcsize("P") * 8


def _check_python_bitness():
  """Raises if the interpreter is not 64-bit.

  Raises:
    ImportError: If running under a 32-bit Python interpreter, for which no
      TensorFlow wheels are published.
  """
  bitness = _python_bitness()
  if bitness != 64:
    raise ImportError(
        "TensorFlow requires a 64-bit Python interpreter, but this "
        "interpreter is %d-bit. Reinstall the 64-bit build of Python and "
        "reinstall TensorFlow into it." % bitness
    )


def preload_check():
  """Raises an exception if the environment is not correctly configured.

  Raises:
    ImportError: If the check detects that the environment is not correctly
      configured, and attempting to load the TensorFlow runtime will fail.
  """
  if _is_windows():
    _check_python_bitness()
    _check_conflicting_installs()

    # Attempt to load any DLLs that the Python extension depends on before
    # we load the Python extension, so that we can raise an actionable error
    # message if they are not found.
    if MSVCP_DLL_NAMES in build_info.build_info:
      missing = []
      for dll_name in build_info.build_info[MSVCP_DLL_NAMES].split(","):
        found = False
        for path_dir in os.environ.get("PATH", "").split(";"):
          if os.path.isfile(os.path.join(path_dir, dll_name)):
            found = True
            break
        if not found:
          missing.append(dll_name)
      if missing:
        raise ImportError(
            "Could not find the DLL(s) %r. TensorFlow requires that these DLLs "
            "be installed in a directory that is named in your %%PATH%% "
            "environment variable. You may install these DLLs by downloading "
            '"Microsoft C++ Redistributable for Visual Studio 2015, 2017 and '
            '2019" for your platform from this URL: '
            "https://support.microsoft.com/help/2977003/the-latest-supported-visual-c-downloads"
            % " or ".join(missing)
        )
  else:
    # Load a library that performs CPU feature guard checking.  Doing this here
    # as a preload check makes it more likely that we detect any CPU feature
    # incompatibilities before we trigger them (which would typically result in
    # SIGILL).
    from tensorflow.python.platform import _pywrap_cpu_feature_guard

    _pywrap_cpu_feature_guard.InfoAboutUnusedCPUFeatures()
