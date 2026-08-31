# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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
"""Packages the built plugin so that `import tensorflow` finds it.

TensorFlow scans `site-packages/tensorflow-plugins` at import and loads every
shared object in it as a PluggableDevice, so installing the dylib there is the
whole of what installation means.

The dylib is built against the TensorFlow of the interpreter doing the
install, which is why this runs make rather than shipping a prebuilt binary: a
plugin built against a different TensorFlow is not portable to this one.
"""

import os
import subprocess
import sys
import sysconfig
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.dist import Distribution

HERE = Path(__file__).resolve().parent


def tensorflow_paths():
  """Where TensorFlow's headers and libraries are.

  pip builds in an isolated environment by default, so `import tensorflow`
  fails here even though the interpreter installing the package has it. The
  interpreter is still the real one, so its own site-packages is where to
  look; importing is only the shortcut for when there is no isolation.
  """
  try:
    import tensorflow as tf  # pylint: disable=g-import-not-at-top
    return tf.sysconfig.get_include(), tf.sysconfig.get_lib()
  except ImportError:
    pass
  for key in ("purelib", "platlib"):
    root = os.path.join(sysconfig.get_paths()[key], "tensorflow")
    if os.path.isdir(os.path.join(root, "include")):
      return os.path.join(root, "include"), root
  raise SystemExit(
      "TensorFlow was not found. Install it before this plugin: a "
      "PluggableDevice is compiled against the TensorFlow it will be loaded "
      "into, so there is nothing to build against until it is there.")


class BuildPlugin(build_py):

  def run(self):
    include, lib = tensorflow_paths()
    subprocess.check_call(
        ["make", f"PYTHON={sys.executable}", f"TF_INCLUDE={include}",
         f"TF_LIB={lib}"], cwd=HERE)
    # After the declared files are copied, so this is not overwritten.
    super().run()
    target = Path(self.build_lib) / "tensorflow-plugins"
    target.mkdir(parents=True, exist_ok=True)
    self.copy_file(str(HERE / "build" / "libmetal_plugin.dylib"),
                   str(target / "libmetal_plugin.dylib"))


class BinaryDistribution(Distribution):
  """Marks the wheel as platform-specific.

  The package is pure Python by setuptools' reckoning, since the dylib is
  produced by a make rule rather than by an Extension, so the wheel would be
  tagged py3-none-any: portable, which a compiled arm64 Metal backend is very
  much not.
  """

  def has_ext_modules(self):
    return True


setup(
    name="tensorflow-metal-plugin",
    version="0.2.0",
    description="Metal GPU backend for TensorFlow on Apple silicon",
    long_description=(HERE / "README.md").read_text(),
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    python_requires=">=3.10",
    install_requires=["tensorflow>=2.16"],
    # tensorflow-plugins is the directory TensorFlow scans at import, so that
    # is where the shared object has to land. Declaring it as a package is
    # what makes the wheel carry it; data_files would not, since those install
    # relative to sys.prefix rather than into site-packages, and setuptools
    # resolves them before the build has produced anything.
    packages=["tensorflow-plugins"],
    package_data={"tensorflow-plugins": ["*.dylib"]},
    cmdclass={"build_py": BuildPlugin},
    distclass=BinaryDistribution,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: GPU",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python :: 3",
    ],
)
