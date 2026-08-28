"""Packages the built plugin so that `import tensorflow` finds it.

TensorFlow scans `site-packages/tensorflow-plugins` at import and loads every
shared object in it as a PluggableDevice, so installing the dylib there is the
whole of what installation means.

The dylib is built against the TensorFlow of the interpreter doing the
install, which is why this runs make rather than shipping a prebuilt binary: a
plugin built against a different TensorFlow is not portable to this one.
"""

import subprocess
import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py

HERE = Path(__file__).resolve().parent


class BuildPlugin(build_py):

  def run(self):
    subprocess.check_call(["make", f"PYTHON={sys.executable}"], cwd=HERE)
    # After the declared files are copied, so this is not overwritten.
    super().run()
    target = Path(self.build_lib) / "tensorflow-plugins"
    target.mkdir(parents=True, exist_ok=True)
    self.copy_file(str(HERE / "build" / "libmetal_plugin.dylib"),
                   str(target / "libmetal_plugin.dylib"))


setup(
    name="tensorflow-metal-plugin",
    version="0.1.0",
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
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: GPU",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python :: 3",
    ],
)
