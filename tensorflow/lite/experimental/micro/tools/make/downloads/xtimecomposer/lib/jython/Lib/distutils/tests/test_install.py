"""Tests for distutils.command.install."""

import os
import unittest

from distutils.command.install import install
from distutils.core import Distribution

from distutils.tests import support


class InstallTestCase(support.TempdirManager, unittest.TestCase):

    def test_home_installation_scheme(self):
        # This ensure two things:
        # - that --home generates the desired set of directory names
        # - test --home is supported on all platforms
        builddir = self.mkdtemp()
        destination = os.path.join(builddir, "installation")

        dist = Distribution({"name": "foopkg"})
        # script_name need not exist, it just need to be initialized
        dist.script_name = os.path.join(builddir, "setup.py")
        dist.command_obj["build"] = support.DummyCommand(
            build_base=builddir,
            build_lib=os.path.join(builddir, "lib"),
            )

        cmd = install(dist)
        cmd.home = destination
        cmd.ensure_finalized()

        self.assertEqual(cmd.install_base, destination)
        self.assertEqual(cmd.install_platbase, destination)

        def check_path(got, expected):
            got = os.path.normpath(got)
            expected = os.path.normpath(expected)
            self.assertEqual(got, expected)

        libdir = os.path.join(destination, "lib", "python")
        check_path(cmd.install_lib, libdir)
        check_path(cmd.install_platlib, libdir)
        check_path(cmd.install_purelib, libdir)
        check_path(cmd.install_headers,
                   os.path.join(destination, "include", "python", "foopkg"))
        check_path(cmd.install_scripts, os.path.join(destination, "bin"))
        check_path(cmd.install_data, destination)


def test_suite():
    return unittest.makeSuite(InstallTestCase)

if __name__ == "__main__":
    unittest.main(defaultTest="test_suite")
