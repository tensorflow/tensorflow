"""Tests for distutils.command.build_scripts."""

import os
import unittest

from distutils.command.build_scripts import build_scripts
from distutils.core import Distribution

from distutils.tests import support


class BuildScriptsTestCase(support.TempdirManager,
                           support.LoggingSilencer,
                           unittest.TestCase):

    def test_default_settings(self):
        cmd = self.get_build_scripts_cmd("/foo/bar", [])
        self.assert_(not cmd.force)
        self.assert_(cmd.build_dir is None)

        cmd.finalize_options()

        self.assert_(cmd.force)
        self.assertEqual(cmd.build_dir, "/foo/bar")

    def test_build(self):
        source = self.mkdtemp()
        target = self.mkdtemp()
        expected = self.write_sample_scripts(source)

        cmd = self.get_build_scripts_cmd(target,
                                         [os.path.join(source, fn)
                                          for fn in expected])
        cmd.finalize_options()
        cmd.run()

        built = os.listdir(target)
        for name in expected:
            self.assert_(name in built)

    def get_build_scripts_cmd(self, target, scripts):
        import sys
        dist = Distribution()
        dist.scripts = scripts
        dist.command_obj["build"] = support.DummyCommand(
            build_scripts=target,
            force=1,
            executable=sys.executable
            )
        return build_scripts(dist)

    def write_sample_scripts(self, dir):
        expected = []
        expected.append("script1.py")
        self.write_script(dir, "script1.py",
                          ("#! /usr/bin/env python2.3\n"
                           "# bogus script w/ Python sh-bang\n"
                           "pass\n"))
        expected.append("script2.py")
        self.write_script(dir, "script2.py",
                          ("#!/usr/bin/python\n"
                           "# bogus script w/ Python sh-bang\n"
                           "pass\n"))
        expected.append("shell.sh")
        self.write_script(dir, "shell.sh",
                          ("#!/bin/sh\n"
                           "# bogus shell script w/ sh-bang\n"
                           "exit 0\n"))
        return expected

    def write_script(self, dir, name, text):
        f = open(os.path.join(dir, name), "w")
        f.write(text)
        f.close()


def test_suite():
    return unittest.makeSuite(BuildScriptsTestCase)

if __name__ == "__main__":
    unittest.main(defaultTest="test_suite")
