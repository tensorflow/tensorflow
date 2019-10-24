#! /usr/bin/env python
# encoding: utf-8
# Thomas Nagy, 2015 (ita)

"""
Override the build commands to write empty files.
This is useful for profiling and evaluating the Python overhead.

To use::

    def build(bld):
        ...
        bld.load('nobuild')

"""

from waflib import Task
def build(bld):
	def run(self):
		for x in self.outputs:
			x.write('')
	for (name, cls) in Task.classes.items():
		cls.run = run

