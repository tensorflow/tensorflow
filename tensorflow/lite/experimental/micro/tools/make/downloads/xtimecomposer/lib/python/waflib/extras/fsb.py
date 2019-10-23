#! /usr/bin/env python
# encoding: utf-8
# Thomas Nagy, 2011 (ita)

"""
Fully sequential builds

The previous tasks from task generators are re-processed, and this may lead to speed issues
Yet, if you are using this, speed is probably a minor concern
"""

from waflib import Build

def options(opt):
	pass

def configure(conf):
	pass

class FSBContext(Build.BuildContext):
	def __call__(self, *k, **kw):
		ret = Build.BuildContext.__call__(self, *k, **kw)

		# evaluate the results immediately
		Build.BuildContext.compile(self)

		return ret

	def compile(self):
		pass

