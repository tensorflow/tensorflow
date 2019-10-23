#! /usr/bin/env python
# encoding: utf-8
# Thomas Nagy, 2010 (ita)

"""
This tool modifies the task signature scheme to store and obtain
information about the task execution (why it must run, etc)::

	def configure(conf):
		conf.load('why')

After adding the tool, a full rebuild is necessary:
waf clean build --zones=task
"""

from waflib import Task, Utils, Logs, Errors

def signature(self):
	# compute the result one time, and suppose the scan_signature will give the good result
	try:
		return self.cache_sig
	except AttributeError:
		pass

	self.m = Utils.md5()
	self.m.update(self.hcode)
	id_sig = self.m.digest()

	# explicit deps
	self.m = Utils.md5()
	self.sig_explicit_deps()
	exp_sig = self.m.digest()

	# env vars
	self.m = Utils.md5()
	self.sig_vars()
	var_sig = self.m.digest()

	# implicit deps / scanner results
	self.m = Utils.md5()
	if self.scan:
		try:
			self.sig_implicit_deps()
		except Errors.TaskRescan:
			return self.signature()
	impl_sig = self.m.digest()

	ret = self.cache_sig = impl_sig + id_sig + exp_sig + var_sig
	return ret


Task.Task.signature = signature

old = Task.Task.runnable_status
def runnable_status(self):
	ret = old(self)
	if ret == Task.RUN_ME:
		try:
			old_sigs = self.generator.bld.task_sigs[self.uid()]
		except (KeyError, AttributeError):
			Logs.debug("task: task must run as no previous signature exists")
		else:
			new_sigs = self.cache_sig
			def v(x):
				return Utils.to_hex(x)

			Logs.debug('Task %r', self)
			msgs = ['* Implicit or scanner dependency', '* Task code', '* Source file, explicit or manual dependency', '* Configuration data variable']
			tmp = 'task: -> %s: %s %s'
			for x in range(len(msgs)):
				l = len(Utils.SIG_NIL)
				a = new_sigs[x*l : (x+1)*l]
				b = old_sigs[x*l : (x+1)*l]
				if (a != b):
					Logs.debug(tmp, msgs[x].ljust(35), v(a), v(b))
	return ret
Task.Task.runnable_status = runnable_status

