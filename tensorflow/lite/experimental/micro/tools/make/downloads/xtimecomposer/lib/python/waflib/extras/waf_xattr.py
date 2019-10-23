#! /usr/bin/env python
# encoding: utf-8

"""
Use extended attributes instead of database files

1. Input files will be made writable
2. This is only for systems providing extended filesystem attributes
3. By default, hashes are calculated only if timestamp/size change (HASH_CACHE below)
4. The module enables "deep_inputs" on all tasks by propagating task signatures
5. This module also skips task signature comparisons for task code changes due to point 4.
6. This module is for Python3/Linux only, but it could be extended to Python2/other systems
   using the xattr library
7. For projects in which tasks always declare output files, it should be possible to
   store the rest of build context attributes on output files (imp_sigs, raw_deps and node_deps)
   but this is not done here

On a simple C++ project benchmark, the variations before and after adding waf_xattr.py were observed:
total build time: 20s -> 22s
no-op build time: 2.4s -> 1.8s
pickle file size: 2.9MB -> 2.6MB
"""

import os
from waflib import Logs, Node, Task, Utils, Errors
from waflib.Task import SKIP_ME, RUN_ME, CANCEL_ME, ASK_LATER, SKIPPED, MISSING

HASH_CACHE = True
SIG_VAR = 'user.waf.sig'
SEP = ','.encode()
TEMPLATE = '%b%d,%d'.encode()

try:
	PermissionError
except NameError:
	PermissionError = IOError

def getxattr(self):
	return os.getxattr(self.abspath(), SIG_VAR)

def setxattr(self, val):
	os.setxattr(self.abspath(), SIG_VAR, val)

def h_file(self):
	try:
		ret = getxattr(self)
	except OSError:
		if HASH_CACHE:
			st = os.stat(self.abspath())
			mtime = st.st_mtime
			size = st.st_size
	else:
		if len(ret) == 16:
			# for build directory files
			return ret

		if HASH_CACHE:
			# check if timestamp and mtime match to avoid re-hashing
			st = os.stat(self.abspath())
			mtime, size = ret[16:].split(SEP)
			if int(1000 * st.st_mtime) == int(mtime) and st.st_size == int(size):
				return ret[:16]

	ret = Utils.h_file(self.abspath())
	if HASH_CACHE:
		val = TEMPLATE % (ret, int(1000 * st.st_mtime), int(st.st_size))
		try:
			setxattr(self, val)
		except PermissionError:
			os.chmod(self.abspath(), st.st_mode | 128)
			setxattr(self, val)
	return ret

def runnable_status(self):
	bld = self.generator.bld
	if bld.is_install < 0:
		return SKIP_ME

	for t in self.run_after:
		if not t.hasrun:
			return ASK_LATER
		elif t.hasrun < SKIPPED:
			# a dependency has an error
			return CANCEL_ME

	# first compute the signature
	try:
		new_sig = self.signature()
	except Errors.TaskNotReady:
		return ASK_LATER

	if not self.outputs:
		# compare the signature to a signature computed previously
		# this part is only for tasks with no output files
		key = self.uid()
		try:
			prev_sig = bld.task_sigs[key]
		except KeyError:
			Logs.debug('task: task %r must run: it was never run before or the task code changed', self)
			return RUN_ME
		if new_sig != prev_sig:
			Logs.debug('task: task %r must run: the task signature changed', self)
			return RUN_ME

	# compare the signatures of the outputs to make a decision
	for node in self.outputs:
		try:
			sig = node.h_file()
		except EnvironmentError:
			Logs.debug('task: task %r must run: an output node does not exist', self)
			return RUN_ME
		if sig != new_sig:
			Logs.debug('task: task %r must run: an output node is stale', self)
			return RUN_ME

	return (self.always_run and RUN_ME) or SKIP_ME

def post_run(self):
	bld = self.generator.bld
	sig = self.signature()
	for node in self.outputs:
		if not node.exists():
			self.hasrun = MISSING
			self.err_msg = '-> missing file: %r' % node.abspath()
			raise Errors.WafError(self.err_msg)
		os.setxattr(node.abspath(), 'user.waf.sig', sig)
	if not self.outputs:
		# only for task with no outputs
		bld.task_sigs[self.uid()] = sig
	if not self.keep_last_cmd:
		try:
			del self.last_cmd
		except AttributeError:
			pass

try:
	os.getxattr
except AttributeError:
	pass
else:
	h_file.__doc__ = Node.Node.h_file.__doc__

	# keep file hashes as file attributes
	Node.Node.h_file = h_file

	# enable "deep_inputs" on all tasks
	Task.Task.runnable_status = runnable_status
	Task.Task.post_run = post_run
	Task.Task.sig_deep_inputs = Utils.nada

