#!/usr/bin/env python
# encoding: utf-8
# Thomas Nagy, 2015 (ita)

"""
Execute tasks through strace to obtain dependencies after the process is run. This
scheme is similar to that of the Fabricate script.

To use::

  def configure(conf):
     conf.load('strace')

WARNING:
* This will not work when advanced scanners are needed (qt4/qt5)
* The overhead of running 'strace' is significant (56s -> 1m29s)
* It will not work on Windows :-)
"""

import os, re, threading
from waflib import Task, Logs, Utils

#TRACECALLS = 'trace=access,chdir,clone,creat,execve,exit_group,fork,lstat,lstat64,mkdir,open,rename,stat,stat64,symlink,vfork'
TRACECALLS = 'trace=process,file'

BANNED = ('/tmp', '/proc', '/sys', '/dev')

s_process = r'(?:clone|fork|vfork)\(.*?(?P<npid>\d+)'
s_file = r'(?P<call>\w+)\("(?P<path>([^"\\]|\\.)*)"(.*)'
re_lines = re.compile(r'^(?P<pid>\d+)\s+(?:(?:%s)|(?:%s))\r*$' % (s_file, s_process), re.IGNORECASE | re.MULTILINE)
strace_lock = threading.Lock()

def configure(conf):
	conf.find_program('strace')

def task_method(func):
	# Decorator function to bind/replace methods on the base Task class
	#
	# The methods Task.exec_command and Task.sig_implicit_deps already exists and are rarely overridden
	# we thus expect that we are the only ones doing this
	try:
		setattr(Task.Task, 'nostrace_%s' % func.__name__, getattr(Task.Task, func.__name__))
	except AttributeError:
		pass
	setattr(Task.Task, func.__name__, func)
	return func

@task_method
def get_strace_file(self):
	try:
		return self.strace_file
	except AttributeError:
		pass

	if self.outputs:
		ret = self.outputs[0].abspath() + '.strace'
	else:
		ret = '%s%s%d%s' % (self.generator.bld.bldnode.abspath(), os.sep, id(self), '.strace')
	self.strace_file = ret
	return ret

@task_method
def get_strace_args(self):
	return (self.env.STRACE or ['strace']) + ['-e', TRACECALLS, '-f', '-o', self.get_strace_file()]

@task_method
def exec_command(self, cmd, **kw):
	bld = self.generator.bld
	if not 'cwd' in kw:
		kw['cwd'] = self.get_cwd()

	args = self.get_strace_args()
	fname = self.get_strace_file()
	if isinstance(cmd, list):
		cmd = args + cmd
	else:
		cmd = '%s %s' % (' '.join(args), cmd)

	try:
		ret = bld.exec_command(cmd, **kw)
	finally:
		if not ret:
			self.parse_strace_deps(fname, kw['cwd'])
	return ret

@task_method
def sig_implicit_deps(self):
	# bypass the scanner functions
	return

@task_method
def parse_strace_deps(self, path, cwd):
	# uncomment the following line to disable the dependencies and force a file scan
	# return
	try:
		cnt = Utils.readf(path)
	finally:
		try:
			os.remove(path)
		except OSError:
			pass

	if not isinstance(cwd, str):
		cwd = cwd.abspath()

	nodes = []
	bld = self.generator.bld
	try:
		cache = bld.strace_cache
	except AttributeError:
		cache = bld.strace_cache = {}

	# chdir and relative paths
	pid_to_cwd = {}

	global BANNED
	done = set()
	for m in re.finditer(re_lines, cnt):
		# scraping the output of strace
		pid = m.group('pid')
		if m.group('npid'):
			npid = m.group('npid')
			pid_to_cwd[npid] = pid_to_cwd.get(pid, cwd)
			continue

		p = m.group('path').replace('\\"', '"')

		if p == '.' or m.group().find('= -1 ENOENT') > -1:
			# just to speed it up a bit
			continue

		if not os.path.isabs(p):
			p = os.path.join(pid_to_cwd.get(pid, cwd), p)

		call = m.group('call')
		if call == 'chdir':
			pid_to_cwd[pid] = p
			continue

		if p in done:
			continue
		done.add(p)

		for x in BANNED:
			if p.startswith(x):
				break
		else:
			if p.endswith('/') or os.path.isdir(p):
				continue

			try:
				node = cache[p]
			except KeyError:
				strace_lock.acquire()
				try:
					cache[p] = node = bld.root.find_node(p)
					if not node:
						continue
				finally:
					strace_lock.release()
			nodes.append(node)

	# record the dependencies then force the task signature recalculation for next time
	if Logs.verbose:
		Logs.debug('deps: real scanner for %r returned %r', self, nodes)
	bld = self.generator.bld
	bld.node_deps[self.uid()] = nodes
	bld.raw_deps[self.uid()] = []
	try:
		del self.cache_sig
	except AttributeError:
		pass
	self.signature()

