#!/usr/bin/env python
# encoding: utf-8
# Philipp Bender, 2012
# Matt Clarkson, 2012

import re, os
from waflib.Task import Task
from waflib.TaskGen import extension
from waflib import Errors, Context, Logs

"""
A simple tool to integrate protocol buffers into your build system.

Example for C++:

    def configure(conf):
        conf.load('compiler_cxx cxx protoc')

    def build(bld):
        bld(
                features = 'cxx cxxprogram'
                source   = 'main.cpp file1.proto proto/file2.proto',
                includes = '. proto',
                target   = 'executable')

Example for Python:

    def configure(conf):
        conf.load('python protoc')

    def build(bld):
        bld(
                features = 'py'
                source   = 'main.py file1.proto proto/file2.proto',
                protoc_includes  = 'proto')

Example for both Python and C++ at same time:

    def configure(conf):
        conf.load('cxx python protoc')

    def build(bld):
        bld(
                features = 'cxx py'
                source   = 'file1.proto proto/file2.proto',
                protoc_includes  = 'proto')	# or includes


Example for Java:

    def options(opt):
        opt.load('java')

    def configure(conf):
        conf.load('python java protoc')
        # Here you have to point to your protobuf-java JAR and have it in classpath
        conf.env.CLASSPATH_PROTOBUF = ['protobuf-java-2.5.0.jar']

    def build(bld):
        bld(
                features = 'javac protoc',
                name = 'pbjava',
                srcdir = 'inc/ src',	# directories used by javac
                source   = ['inc/message_inc.proto', 'inc/message.proto'],
					# source is used by protoc for .proto files
                use = 'PROTOBUF',
                protoc_includes = ['inc']) # for protoc to search dependencies


Protoc includes passed via protoc_includes are either relative to the taskgen
or to the project and are searched in this order.

Include directories external to the waf project can also be passed to the
extra by using protoc_extincludes

                protoc_extincludes = ['/usr/include/pblib']


Notes when using this tool:

- protoc command line parsing is tricky.

  The generated files can be put in subfolders which depend on
  the order of the include paths.

  Try to be simple when creating task generators
  containing protoc stuff.

"""

class protoc(Task):
	run_str = '${PROTOC} ${PROTOC_FL:PROTOC_FLAGS} ${PROTOC_ST:INCPATHS} ${PROTOC_ST:PROTOC_INCPATHS} ${PROTOC_ST:PROTOC_EXTINCPATHS} ${SRC[0].bldpath()}'
	color   = 'BLUE'
	ext_out = ['.h', 'pb.cc', '.py', '.java']
	def scan(self):
		"""
		Scan .proto dependencies
		"""
		node = self.inputs[0]

		nodes = []
		names = []
		seen = []
		search_nodes = []

		if not node:
			return (nodes, names)

		if 'cxx' in self.generator.features:
			search_nodes = self.generator.includes_nodes

		if 'py' in self.generator.features or 'javac' in self.generator.features:
			for incpath in getattr(self.generator, 'protoc_includes', []):
				incpath_node = self.generator.path.find_node(incpath)
				if incpath_node:
					search_nodes.append(incpath_node)
				else:
					# Check if relative to top-level for extra tg dependencies
					incpath_node = self.generator.bld.path.find_node(incpath)
					if incpath_node:
						search_nodes.append(incpath_node)
					else:
						raise Errors.WafError('protoc: include path %r does not exist' % incpath)


		def parse_node(node):
			if node in seen:
				return
			seen.append(node)
			code = node.read().splitlines()
			for line in code:
				m = re.search(r'^import\s+"(.*)";.*(//)?.*', line)
				if m:
					dep = m.groups()[0]
					for incnode in search_nodes:
						found = incnode.find_resource(dep)
						if found:
							nodes.append(found)
							parse_node(found)
						else:
							names.append(dep)

		parse_node(node)
		# Add also dependencies path to INCPATHS so protoc will find the included file
		for deppath in nodes:
			self.env.append_unique('INCPATHS', deppath.parent.bldpath())
		return (nodes, names)

@extension('.proto')
def process_protoc(self, node):
	incdirs = []
	out_nodes = []
	protoc_flags = []

	# ensure PROTOC_FLAGS is a list; a copy is used below anyway
	self.env.PROTOC_FLAGS = self.to_list(self.env.PROTOC_FLAGS)

	if 'cxx' in self.features:
		cpp_node = node.change_ext('.pb.cc')
		hpp_node = node.change_ext('.pb.h')
		self.source.append(cpp_node)
		out_nodes.append(cpp_node)
		out_nodes.append(hpp_node)
		protoc_flags.append('--cpp_out=%s' % node.parent.get_bld().bldpath())

	if 'py' in self.features:
		py_node = node.change_ext('_pb2.py')
		self.source.append(py_node)
		out_nodes.append(py_node)
		protoc_flags.append('--python_out=%s' % node.parent.get_bld().bldpath())

	if 'javac' in self.features:
		# Make javac get also pick java code generated in build
		if not node.parent.get_bld() in self.javac_task.srcdir:
			self.javac_task.srcdir.append(node.parent.get_bld())

		protoc_flags.append('--java_out=%s' % node.parent.get_bld().bldpath())
		node.parent.get_bld().mkdir()

	tsk = self.create_task('protoc', node, out_nodes)
	tsk.env.append_value('PROTOC_FLAGS', protoc_flags)

	if 'javac' in self.features:
		self.javac_task.set_run_after(tsk)

	# Instruct protoc where to search for .proto included files.
	# For C++ standard include files dirs are used,
	# but this doesn't apply to Python for example
	for incpath in getattr(self, 'protoc_includes', []):
		incpath_node = self.path.find_node(incpath)
		if incpath_node:
			incdirs.append(incpath_node.bldpath())
		else:
			# Check if relative to top-level for extra tg dependencies
			incpath_node = self.bld.path.find_node(incpath)
			if incpath_node:
				incdirs.append(incpath_node.bldpath())
			else:
				raise Errors.WafError('protoc: include path %r does not exist' % incpath)

	tsk.env.PROTOC_INCPATHS = incdirs

	# Include paths external to the waf project (ie. shared pb repositories)
	tsk.env.PROTOC_EXTINCPATHS = getattr(self, 'protoc_extincludes', [])

	# PR2115: protoc generates output of .proto files in nested
	# directories  by canonicalizing paths. To avoid this we have to pass
	# as first include the full directory file of the .proto file
	tsk.env.prepend_value('INCPATHS', node.parent.bldpath())

	use = getattr(self, 'use', '')
	if not 'PROTOBUF' in use:
		self.use = self.to_list(use) + ['PROTOBUF']

def configure(conf):
	conf.check_cfg(package='protobuf', uselib_store='PROTOBUF', args=['--cflags', '--libs'])
	conf.find_program('protoc', var='PROTOC')
	conf.start_msg('Checking for protoc version')
	protocver = conf.cmd_and_log(conf.env.PROTOC + ['--version'], output=Context.BOTH)
	protocver = ''.join(protocver).strip()[protocver[0].rfind(' ')+1:]
	conf.end_msg(protocver)
	conf.env.PROTOC_MAJOR = protocver[:protocver.find('.')]
	conf.env.PROTOC_ST = '-I%s'
	conf.env.PROTOC_FL = '%s'
