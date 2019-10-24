#!/usr/bin/env python
# encoding: utf-8
# Thomas Nagy, 2005-2018 (ita)

"""
Node: filesystem structure

#. Each file/folder is represented by exactly one node.

#. Some potential class properties are stored on :py:class:`waflib.Build.BuildContext` : nodes to depend on, etc.
   Unused class members can increase the `.wafpickle` file size sensibly.

#. Node objects should never be created directly, use
   the methods :py:func:`Node.make_node` or :py:func:`Node.find_node` for the low-level operations

#. The methods :py:func:`Node.find_resource`, :py:func:`Node.find_dir` :py:func:`Node.find_or_declare` must be
   used when a build context is present

#. Each instance of :py:class:`waflib.Context.Context` has a unique :py:class:`Node` subclass required for serialization.
   (:py:class:`waflib.Node.Nod3`, see the :py:class:`waflib.Context.Context` initializer). A reference to the context
   owning a node is held as *self.ctx*
"""

import os, re, sys, shutil
from waflib import Utils, Errors

exclude_regs = '''
**/*~
**/#*#
**/.#*
**/%*%
**/._*
**/*.swp
**/CVS
**/CVS/**
**/.cvsignore
**/SCCS
**/SCCS/**
**/vssver.scc
**/.svn
**/.svn/**
**/BitKeeper
**/.git
**/.git/**
**/.gitignore
**/.bzr
**/.bzrignore
**/.bzr/**
**/.hg
**/.hg/**
**/_MTN
**/_MTN/**
**/.arch-ids
**/{arch}
**/_darcs
**/_darcs/**
**/.intlcache
**/.DS_Store'''
"""
Ant patterns for files and folders to exclude while doing the
recursive traversal in :py:meth:`waflib.Node.Node.ant_glob`
"""

def ant_matcher(s, ignorecase):
	reflags = re.I if ignorecase else 0
	ret = []
	for x in Utils.to_list(s):
		x = x.replace('\\', '/').replace('//', '/')
		if x.endswith('/'):
			x += '**'
		accu = []
		for k in x.split('/'):
			if k == '**':
				accu.append(k)
			else:
				k = k.replace('.', '[.]').replace('*', '.*').replace('?', '.').replace('+', '\\+')
				k = '^%s$' % k
				try:
					exp = re.compile(k, flags=reflags)
				except Exception as e:
					raise Errors.WafError('Invalid pattern: %s' % k, e)
				else:
					accu.append(exp)
		ret.append(accu)
	return ret

def ant_sub_filter(name, nn):
	ret = []
	for lst in nn:
		if not lst:
			pass
		elif lst[0] == '**':
			ret.append(lst)
			if len(lst) > 1:
				if lst[1].match(name):
					ret.append(lst[2:])
			else:
				ret.append([])
		elif lst[0].match(name):
			ret.append(lst[1:])
	return ret

def ant_sub_matcher(name, pats):
	nacc = ant_sub_filter(name, pats[0])
	nrej = ant_sub_filter(name, pats[1])
	if [] in nrej:
		nacc = []
	return [nacc, nrej]

class Node(object):
	"""
	This class is organized in two parts:

	* The basic methods meant for filesystem access (compute paths, create folders, etc)
	* The methods bound to a :py:class:`waflib.Build.BuildContext` (require ``bld.srcnode`` and ``bld.bldnode``)
	"""

	dict_class = dict
	"""
	Subclasses can provide a dict class to enable case insensitivity for example.
	"""

	__slots__ = ('name', 'parent', 'children', 'cache_abspath', 'cache_isdir')
	def __init__(self, name, parent):
		"""
		.. note:: Use :py:func:`Node.make_node` or :py:func:`Node.find_node` instead of calling this constructor
		"""
		self.name = name
		self.parent = parent
		if parent:
			if name in parent.children:
				raise Errors.WafError('node %s exists in the parent files %r already' % (name, parent))
			parent.children[name] = self

	def __setstate__(self, data):
		"Deserializes node information, used for persistence"
		self.name = data[0]
		self.parent = data[1]
		if data[2] is not None:
			# Issue 1480
			self.children = self.dict_class(data[2])

	def __getstate__(self):
		"Serializes node information, used for persistence"
		return (self.name, self.parent, getattr(self, 'children', None))

	def __str__(self):
		"""
		String representation (abspath), for debugging purposes

		:rtype: string
		"""
		return self.abspath()

	def __repr__(self):
		"""
		String representation (abspath), for debugging purposes

		:rtype: string
		"""
		return self.abspath()

	def __copy__(self):
		"""
		Provided to prevent nodes from being copied

		:raises: :py:class:`waflib.Errors.WafError`
		"""
		raise Errors.WafError('nodes are not supposed to be copied')

	def read(self, flags='r', encoding='latin-1'):
		"""
		Reads and returns the contents of the file represented by this node, see :py:func:`waflib.Utils.readf`::

			def build(bld):
				bld.path.find_node('wscript').read()

		:param flags: Open mode
		:type  flags: string
		:param encoding: encoding value for Python3
		:type encoding: string
		:rtype: string or bytes
		:return: File contents
		"""
		return Utils.readf(self.abspath(), flags, encoding)

	def write(self, data, flags='w', encoding='latin-1'):
		"""
		Writes data to the file represented by this node, see :py:func:`waflib.Utils.writef`::

			def build(bld):
				bld.path.make_node('foo.txt').write('Hello, world!')

		:param data: data to write
		:type  data: string
		:param flags: Write mode
		:type  flags: string
		:param encoding: encoding value for Python3
		:type encoding: string
		"""
		Utils.writef(self.abspath(), data, flags, encoding)

	def read_json(self, convert=True, encoding='utf-8'):
		"""
		Reads and parses the contents of this node as JSON (Python ≥ 2.6)::

			def build(bld):
				bld.path.find_node('abc.json').read_json()

		Note that this by default automatically decodes unicode strings on Python2, unlike what the Python JSON module does.

		:type  convert: boolean
		:param convert: Prevents decoding of unicode strings on Python2
		:type  encoding: string
		:param encoding: The encoding of the file to read. This default to UTF8 as per the JSON standard
		:rtype: object
		:return: Parsed file contents
		"""
		import json # Python 2.6 and up
		object_pairs_hook = None
		if convert and sys.hexversion < 0x3000000:
			try:
				_type = unicode
			except NameError:
				_type = str

			def convert(value):
				if isinstance(value, list):
					return [convert(element) for element in value]
				elif isinstance(value, _type):
					return str(value)
				else:
					return value

			def object_pairs(pairs):
				return dict((str(pair[0]), convert(pair[1])) for pair in pairs)

			object_pairs_hook = object_pairs

		return json.loads(self.read(encoding=encoding), object_pairs_hook=object_pairs_hook)

	def write_json(self, data, pretty=True):
		"""
		Writes a python object as JSON to disk (Python ≥ 2.6) as UTF-8 data (JSON standard)::

			def build(bld):
				bld.path.find_node('xyz.json').write_json(199)

		:type  data: object
		:param data: The data to write to disk
		:type  pretty: boolean
		:param pretty: Determines if the JSON will be nicely space separated
		"""
		import json # Python 2.6 and up
		indent = 2
		separators = (',', ': ')
		sort_keys = pretty
		newline = os.linesep
		if not pretty:
			indent = None
			separators = (',', ':')
			newline = ''
		output = json.dumps(data, indent=indent, separators=separators, sort_keys=sort_keys) + newline
		self.write(output, encoding='utf-8')

	def exists(self):
		"""
		Returns whether the Node is present on the filesystem

		:rtype: bool
		"""
		return os.path.exists(self.abspath())

	def isdir(self):
		"""
		Returns whether the Node represents a folder

		:rtype: bool
		"""
		return os.path.isdir(self.abspath())

	def chmod(self, val):
		"""
		Changes the file/dir permissions::

			def build(bld):
				bld.path.chmod(493) # 0755
		"""
		os.chmod(self.abspath(), val)

	def delete(self, evict=True):
		"""
		Removes the file/folder from the filesystem (equivalent to `rm -rf`), and remove this object from the Node tree.
		Do not use this object after calling this method.
		"""
		try:
			try:
				if os.path.isdir(self.abspath()):
					shutil.rmtree(self.abspath())
				else:
					os.remove(self.abspath())
			except OSError:
				if os.path.exists(self.abspath()):
					raise
		finally:
			if evict:
				self.evict()

	def evict(self):
		"""
		Removes this node from the Node tree
		"""
		del self.parent.children[self.name]

	def suffix(self):
		"""
		Returns the file rightmost extension, for example `a.b.c.d → .d`

		:rtype: string
		"""
		k = max(0, self.name.rfind('.'))
		return self.name[k:]

	def height(self):
		"""
		Returns the depth in the folder hierarchy from the filesystem root or from all the file drives

		:returns: filesystem depth
		:rtype: integer
		"""
		d = self
		val = -1
		while d:
			d = d.parent
			val += 1
		return val

	def listdir(self):
		"""
		Lists the folder contents

		:returns: list of file/folder names ordered alphabetically
		:rtype: list of string
		"""
		lst = Utils.listdir(self.abspath())
		lst.sort()
		return lst

	def mkdir(self):
		"""
		Creates a folder represented by this node. Intermediate folders are created as needed.

		:raises: :py:class:`waflib.Errors.WafError` when the folder is missing
		"""
		if self.isdir():
			return

		try:
			self.parent.mkdir()
		except OSError:
			pass

		if self.name:
			try:
				os.makedirs(self.abspath())
			except OSError:
				pass

			if not self.isdir():
				raise Errors.WafError('Could not create the directory %r' % self)

			try:
				self.children
			except AttributeError:
				self.children = self.dict_class()

	def find_node(self, lst):
		"""
		Finds a node on the file system (files or folders), and creates the corresponding Node objects if it exists

		:param lst: relative path
		:type lst: string or list of string
		:returns: The corresponding Node object or None if no entry was found on the filesystem
		:rtype: :py:class:´waflib.Node.Node´
		"""

		if isinstance(lst, str):
			lst = [x for x in Utils.split_path(lst) if x and x != '.']

		if lst and lst[0].startswith('\\\\') and not self.parent:
			node = self.ctx.root.make_node(lst[0])
			node.cache_isdir = True
			return node.find_node(lst[1:])

		cur = self
		for x in lst:
			if x == '..':
				cur = cur.parent or cur
				continue

			try:
				ch = cur.children
			except AttributeError:
				cur.children = self.dict_class()
			else:
				try:
					cur = ch[x]
					continue
				except KeyError:
					pass

			# optimistic: create the node first then look if it was correct to do so
			cur = self.__class__(x, cur)
			if not cur.exists():
				cur.evict()
				return None

		if not cur.exists():
			cur.evict()
			return None

		return cur

	def make_node(self, lst):
		"""
		Returns or creates a Node object corresponding to the input path without considering the filesystem.

		:param lst: relative path
		:type lst: string or list of string
		:rtype: :py:class:´waflib.Node.Node´
		"""
		if isinstance(lst, str):
			lst = [x for x in Utils.split_path(lst) if x and x != '.']

		cur = self
		for x in lst:
			if x == '..':
				cur = cur.parent or cur
				continue

			try:
				cur = cur.children[x]
			except AttributeError:
				cur.children = self.dict_class()
			except KeyError:
				pass
			else:
				continue
			cur = self.__class__(x, cur)
		return cur

	def search_node(self, lst):
		"""
		Returns a Node previously defined in the data structure. The filesystem is not considered.

		:param lst: relative path
		:type lst: string or list of string
		:rtype: :py:class:´waflib.Node.Node´ or None if there is no entry in the Node datastructure
		"""
		if isinstance(lst, str):
			lst = [x for x in Utils.split_path(lst) if x and x != '.']

		cur = self
		for x in lst:
			if x == '..':
				cur = cur.parent or cur
			else:
				try:
					cur = cur.children[x]
				except (AttributeError, KeyError):
					return None
		return cur

	def path_from(self, node):
		"""
		Path of this node seen from the other::

			def build(bld):
				n1 = bld.path.find_node('foo/bar/xyz.txt')
				n2 = bld.path.find_node('foo/stuff/')
				n1.path_from(n2) # '../bar/xyz.txt'

		:param node: path to use as a reference
		:type node: :py:class:`waflib.Node.Node`
		:returns: a relative path or an absolute one if that is better
		:rtype: string
		"""
		c1 = self
		c2 = node

		c1h = c1.height()
		c2h = c2.height()

		lst = []
		up = 0

		while c1h > c2h:
			lst.append(c1.name)
			c1 = c1.parent
			c1h -= 1

		while c2h > c1h:
			up += 1
			c2 = c2.parent
			c2h -= 1

		while not c1 is c2:
			lst.append(c1.name)
			up += 1

			c1 = c1.parent
			c2 = c2.parent

		if c1.parent:
			lst.extend(['..'] * up)
			lst.reverse()
			return os.sep.join(lst) or '.'
		else:
			return self.abspath()

	def abspath(self):
		"""
		Returns the absolute path. A cache is kept in the context as ``cache_node_abspath``

		:rtype: string
		"""
		try:
			return self.cache_abspath
		except AttributeError:
			pass
		# think twice before touching this (performance + complexity + correctness)

		if not self.parent:
			val = os.sep
		elif not self.parent.name:
			val = os.sep + self.name
		else:
			val = self.parent.abspath() + os.sep + self.name
		self.cache_abspath = val
		return val

	if Utils.is_win32:
		def abspath(self):
			try:
				return self.cache_abspath
			except AttributeError:
				pass
			if not self.parent:
				val = ''
			elif not self.parent.name:
				val = self.name + os.sep
			else:
				val = self.parent.abspath().rstrip(os.sep) + os.sep + self.name
			self.cache_abspath = val
			return val

	def is_child_of(self, node):
		"""
		Returns whether the object belongs to a subtree of the input node::

			def build(bld):
				node = bld.path.find_node('wscript')
				node.is_child_of(bld.path) # True

		:param node: path to use as a reference
		:type node: :py:class:`waflib.Node.Node`
		:rtype: bool
		"""
		p = self
		diff = self.height() - node.height()
		while diff > 0:
			diff -= 1
			p = p.parent
		return p is node

	def ant_iter(self, accept=None, maxdepth=25, pats=[], dir=False, src=True, remove=True, quiet=False):
		"""
		Recursive method used by :py:meth:`waflib.Node.ant_glob`.

		:param accept: function used for accepting/rejecting a node, returns the patterns that can be still accepted in recursion
		:type accept: function
		:param maxdepth: maximum depth in the filesystem (25)
		:type maxdepth: int
		:param pats: list of patterns to accept and list of patterns to exclude
		:type pats: tuple
		:param dir: return folders too (False by default)
		:type dir: bool
		:param src: return files (True by default)
		:type src: bool
		:param remove: remove files/folders that do not exist (True by default)
		:type remove: bool
		:param quiet: disable build directory traversal warnings (verbose mode)
		:type quiet: bool
		:returns: A generator object to iterate from
		:rtype: iterator
		"""
		dircont = self.listdir()

		try:
			lst = set(self.children.keys())
		except AttributeError:
			self.children = self.dict_class()
		else:
			if remove:
				for x in lst - set(dircont):
					self.children[x].evict()

		for name in dircont:
			npats = accept(name, pats)
			if npats and npats[0]:
				accepted = [] in npats[0]

				node = self.make_node([name])

				isdir = node.isdir()
				if accepted:
					if isdir:
						if dir:
							yield node
					elif src:
						yield node

				if isdir:
					node.cache_isdir = True
					if maxdepth:
						for k in node.ant_iter(accept=accept, maxdepth=maxdepth - 1, pats=npats, dir=dir, src=src, remove=remove, quiet=quiet):
							yield k

	def ant_glob(self, *k, **kw):
		"""
		Finds files across folders and returns Node objects:

		* ``**/*`` find all files recursively
		* ``**/*.class`` find all files ending by .class
		* ``..`` find files having two dot characters

		For example::

			def configure(cfg):
				# find all .cpp files
				cfg.path.ant_glob('**/*.cpp')
				# find particular files from the root filesystem (can be slow)
				cfg.root.ant_glob('etc/*.txt')
				# simple exclusion rule example
				cfg.path.ant_glob('*.c*', excl=['*.c'], src=True, dir=False)

		For more information about the patterns, consult http://ant.apache.org/manual/dirtasks.html
		Please remember that the '..' sequence does not represent the parent directory::

			def configure(cfg):
				cfg.path.ant_glob('../*.h') # incorrect
				cfg.path.parent.ant_glob('*.h') # correct

		The Node structure is itself a filesystem cache, so certain precautions must
		be taken while matching files in the build or installation phases.
		Nodes objects that do have a corresponding file or folder are garbage-collected by default.
		This garbage collection is usually required to prevent returning files that do not
		exist anymore. Yet, this may also remove Node objects of files that are yet-to-be built.

		This typically happens when trying to match files in the build directory,
		but there are also cases when files are created in the source directory.
		Run ``waf -v`` to display any warnings, and try consider passing ``remove=False``
		when matching files in the build directory.

		Since ant_glob can traverse both source and build folders, it is a best practice
		to call this method only from the most specific build node::

			def build(bld):
				# traverses the build directory, may need ``remove=False``:
				bld.path.ant_glob('project/dir/**/*.h')
				# better, no accidental build directory traversal:
				bld.path.find_node('project/dir').ant_glob('**/*.h') # best

		In addition, files and folders are listed immediately. When matching files in the
		build folders, consider passing ``generator=True`` so that the generator object
		returned can defer computation to a later stage. For example::

			def build(bld):
				bld(rule='tar xvf ${SRC}', source='arch.tar')
				bld.add_group()
				gen = bld.bldnode.ant_glob("*.h", generator=True, remove=True)
				# files will be listed only after the arch.tar is unpacked
				bld(rule='ls ${SRC}', source=gen, name='XYZ')


		:param incl: ant patterns or list of patterns to include
		:type incl: string or list of strings
		:param excl: ant patterns or list of patterns to exclude
		:type excl: string or list of strings
		:param dir: return folders too (False by default)
		:type dir: bool
		:param src: return files (True by default)
		:type src: bool
		:param maxdepth: maximum depth of recursion
		:type maxdepth: int
		:param ignorecase: ignore case while matching (False by default)
		:type ignorecase: bool
		:param generator: Whether to evaluate the Nodes lazily
		:type generator: bool
		:param remove: remove files/folders that do not exist (True by default)
		:type remove: bool
		:param quiet: disable build directory traversal warnings (verbose mode)
		:type quiet: bool
		:returns: The corresponding Node objects as a list or as a generator object (generator=True)
		:rtype: by default, list of :py:class:`waflib.Node.Node` instances
		"""
		src = kw.get('src', True)
		dir = kw.get('dir')
		excl = kw.get('excl', exclude_regs)
		incl = k and k[0] or kw.get('incl', '**')
		remove = kw.get('remove', True)
		maxdepth = kw.get('maxdepth', 25)
		ignorecase = kw.get('ignorecase', False)
		quiet = kw.get('quiet', False)
		pats = (ant_matcher(incl, ignorecase), ant_matcher(excl, ignorecase))

		if kw.get('generator'):
			return Utils.lazy_generator(self.ant_iter, (ant_sub_matcher, maxdepth, pats, dir, src, remove, quiet))

		it = self.ant_iter(ant_sub_matcher, maxdepth, pats, dir, src, remove, quiet)
		if kw.get('flat'):
			# returns relative paths as a space-delimited string
			# prefer Node objects whenever possible
			return ' '.join(x.path_from(self) for x in it)
		return list(it)

	# ----------------------------------------------------------------------------
	# the methods below require the source/build folders (bld.srcnode/bld.bldnode)

	def is_src(self):
		"""
		Returns True if the node is below the source directory. Note that ``!is_src() ≠ is_bld()``

		:rtype: bool
		"""
		cur = self
		x = self.ctx.srcnode
		y = self.ctx.bldnode
		while cur.parent:
			if cur is y:
				return False
			if cur is x:
				return True
			cur = cur.parent
		return False

	def is_bld(self):
		"""
		Returns True if the node is below the build directory. Note that ``!is_bld() ≠ is_src()``

		:rtype: bool
		"""
		cur = self
		y = self.ctx.bldnode
		while cur.parent:
			if cur is y:
				return True
			cur = cur.parent
		return False

	def get_src(self):
		"""
		Returns the corresponding Node object in the source directory (or self if already
		under the source directory). Use this method only if the purpose is to create
		a Node object (this is common with folders but not with files, see ticket 1937)

		:rtype: :py:class:`waflib.Node.Node`
		"""
		cur = self
		x = self.ctx.srcnode
		y = self.ctx.bldnode
		lst = []
		while cur.parent:
			if cur is y:
				lst.reverse()
				return x.make_node(lst)
			if cur is x:
				return self
			lst.append(cur.name)
			cur = cur.parent
		return self

	def get_bld(self):
		"""
		Return the corresponding Node object in the build directory (or self if already
		under the build directory). Use this method only if the purpose is to create
		a Node object (this is common with folders but not with files, see ticket 1937)

		:rtype: :py:class:`waflib.Node.Node`
		"""
		cur = self
		x = self.ctx.srcnode
		y = self.ctx.bldnode
		lst = []
		while cur.parent:
			if cur is y:
				return self
			if cur is x:
				lst.reverse()
				return self.ctx.bldnode.make_node(lst)
			lst.append(cur.name)
			cur = cur.parent
		# the file is external to the current project, make a fake root in the current build directory
		lst.reverse()
		if lst and Utils.is_win32 and len(lst[0]) == 2 and lst[0].endswith(':'):
			lst[0] = lst[0][0]
		return self.ctx.bldnode.make_node(['__root__'] + lst)

	def find_resource(self, lst):
		"""
		Use this method in the build phase to find source files corresponding to the relative path given.

		First it looks up the Node data structure to find any declared Node object in the build directory.
		If None is found, it then considers the filesystem in the source directory.

		:param lst: relative path
		:type lst: string or list of string
		:returns: the corresponding Node object or None
		:rtype: :py:class:`waflib.Node.Node`
		"""
		if isinstance(lst, str):
			lst = [x for x in Utils.split_path(lst) if x and x != '.']

		node = self.get_bld().search_node(lst)
		if not node:
			node = self.get_src().find_node(lst)
		if node and node.isdir():
			return None
		return node

	def find_or_declare(self, lst):
		"""
		Use this method in the build phase to declare output files which
		are meant to be written in the build directory.

		This method creates the Node object and its parent folder
		as needed.

		:param lst: relative path
		:type lst: string or list of string
		"""
		if isinstance(lst, str) and os.path.isabs(lst):
			node = self.ctx.root.make_node(lst)
		else:
			node = self.get_bld().make_node(lst)
		node.parent.mkdir()
		return node

	def find_dir(self, lst):
		"""
		Searches for a folder on the filesystem (see :py:meth:`waflib.Node.Node.find_node`)

		:param lst: relative path
		:type lst: string or list of string
		:returns: The corresponding Node object or None if there is no such folder
		:rtype: :py:class:`waflib.Node.Node`
		"""
		if isinstance(lst, str):
			lst = [x for x in Utils.split_path(lst) if x and x != '.']

		node = self.find_node(lst)
		if node and not node.isdir():
			return None
		return node

	# helpers for building things
	def change_ext(self, ext, ext_in=None):
		"""
		Declares a build node with a distinct extension; this is uses :py:meth:`waflib.Node.Node.find_or_declare`

		:return: A build node of the same path, but with a different extension
		:rtype: :py:class:`waflib.Node.Node`
		"""
		name = self.name
		if ext_in is None:
			k = name.rfind('.')
			if k >= 0:
				name = name[:k] + ext
			else:
				name = name + ext
		else:
			name = name[:- len(ext_in)] + ext

		return self.parent.find_or_declare([name])

	def bldpath(self):
		"""
		Returns the relative path seen from the build directory ``src/foo.cpp``

		:rtype: string
		"""
		return self.path_from(self.ctx.bldnode)

	def srcpath(self):
		"""
		Returns the relative path seen from the source directory ``../src/foo.cpp``

		:rtype: string
		"""
		return self.path_from(self.ctx.srcnode)

	def relpath(self):
		"""
		If a file in the build directory, returns :py:meth:`waflib.Node.Node.bldpath`,
		else returns :py:meth:`waflib.Node.Node.srcpath`

		:rtype: string
		"""
		cur = self
		x = self.ctx.bldnode
		while cur.parent:
			if cur is x:
				return self.bldpath()
			cur = cur.parent
		return self.srcpath()

	def bld_dir(self):
		"""
		Equivalent to self.parent.bldpath()

		:rtype: string
		"""
		return self.parent.bldpath()

	def h_file(self):
		"""
		See :py:func:`waflib.Utils.h_file`

		:return: a hash representing the file contents
		:rtype: string or bytes
		"""
		return Utils.h_file(self.abspath())

	def get_bld_sig(self):
		"""
		Returns a signature (see :py:meth:`waflib.Node.Node.h_file`) for the purpose
		of build dependency calculation. This method uses a per-context cache.

		:return: a hash representing the object contents
		:rtype: string or bytes
		"""
		# previous behaviour can be set by returning self.ctx.node_sigs[self] when a build node
		try:
			cache = self.ctx.cache_sig
		except AttributeError:
			cache = self.ctx.cache_sig = {}
		try:
			ret = cache[self]
		except KeyError:
			p = self.abspath()
			try:
				ret = cache[self] = self.h_file()
			except EnvironmentError:
				if self.isdir():
					# allow folders as build nodes, do not use the creation time
					st = os.stat(p)
					ret = cache[self] = Utils.h_list([p, st.st_ino, st.st_mode])
					return ret
				raise
		return ret

pickle_lock = Utils.threading.Lock()
"""Lock mandatory for thread-safe node serialization"""

class Nod3(Node):
	"""Mandatory subclass for thread-safe node serialization"""
	pass # do not remove


