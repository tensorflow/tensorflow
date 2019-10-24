#!/usr/bin/env python
# encoding: utf-8
# Thomas Nagy, 2005-2018 (ita)

"""

ConfigSet: a special dict

The values put in :py:class:`ConfigSet` must be serializable (dicts, lists, strings)
"""

import copy, re, os
from waflib import Logs, Utils
re_imp = re.compile(r'^(#)*?([^#=]*?)\ =\ (.*?)$', re.M)

class ConfigSet(object):
	"""
	A copy-on-write dict with human-readable serialized format. The serialization format
	is human-readable (python-like) and performed by using eval() and repr().
	For high performance prefer pickle. Do not store functions as they are not serializable.

	The values can be accessed by attributes or by keys::

		from waflib.ConfigSet import ConfigSet
		env = ConfigSet()
		env.FOO = 'test'
		env['FOO'] = 'test'
	"""
	__slots__ = ('table', 'parent')
	def __init__(self, filename=None):
		self.table = {}
		"""
		Internal dict holding the object values
		"""
		#self.parent = None

		if filename:
			self.load(filename)

	def __contains__(self, key):
		"""
		Enables the *in* syntax::

			if 'foo' in env:
				print(env['foo'])
		"""
		if key in self.table:
			return True
		try:
			return self.parent.__contains__(key)
		except AttributeError:
			return False # parent may not exist

	def keys(self):
		"""Dict interface"""
		keys = set()
		cur = self
		while cur:
			keys.update(cur.table.keys())
			cur = getattr(cur, 'parent', None)
		keys = list(keys)
		keys.sort()
		return keys

	def __iter__(self):
		return iter(self.keys())

	def __str__(self):
		"""Text representation of the ConfigSet (for debugging purposes)"""
		return "\n".join(["%r %r" % (x, self.__getitem__(x)) for x in self.keys()])

	def __getitem__(self, key):
		"""
		Dictionary interface: get value from key::

			def configure(conf):
				conf.env['foo'] = {}
				print(env['foo'])
		"""
		try:
			while 1:
				x = self.table.get(key)
				if not x is None:
					return x
				self = self.parent
		except AttributeError:
			return []

	def __setitem__(self, key, value):
		"""
		Dictionary interface: set value from key
		"""
		self.table[key] = value

	def __delitem__(self, key):
		"""
		Dictionary interface: mark the value as missing
		"""
		self[key] = []

	def __getattr__(self, name):
		"""
		Attribute access provided for convenience. The following forms are equivalent::

			def configure(conf):
				conf.env.value
				conf.env['value']
		"""
		if name in self.__slots__:
			return object.__getattribute__(self, name)
		else:
			return self[name]

	def __setattr__(self, name, value):
		"""
		Attribute access provided for convenience. The following forms are equivalent::

			def configure(conf):
				conf.env.value = x
				env['value'] = x
		"""
		if name in self.__slots__:
			object.__setattr__(self, name, value)
		else:
			self[name] = value

	def __delattr__(self, name):
		"""
		Attribute access provided for convenience. The following forms are equivalent::

			def configure(conf):
				del env.value
				del env['value']
		"""
		if name in self.__slots__:
			object.__delattr__(self, name)
		else:
			del self[name]

	def derive(self):
		"""
		Returns a new ConfigSet deriving from self. The copy returned
		will be a shallow copy::

			from waflib.ConfigSet import ConfigSet
			env = ConfigSet()
			env.append_value('CFLAGS', ['-O2'])
			child = env.derive()
			child.CFLAGS.append('test') # warning! this will modify 'env'
			child.CFLAGS = ['-O3'] # new list, ok
			child.append_value('CFLAGS', ['-O3']) # ok

		Use :py:func:`ConfigSet.detach` to detach the child from the parent.
		"""
		newenv = ConfigSet()
		newenv.parent = self
		return newenv

	def detach(self):
		"""
		Detaches this instance from its parent (if present)

		Modifying the parent :py:class:`ConfigSet` will not change the current object
		Modifying this :py:class:`ConfigSet` will not modify the parent one.
		"""
		tbl = self.get_merged_dict()
		try:
			delattr(self, 'parent')
		except AttributeError:
			pass
		else:
			keys = tbl.keys()
			for x in keys:
				tbl[x] = copy.deepcopy(tbl[x])
			self.table = tbl
		return self

	def get_flat(self, key):
		"""
		Returns a value as a string. If the input is a list, the value returned is space-separated.

		:param key: key to use
		:type key: string
		"""
		s = self[key]
		if isinstance(s, str):
			return s
		return ' '.join(s)

	def _get_list_value_for_modification(self, key):
		"""
		Returns a list value for further modification.

		The list may be modified inplace and there is no need to do this afterwards::

			self.table[var] = value
		"""
		try:
			value = self.table[key]
		except KeyError:
			try:
				value = self.parent[key]
			except AttributeError:
				value = []
			else:
				if isinstance(value, list):
					# force a copy
					value = value[:]
				else:
					value = [value]
			self.table[key] = value
		else:
			if not isinstance(value, list):
				self.table[key] = value = [value]
		return value

	def append_value(self, var, val):
		"""
		Appends a value to the specified config key::

			def build(bld):
				bld.env.append_value('CFLAGS', ['-O2'])

		The value must be a list or a tuple
		"""
		if isinstance(val, str): # if there were string everywhere we could optimize this
			val = [val]
		current_value = self._get_list_value_for_modification(var)
		current_value.extend(val)

	def prepend_value(self, var, val):
		"""
		Prepends a value to the specified item::

			def configure(conf):
				conf.env.prepend_value('CFLAGS', ['-O2'])

		The value must be a list or a tuple
		"""
		if isinstance(val, str):
			val = [val]
		self.table[var] =  val + self._get_list_value_for_modification(var)

	def append_unique(self, var, val):
		"""
		Appends a value to the specified item only if it's not already present::

			def build(bld):
				bld.env.append_unique('CFLAGS', ['-O2', '-g'])

		The value must be a list or a tuple
		"""
		if isinstance(val, str):
			val = [val]
		current_value = self._get_list_value_for_modification(var)

		for x in val:
			if x not in current_value:
				current_value.append(x)

	def get_merged_dict(self):
		"""
		Computes the merged dictionary from the fusion of self and all its parent

		:rtype: a ConfigSet object
		"""
		table_list = []
		env = self
		while 1:
			table_list.insert(0, env.table)
			try:
				env = env.parent
			except AttributeError:
				break
		merged_table = {}
		for table in table_list:
			merged_table.update(table)
		return merged_table

	def store(self, filename):
		"""
		Serializes the :py:class:`ConfigSet` data to a file. See :py:meth:`ConfigSet.load` for reading such files.

		:param filename: file to use
		:type filename: string
		"""
		try:
			os.makedirs(os.path.split(filename)[0])
		except OSError:
			pass

		buf = []
		merged_table = self.get_merged_dict()
		keys = list(merged_table.keys())
		keys.sort()

		try:
			fun = ascii
		except NameError:
			fun = repr

		for k in keys:
			if k != 'undo_stack':
				buf.append('%s = %s\n' % (k, fun(merged_table[k])))
		Utils.writef(filename, ''.join(buf))

	def load(self, filename):
		"""
		Restores contents from a file (current values are not cleared). Files are written using :py:meth:`ConfigSet.store`.

		:param filename: file to use
		:type filename: string
		"""
		tbl = self.table
		code = Utils.readf(filename, m='r')
		for m in re_imp.finditer(code):
			g = m.group
			tbl[g(2)] = eval(g(3))
		Logs.debug('env: %s', self.table)

	def update(self, d):
		"""
		Dictionary interface: replace values with the ones from another dict

		:param d: object to use the value from
		:type d: dict-like object
		"""
		self.table.update(d)

	def stash(self):
		"""
		Stores the object state to provide transactionality semantics::

			env = ConfigSet()
			env.stash()
			try:
				env.append_value('CFLAGS', '-O3')
				call_some_method(env)
			finally:
				env.revert()

		The history is kept in a stack, and is lost during the serialization by :py:meth:`ConfigSet.store`
		"""
		orig = self.table
		tbl = self.table = self.table.copy()
		for x in tbl.keys():
			tbl[x] = copy.deepcopy(tbl[x])
		self.undo_stack = self.undo_stack + [orig]

	def commit(self):
		"""
		Commits transactional changes. See :py:meth:`ConfigSet.stash`
		"""
		self.undo_stack.pop(-1)

	def revert(self):
		"""
		Reverts the object to a previous state. See :py:meth:`ConfigSet.stash`
		"""
		self.table = self.undo_stack.pop(-1)

