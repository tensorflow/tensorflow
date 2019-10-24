#!/usr/bin/env python
# encoding: utf-8
# Thomas Nagy, 2005-2018 (ita)

"""
Utilities and platform-specific fixes

The portability fixes try to provide a consistent behavior of the Waf API
through Python versions 2.5 to 3.X and across different platforms (win32, linux, etc)
"""

from __future__ import with_statement

import atexit, os, sys, errno, inspect, re, datetime, platform, base64, signal, functools, time

try:
	import cPickle
except ImportError:
	import pickle as cPickle

# leave this
if os.name == 'posix' and sys.version_info[0] < 3:
	try:
		import subprocess32 as subprocess
	except ImportError:
		import subprocess
else:
	import subprocess

try:
	TimeoutExpired = subprocess.TimeoutExpired
except AttributeError:
	class TimeoutExpired(Exception):
		pass

from collections import deque, defaultdict

try:
	import _winreg as winreg
except ImportError:
	try:
		import winreg
	except ImportError:
		winreg = None

from waflib import Errors

try:
	from hashlib import md5
except ImportError:
	try:
		from hashlib import sha1 as md5
	except ImportError:
		# never fail to enable potential fixes from another module
		pass
else:
	try:
		md5().digest()
	except ValueError:
		# Fips? #2213
		from hashlib import sha1 as md5

try:
	import threading
except ImportError:
	if not 'JOBS' in os.environ:
		# no threading :-(
		os.environ['JOBS'] = '1'

	class threading(object):
		"""
		A fake threading class for platforms lacking the threading module.
		Use ``waf -j1`` on those platforms
		"""
		pass
	class Lock(object):
		"""Fake Lock class"""
		def acquire(self):
			pass
		def release(self):
			pass
	threading.Lock = threading.Thread = Lock

SIG_NIL = 'SIG_NIL_SIG_NIL_'.encode()
"""Arbitrary null value for hashes. Modify this value according to the hash function in use"""

O644 = 420
"""Constant representing the permissions for regular files (0644 raises a syntax error on python 3)"""

O755 = 493
"""Constant representing the permissions for executable files (0755 raises a syntax error on python 3)"""

rot_chr = ['\\', '|', '/', '-']
"List of characters to use when displaying the throbber (progress bar)"

rot_idx = 0
"Index of the current throbber character (progress bar)"

class ordered_iter_dict(dict):
	"""Ordered dictionary that provides iteration from the most recently inserted keys first"""
	def __init__(self, *k, **kw):
		self.lst = deque()
		dict.__init__(self, *k, **kw)
	def clear(self):
		dict.clear(self)
		self.lst = deque()
	def __setitem__(self, key, value):
		if key in dict.keys(self):
			self.lst.remove(key)
		dict.__setitem__(self, key, value)
		self.lst.append(key)
	def __delitem__(self, key):
		dict.__delitem__(self, key)
		try:
			self.lst.remove(key)
		except ValueError:
			pass
	def __iter__(self):
		return reversed(self.lst)
	def keys(self):
		return reversed(self.lst)

class lru_node(object):
	"""
	Used by :py:class:`waflib.Utils.lru_cache`
	"""
	__slots__ = ('next', 'prev', 'key', 'val')
	def __init__(self):
		self.next = self
		self.prev = self
		self.key = None
		self.val = None

class lru_cache(object):
	"""
	A simple least-recently used cache with lazy allocation
	"""
	__slots__ = ('maxlen', 'table', 'head')
	def __init__(self, maxlen=100):
		self.maxlen = maxlen
		"""
		Maximum amount of elements in the cache
		"""
		self.table = {}
		"""
		Mapping key-value
		"""
		self.head = lru_node()
		self.head.next = self.head
		self.head.prev = self.head

	def __getitem__(self, key):
		node = self.table[key]
		# assert(key==node.key)
		if node is self.head:
			return node.val

		# detach the node found
		node.prev.next = node.next
		node.next.prev = node.prev

		# replace the head
		node.next = self.head.next
		node.prev = self.head
		self.head = node.next.prev = node.prev.next = node

		return node.val

	def __setitem__(self, key, val):
		if key in self.table:
			# update the value for an existing key
			node = self.table[key]
			node.val = val
			self.__getitem__(key)
		else:
			if len(self.table) < self.maxlen:
				# the very first item is unused until the maximum is reached
				node = lru_node()
				node.prev = self.head
				node.next = self.head.next
				node.prev.next = node.next.prev = node
			else:
				node = self.head = self.head.next
				try:
					# that's another key
					del self.table[node.key]
				except KeyError:
					pass

			node.key = key
			node.val = val
			self.table[key] = node

class lazy_generator(object):
	def __init__(self, fun, params):
		self.fun = fun
		self.params = params

	def __iter__(self):
		return self

	def __next__(self):
		try:
			it = self.it
		except AttributeError:
			it = self.it = self.fun(*self.params)
		return next(it)

	next = __next__

is_win32 = os.sep == '\\' or sys.platform == 'win32' or os.name == 'nt' # msys2
"""
Whether this system is a Windows series
"""

def readf(fname, m='r', encoding='latin-1'):
	"""
	Reads an entire file into a string. See also :py:meth:`waflib.Node.Node.readf`::

		def build(ctx):
			from waflib import Utils
			txt = Utils.readf(self.path.find_node('wscript').abspath())
			txt = ctx.path.find_node('wscript').read()

	:type  fname: string
	:param fname: Path to file
	:type  m: string
	:param m: Open mode
	:type encoding: string
	:param encoding: encoding value, only used for python 3
	:rtype: string
	:return: Content of the file
	"""

	if sys.hexversion > 0x3000000 and not 'b' in m:
		m += 'b'
		with open(fname, m) as f:
			txt = f.read()
		if encoding:
			txt = txt.decode(encoding)
		else:
			txt = txt.decode()
	else:
		with open(fname, m) as f:
			txt = f.read()
	return txt

def writef(fname, data, m='w', encoding='latin-1'):
	"""
	Writes an entire file from a string.
	See also :py:meth:`waflib.Node.Node.writef`::

		def build(ctx):
			from waflib import Utils
			txt = Utils.writef(self.path.make_node('i_like_kittens').abspath(), 'some data')
			self.path.make_node('i_like_kittens').write('some data')

	:type  fname: string
	:param fname: Path to file
	:type   data: string
	:param  data: The contents to write to the file
	:type  m: string
	:param m: Open mode
	:type encoding: string
	:param encoding: encoding value, only used for python 3
	"""
	if sys.hexversion > 0x3000000 and not 'b' in m:
		data = data.encode(encoding)
		m += 'b'
	with open(fname, m) as f:
		f.write(data)

def h_file(fname):
	"""
	Computes a hash value for a file by using md5. Use the md5_tstamp
	extension to get faster build hashes if necessary.

	:type fname: string
	:param fname: path to the file to hash
	:return: hash of the file contents
	:rtype: string or bytes
	"""
	m = md5()
	with open(fname, 'rb') as f:
		while fname:
			fname = f.read(200000)
			m.update(fname)
	return m.digest()

def readf_win32(f, m='r', encoding='latin-1'):
	flags = os.O_NOINHERIT | os.O_RDONLY
	if 'b' in m:
		flags |= os.O_BINARY
	if '+' in m:
		flags |= os.O_RDWR
	try:
		fd = os.open(f, flags)
	except OSError:
		raise IOError('Cannot read from %r' % f)

	if sys.hexversion > 0x3000000 and not 'b' in m:
		m += 'b'
		with os.fdopen(fd, m) as f:
			txt = f.read()
		if encoding:
			txt = txt.decode(encoding)
		else:
			txt = txt.decode()
	else:
		with os.fdopen(fd, m) as f:
			txt = f.read()
	return txt

def writef_win32(f, data, m='w', encoding='latin-1'):
	if sys.hexversion > 0x3000000 and not 'b' in m:
		data = data.encode(encoding)
		m += 'b'
	flags = os.O_CREAT | os.O_TRUNC | os.O_WRONLY | os.O_NOINHERIT
	if 'b' in m:
		flags |= os.O_BINARY
	if '+' in m:
		flags |= os.O_RDWR
	try:
		fd = os.open(f, flags)
	except OSError:
		raise OSError('Cannot write to %r' % f)
	with os.fdopen(fd, m) as f:
		f.write(data)

def h_file_win32(fname):
	try:
		fd = os.open(fname, os.O_BINARY | os.O_RDONLY | os.O_NOINHERIT)
	except OSError:
		raise OSError('Cannot read from %r' % fname)
	m = md5()
	with os.fdopen(fd, 'rb') as f:
		while fname:
			fname = f.read(200000)
			m.update(fname)
	return m.digest()

# always save these
readf_unix = readf
writef_unix = writef
h_file_unix = h_file
if hasattr(os, 'O_NOINHERIT') and sys.hexversion < 0x3040000:
	# replace the default functions
	readf = readf_win32
	writef = writef_win32
	h_file = h_file_win32

try:
	x = ''.encode('hex')
except LookupError:
	import binascii
	def to_hex(s):
		ret = binascii.hexlify(s)
		if not isinstance(ret, str):
			ret = ret.decode('utf-8')
		return ret
else:
	def to_hex(s):
		return s.encode('hex')

to_hex.__doc__ = """
Return the hexadecimal representation of a string

:param s: string to convert
:type s: string
"""

def listdir_win32(s):
	"""
	Lists the contents of a folder in a portable manner.
	On Win32, returns the list of drive letters: ['C:', 'X:', 'Z:'] when an empty string is given.

	:type s: string
	:param s: a string, which can be empty on Windows
	"""
	if not s:
		try:
			import ctypes
		except ImportError:
			# there is nothing much we can do
			return [x + ':\\' for x in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ']
		else:
			dlen = 4 # length of "?:\\x00"
			maxdrives = 26
			buf = ctypes.create_string_buffer(maxdrives * dlen)
			ndrives = ctypes.windll.kernel32.GetLogicalDriveStringsA(maxdrives*dlen, ctypes.byref(buf))
			return [ str(buf.raw[4*i:4*i+2].decode('ascii')) for i in range(int(ndrives/dlen)) ]

	if len(s) == 2 and s[1] == ":":
		s += os.sep

	if not os.path.isdir(s):
		e = OSError('%s is not a directory' % s)
		e.errno = errno.ENOENT
		raise e
	return os.listdir(s)

listdir = os.listdir
if is_win32:
	listdir = listdir_win32

def num2ver(ver):
	"""
	Converts a string, tuple or version number into an integer. The number is supposed to have at most 4 digits::

		from waflib.Utils import num2ver
		num2ver('1.3.2') == num2ver((1,3,2)) == num2ver((1,3,2,0))

	:type ver: string or tuple of numbers
	:param ver: a version number
	"""
	if isinstance(ver, str):
		ver = tuple(ver.split('.'))
	if isinstance(ver, tuple):
		ret = 0
		for i in range(4):
			if i < len(ver):
				ret += 256**(3 - i) * int(ver[i])
		return ret
	return ver

def to_list(val):
	"""
	Converts a string argument to a list by splitting it by spaces.
	Returns the object if not a string::

		from waflib.Utils import to_list
		lst = to_list('a b c d')

	:param val: list of string or space-separated string
	:rtype: list
	:return: Argument converted to list
	"""
	if isinstance(val, str):
		return val.split()
	else:
		return val

def console_encoding():
	try:
		import ctypes
	except ImportError:
		pass
	else:
		try:
			codepage = ctypes.windll.kernel32.GetConsoleCP()
		except AttributeError:
			pass
		else:
			if codepage:
				return 'cp%d' % codepage
	return sys.stdout.encoding or ('cp1252' if is_win32 else 'latin-1')

def split_path_unix(path):
	return path.split('/')

def split_path_cygwin(path):
	if path.startswith('//'):
		ret = path.split('/')[2:]
		ret[0] = '/' + ret[0]
		return ret
	return path.split('/')

re_sp = re.compile('[/\\\\]+')
def split_path_win32(path):
	if path.startswith('\\\\'):
		ret = re_sp.split(path)[1:]
		ret[0] = '\\\\' + ret[0]
		if ret[0] == '\\\\?':
			return ret[1:]
		return ret
	return re_sp.split(path)

msysroot = None
def split_path_msys(path):
	if path.startswith(('/', '\\')) and not path.startswith(('//', '\\\\')):
		# msys paths can be in the form /usr/bin
		global msysroot
		if not msysroot:
			# msys has python 2.7 or 3, so we can use this
			msysroot = subprocess.check_output(['cygpath', '-w', '/']).decode(sys.stdout.encoding or 'latin-1')
			msysroot = msysroot.strip()
		path = os.path.normpath(msysroot + os.sep + path)
	return split_path_win32(path)

if sys.platform == 'cygwin':
	split_path = split_path_cygwin
elif is_win32:
	# Consider this an MSYSTEM environment if $MSYSTEM is set and python
	# reports is executable from a unix like path on a windows host.
	if os.environ.get('MSYSTEM') and sys.executable.startswith('/'):
		split_path = split_path_msys
	else:
		split_path = split_path_win32
else:
	split_path = split_path_unix

split_path.__doc__ = """
Splits a path by / or \\; do not confuse this function with with ``os.path.split``

:type  path: string
:param path: path to split
:return:     list of string
"""

def check_dir(path):
	"""
	Ensures that a directory exists (similar to ``mkdir -p``).

	:type  path: string
	:param path: Path to directory
	:raises: :py:class:`waflib.Errors.WafError` if the folder cannot be added.
	"""
	if not os.path.isdir(path):
		try:
			os.makedirs(path)
		except OSError as e:
			if not os.path.isdir(path):
				raise Errors.WafError('Cannot create the folder %r' % path, ex=e)

def check_exe(name, env=None):
	"""
	Ensures that a program exists

	:type name: string
	:param name: path to the program
	:param env: configuration object
	:type env: :py:class:`waflib.ConfigSet.ConfigSet`
	:return: path of the program or None
	:raises: :py:class:`waflib.Errors.WafError` if the folder cannot be added.
	"""
	if not name:
		raise ValueError('Cannot execute an empty string!')
	def is_exe(fpath):
		return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

	fpath, fname = os.path.split(name)
	if fpath and is_exe(name):
		return os.path.abspath(name)
	else:
		env = env or os.environ
		for path in env['PATH'].split(os.pathsep):
			path = path.strip('"')
			exe_file = os.path.join(path, name)
			if is_exe(exe_file):
				return os.path.abspath(exe_file)
	return None

def def_attrs(cls, **kw):
	"""
	Sets default attributes on a class instance

	:type cls: class
	:param cls: the class to update the given attributes in.
	:type kw: dict
	:param kw: dictionary of attributes names and values.
	"""
	for k, v in kw.items():
		if not hasattr(cls, k):
			setattr(cls, k, v)

def quote_define_name(s):
	"""
	Converts a string into an identifier suitable for C defines.

	:type  s: string
	:param s: String to convert
	:rtype: string
	:return: Identifier suitable for C defines
	"""
	fu = re.sub('[^a-zA-Z0-9]', '_', s)
	fu = re.sub('_+', '_', fu)
	fu = fu.upper()
	return fu

re_sh = re.compile('\\s|\'|"')
"""
Regexp used for shell_escape below
"""

def shell_escape(cmd):
	"""
	Escapes a command:
	['ls', '-l', 'arg space'] -> ls -l 'arg space'
	"""
	if isinstance(cmd, str):
		return cmd
	return ' '.join(repr(x) if re_sh.search(x) else x for x in cmd)

def h_list(lst):
	"""
	Hashes lists of ordered data.

	Using hash(tup) for tuples would be much more efficient,
	but Python now enforces hash randomization

	:param lst: list to hash
	:type lst: list of strings
	:return: hash of the list
	"""
	return md5(repr(lst).encode()).digest()

if sys.hexversion < 0x3000000:
	def h_list_python2(lst):
		return md5(repr(lst)).digest()
	h_list_python2.__doc__ = h_list.__doc__
	h_list = h_list_python2

def h_fun(fun):
	"""
	Hash functions

	:param fun: function to hash
	:type  fun: function
	:return: hash of the function
	:rtype: string or bytes
	"""
	try:
		return fun.code
	except AttributeError:
		if isinstance(fun, functools.partial):
			code = list(fun.args)
			# The method items() provides a sequence of tuples where the first element
			# represents an optional argument of the partial function application
			#
			# The sorting result outcome will be consistent because:
			# 1. tuples are compared in order of their elements
			# 2. optional argument namess are unique
			code.extend(sorted(fun.keywords.items()))
			code.append(h_fun(fun.func))
			fun.code = h_list(code)
			return fun.code
		try:
			h = inspect.getsource(fun)
		except EnvironmentError:
			h = 'nocode'
		try:
			fun.code = h
		except AttributeError:
			pass
		return h

def h_cmd(ins):
	"""
	Hashes objects recursively

	:param ins: input object
	:type ins: string or list or tuple or function
	:rtype: string or bytes
	"""
	# this function is not meant to be particularly fast
	if isinstance(ins, str):
		# a command is either a string
		ret = ins
	elif isinstance(ins, list) or isinstance(ins, tuple):
		# or a list of functions/strings
		ret = str([h_cmd(x) for x in ins])
	else:
		# or just a python function
		ret = str(h_fun(ins))
	if sys.hexversion > 0x3000000:
		ret = ret.encode('latin-1', 'xmlcharrefreplace')
	return ret

reg_subst = re.compile(r"(\\\\)|(\$\$)|\$\{([^}]+)\}")
def subst_vars(expr, params):
	"""
	Replaces ${VAR} with the value of VAR taken from a dict or a config set::

		from waflib import Utils
		s = Utils.subst_vars('${PREFIX}/bin', env)

	:type  expr: string
	:param expr: String to perform substitution on
	:param params: Dictionary or config set to look up variable values.
	"""
	def repl_var(m):
		if m.group(1):
			return '\\'
		if m.group(2):
			return '$'
		try:
			# ConfigSet instances may contain lists
			return params.get_flat(m.group(3))
		except AttributeError:
			return params[m.group(3)]
		# if you get a TypeError, it means that 'expr' is not a string...
		# Utils.subst_vars(None, env)  will not work
	return reg_subst.sub(repl_var, expr)

def destos_to_binfmt(key):
	"""
	Returns the binary format based on the unversioned platform name,
	and defaults to ``elf`` if nothing is found.

	:param key: platform name
	:type  key: string
	:return: string representing the binary format
	"""
	if key == 'darwin':
		return 'mac-o'
	elif key in ('win32', 'cygwin', 'uwin', 'msys'):
		return 'pe'
	return 'elf'

def unversioned_sys_platform():
	"""
	Returns the unversioned platform name.
	Some Python platform names contain versions, that depend on
	the build environment, e.g. linux2, freebsd6, etc.
	This returns the name without the version number. Exceptions are
	os2 and win32, which are returned verbatim.

	:rtype: string
	:return: Unversioned platform name
	"""
	s = sys.platform
	if s.startswith('java'):
		# The real OS is hidden under the JVM.
		from java.lang import System
		s = System.getProperty('os.name')
		# see http://lopica.sourceforge.net/os.html for a list of possible values
		if s == 'Mac OS X':
			return 'darwin'
		elif s.startswith('Windows '):
			return 'win32'
		elif s == 'OS/2':
			return 'os2'
		elif s == 'HP-UX':
			return 'hp-ux'
		elif s in ('SunOS', 'Solaris'):
			return 'sunos'
		else: s = s.lower()

	# powerpc == darwin for our purposes
	if s == 'powerpc':
		return 'darwin'
	if s == 'win32' or s == 'os2':
		return s
	if s == 'cli' and os.name == 'nt':
		# ironpython is only on windows as far as we know
		return 'win32'
	return re.split(r'\d+$', s)[0]

def nada(*k, **kw):
	"""
	Does nothing

	:return: None
	"""
	pass

class Timer(object):
	"""
	Simple object for timing the execution of commands.
	Its string representation is the duration::

		from waflib.Utils import Timer
		timer = Timer()
		a_few_operations()
		s = str(timer)
	"""
	def __init__(self):
		self.start_time = self.now()

	def __str__(self):
		delta = self.now() - self.start_time
		if not isinstance(delta, datetime.timedelta):
			delta = datetime.timedelta(seconds=delta)
		days = delta.days
		hours, rem = divmod(delta.seconds, 3600)
		minutes, seconds = divmod(rem, 60)
		seconds += delta.microseconds * 1e-6
		result = ''
		if days:
			result += '%dd' % days
		if days or hours:
			result += '%dh' % hours
		if days or hours or minutes:
			result += '%dm' % minutes
		return '%s%.3fs' % (result, seconds)

	def now(self):
		return datetime.datetime.utcnow()

	if hasattr(time, 'perf_counter'):
		def now(self):
			return time.perf_counter()

def read_la_file(path):
	"""
	Reads property files, used by msvc.py

	:param path: file to read
	:type path: string
	"""
	sp = re.compile(r'^([^=]+)=\'(.*)\'$')
	dc = {}
	for line in readf(path).splitlines():
		try:
			_, left, right, _ = sp.split(line.strip())
			dc[left] = right
		except ValueError:
			pass
	return dc

def run_once(fun):
	"""
	Decorator: let a function cache its results, use like this::

		@run_once
		def foo(k):
			return 345*2343

	.. note:: in practice this can cause memory leaks, prefer a :py:class:`waflib.Utils.lru_cache`

	:param fun: function to execute
	:type fun: function
	:return: the return value of the function executed
	"""
	cache = {}
	def wrap(*k):
		try:
			return cache[k]
		except KeyError:
			ret = fun(*k)
			cache[k] = ret
			return ret
	wrap.__cache__ = cache
	wrap.__name__ = fun.__name__
	return wrap

def get_registry_app_path(key, filename):
	"""
	Returns the value of a registry key for an executable

	:type key: string
	:type filename: list of string
	"""
	if not winreg:
		return None
	try:
		result = winreg.QueryValue(key, "Software\\Microsoft\\Windows\\CurrentVersion\\App Paths\\%s.exe" % filename[0])
	except OSError:
		pass
	else:
		if os.path.isfile(result):
			return result

def lib64():
	"""
	Guess the default ``/usr/lib`` extension for 64-bit applications

	:return: '64' or ''
	:rtype: string
	"""
	# default settings for /usr/lib
	if os.sep == '/':
		if platform.architecture()[0] == '64bit':
			if os.path.exists('/usr/lib64') and not os.path.exists('/usr/lib32'):
				return '64'
	return ''

def sane_path(p):
	# private function for the time being!
	return os.path.abspath(os.path.expanduser(p))

process_pool = []
"""
List of processes started to execute sub-process commands
"""

def get_process():
	"""
	Returns a process object that can execute commands as sub-processes

	:rtype: subprocess.Popen
	"""
	try:
		return process_pool.pop()
	except IndexError:
		filepath = os.path.dirname(os.path.abspath(__file__)) + os.sep + 'processor.py'
		cmd = [sys.executable, '-c', readf(filepath)]
		return subprocess.Popen(cmd, stdout=subprocess.PIPE, stdin=subprocess.PIPE, bufsize=0, close_fds=not is_win32)

def run_prefork_process(cmd, kwargs, cargs):
	"""
	Delegates process execution to a pre-forked process instance.
	"""
	if not 'env' in kwargs:
		kwargs['env'] = dict(os.environ)
	try:
		obj = base64.b64encode(cPickle.dumps([cmd, kwargs, cargs]))
	except (TypeError, AttributeError):
		return run_regular_process(cmd, kwargs, cargs)

	proc = get_process()
	if not proc:
		return run_regular_process(cmd, kwargs, cargs)

	proc.stdin.write(obj)
	proc.stdin.write('\n'.encode())
	proc.stdin.flush()
	obj = proc.stdout.readline()
	if not obj:
		raise OSError('Preforked sub-process %r died' % proc.pid)

	process_pool.append(proc)
	lst = cPickle.loads(base64.b64decode(obj))
	# Jython wrapper failures (bash/execvp)
	assert len(lst) == 5
	ret, out, err, ex, trace = lst
	if ex:
		if ex == 'OSError':
			raise OSError(trace)
		elif ex == 'ValueError':
			raise ValueError(trace)
		elif ex == 'TimeoutExpired':
			exc = TimeoutExpired(cmd, timeout=cargs['timeout'], output=out)
			exc.stderr = err
			raise exc
		else:
			raise Exception(trace)
	return ret, out, err

def lchown(path, user=-1, group=-1):
	"""
	Change the owner/group of a path, raises an OSError if the
	ownership change fails.

	:param user: user to change
	:type user: int or str
	:param group: group to change
	:type group: int or str
	"""
	if isinstance(user, str):
		import pwd
		entry = pwd.getpwnam(user)
		if not entry:
			raise OSError('Unknown user %r' % user)
		user = entry[2]
	if isinstance(group, str):
		import grp
		entry = grp.getgrnam(group)
		if not entry:
			raise OSError('Unknown group %r' % group)
		group = entry[2]
	return os.lchown(path, user, group)

def run_regular_process(cmd, kwargs, cargs={}):
	"""
	Executes a subprocess command by using subprocess.Popen
	"""
	proc = subprocess.Popen(cmd, **kwargs)
	if kwargs.get('stdout') or kwargs.get('stderr'):
		try:
			out, err = proc.communicate(**cargs)
		except TimeoutExpired:
			if kwargs.get('start_new_session') and hasattr(os, 'killpg'):
				os.killpg(proc.pid, signal.SIGKILL)
			else:
				proc.kill()
			out, err = proc.communicate()
			exc = TimeoutExpired(proc.args, timeout=cargs['timeout'], output=out)
			exc.stderr = err
			raise exc
		status = proc.returncode
	else:
		out, err = (None, None)
		try:
			status = proc.wait(**cargs)
		except TimeoutExpired as e:
			if kwargs.get('start_new_session') and hasattr(os, 'killpg'):
				os.killpg(proc.pid, signal.SIGKILL)
			else:
				proc.kill()
			proc.wait()
			raise e
	return status, out, err

def run_process(cmd, kwargs, cargs={}):
	"""
	Executes a subprocess by using a pre-forked process when possible
	or falling back to subprocess.Popen. See :py:func:`waflib.Utils.run_prefork_process`
	and :py:func:`waflib.Utils.run_regular_process`
	"""
	if kwargs.get('stdout') and kwargs.get('stderr'):
		return run_prefork_process(cmd, kwargs, cargs)
	else:
		return run_regular_process(cmd, kwargs, cargs)

def alloc_process_pool(n, force=False):
	"""
	Allocates an amount of processes to the default pool so its size is at least *n*.
	It is useful to call this function early so that the pre-forked
	processes use as little memory as possible.

	:param n: pool size
	:type n: integer
	:param force: if True then *n* more processes are added to the existing pool
	:type force: bool
	"""
	# mandatory on python2, unnecessary on python >= 3.2
	global run_process, get_process, alloc_process_pool
	if not force:
		n = max(n - len(process_pool), 0)
	try:
		lst = [get_process() for x in range(n)]
	except OSError:
		run_process = run_regular_process
		get_process = alloc_process_pool = nada
	else:
		for x in lst:
			process_pool.append(x)

def atexit_pool():
	for k in process_pool:
		try:
			os.kill(k.pid, 9)
		except OSError:
			pass
		else:
			k.wait()
# see #1889
if (sys.hexversion<0x207000f and not is_win32) or sys.hexversion>=0x306000f:
	atexit.register(atexit_pool)

if os.environ.get('WAF_NO_PREFORK') or sys.platform == 'cli' or not sys.executable:
	run_process = run_regular_process
	get_process = alloc_process_pool = nada

