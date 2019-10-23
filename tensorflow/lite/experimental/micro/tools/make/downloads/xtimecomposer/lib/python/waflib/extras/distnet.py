#! /usr/bin/env python
# encoding: utf-8

"""
waf-powered distributed network builds, with a network cache.

Caching files from a server has advantages over a NFS/Samba shared folder:

- builds are much faster because they use local files
- builds just continue to work in case of a network glitch
- permissions are much simpler to manage
"""

import os, urllib, tarfile, re, shutil, tempfile, sys
from collections import OrderedDict
from waflib import Context, Utils, Logs

try:
	from urllib.parse import urlencode
except ImportError:
	urlencode = urllib.urlencode

def safe_urlencode(data):
	x = urlencode(data)
	try:
		x = x.encode('utf-8')
	except Exception:
		pass
	return x

try:
	from urllib.error import URLError
except ImportError:
	from urllib2 import URLError

try:
	from urllib.request import Request, urlopen
except ImportError:
	from urllib2 import Request, urlopen

DISTNETCACHE = os.environ.get('DISTNETCACHE', '/tmp/distnetcache')
DISTNETSERVER = os.environ.get('DISTNETSERVER', 'http://localhost:8000/cgi-bin/')
TARFORMAT = 'w:bz2'
TIMEOUT = 60
REQUIRES = 'requires.txt'

re_com = re.compile(r'\s*#.*', re.M)

def total_version_order(num):
	lst = num.split('.')
	template = '%10s' * len(lst)
	ret = template % tuple(lst)
	return ret

def get_distnet_cache():
	return getattr(Context.g_module, 'DISTNETCACHE', DISTNETCACHE)

def get_server_url():
	return getattr(Context.g_module, 'DISTNETSERVER', DISTNETSERVER)

def get_download_url():
	return '%s/download.py' % get_server_url()

def get_upload_url():
	return '%s/upload.py' % get_server_url()

def get_resolve_url():
	return '%s/resolve.py' % get_server_url()

def send_package_name():
	out = getattr(Context.g_module, 'out', 'build')
	pkgfile = '%s/package_to_upload.tarfile' % out
	return pkgfile

class package(Context.Context):
	fun = 'package'
	cmd = 'package'

	def execute(self):
		try:
			files = self.files
		except AttributeError:
			files = self.files = []

		Context.Context.execute(self)
		pkgfile = send_package_name()
		if not pkgfile in files:
			if not REQUIRES in files:
				files.append(REQUIRES)
			self.make_tarfile(pkgfile, files, add_to_package=False)

	def make_tarfile(self, filename, files, **kw):
		if kw.get('add_to_package', True):
			self.files.append(filename)

		with tarfile.open(filename, TARFORMAT) as tar:
			endname = os.path.split(filename)[-1]
			endname = endname.split('.')[0] + '/'
			for x in files:
				tarinfo = tar.gettarinfo(x, x)
				tarinfo.uid   = tarinfo.gid   = 0
				tarinfo.uname = tarinfo.gname = 'root'
				tarinfo.size = os.stat(x).st_size

				# TODO - more archive creation options?
				if kw.get('bare', True):
					tarinfo.name = os.path.split(x)[1]
				else:
					tarinfo.name = endname + x # todo, if tuple, then..
				Logs.debug('distnet: adding %r to %s', tarinfo.name, filename)
				with open(x, 'rb') as f:
					tar.addfile(tarinfo, f)
		Logs.info('Created %s', filename)

class publish(Context.Context):
	fun = 'publish'
	cmd = 'publish'
	def execute(self):
		if hasattr(Context.g_module, 'publish'):
			Context.Context.execute(self)
		mod = Context.g_module

		rfile = getattr(self, 'rfile', send_package_name())
		if not os.path.isfile(rfile):
			self.fatal('Create the release file with "waf release" first! %r' % rfile)

		fdata = Utils.readf(rfile, m='rb')
		data = safe_urlencode([('pkgdata', fdata), ('pkgname', mod.APPNAME), ('pkgver', mod.VERSION)])

		req = Request(get_upload_url(), data)
		response = urlopen(req, timeout=TIMEOUT)
		data = response.read().strip()

		if sys.hexversion>0x300000f:
			data = data.decode('utf-8')

		if data != 'ok':
			self.fatal('Could not publish the package %r' % data)

class constraint(object):
	def __init__(self, line=''):
		self.required_line = line
		self.info = []

		line = line.strip()
		if not line:
			return

		lst = line.split(',')
		if lst:
			self.pkgname = lst[0]
			self.required_version = lst[1]
			for k in lst:
				a, b, c = k.partition('=')
				if a and c:
					self.info.append((a, c))
	def __str__(self):
		buf = []
		buf.append(self.pkgname)
		buf.append(self.required_version)
		for k in self.info:
			buf.append('%s=%s' % k)
		return ','.join(buf)

	def __repr__(self):
		return "requires %s-%s" % (self.pkgname, self.required_version)

	def human_display(self, pkgname, pkgver):
		return '%s-%s requires %s-%s' % (pkgname, pkgver, self.pkgname, self.required_version)

	def why(self):
		ret = []
		for x in self.info:
			if x[0] == 'reason':
				ret.append(x[1])
		return ret

	def add_reason(self, reason):
		self.info.append(('reason', reason))

def parse_constraints(text):
	assert(text is not None)
	constraints = []
	text = re.sub(re_com, '', text)
	lines = text.splitlines()
	for line in lines:
		line = line.strip()
		if not line:
			continue
		constraints.append(constraint(line))
	return constraints

def list_package_versions(cachedir, pkgname):
	pkgdir = os.path.join(cachedir, pkgname)
	try:
		versions = os.listdir(pkgdir)
	except OSError:
		return []
	versions.sort(key=total_version_order)
	versions.reverse()
	return versions

class package_reader(Context.Context):
	cmd = 'solver'
	fun = 'solver'

	def __init__(self, **kw):
		Context.Context.__init__(self, **kw)

		self.myproject = getattr(Context.g_module, 'APPNAME', 'project')
		self.myversion = getattr(Context.g_module, 'VERSION', '1.0')
		self.cache_constraints = {}
		self.constraints = []

	def compute_dependencies(self, filename=REQUIRES):
		text = Utils.readf(filename)
		data = safe_urlencode([('text', text)])

		if '--offline' in sys.argv:
			self.constraints = self.local_resolve(text)
		else:
			req = Request(get_resolve_url(), data)
			try:
				response = urlopen(req, timeout=TIMEOUT)
			except URLError as e:
				Logs.warn('The package server is down! %r', e)
				self.constraints = self.local_resolve(text)
			else:
				ret = response.read()
				try:
					ret = ret.decode('utf-8')
				except Exception:
					pass
				self.trace(ret)
				self.constraints = parse_constraints(ret)
		self.check_errors()

	def check_errors(self):
		errors = False
		for c in self.constraints:
			if not c.required_version:
				errors = True

				reasons = c.why()
				if len(reasons) == 1:
					Logs.error('%s but no matching package could be found in this repository', reasons[0])
				else:
					Logs.error('Conflicts on package %r:', c.pkgname)
					for r in reasons:
						Logs.error('  %s', r)
		if errors:
			self.fatal('The package requirements cannot be satisfied!')

	def load_constraints(self, pkgname, pkgver, requires=REQUIRES):
		try:
			return self.cache_constraints[(pkgname, pkgver)]
		except KeyError:
			text = Utils.readf(os.path.join(get_distnet_cache(), pkgname, pkgver, requires))
			ret = parse_constraints(text)
			self.cache_constraints[(pkgname, pkgver)] = ret
			return ret

	def apply_constraint(self, domain, constraint):
		vname = constraint.required_version.replace('*', '.*')
		rev = re.compile(vname, re.M)
		ret = [x for x in domain if rev.match(x)]
		return ret

	def trace(self, *k):
		if getattr(self, 'debug', None):
			Logs.error(*k)

	def solve(self, packages_to_versions={}, packages_to_constraints={}, pkgname='', pkgver='', todo=[], done=[]):
		# breadth first search
		n_packages_to_versions = dict(packages_to_versions)
		n_packages_to_constraints = dict(packages_to_constraints)

		self.trace("calling solve with %r    %r %r" % (packages_to_versions, todo, done))
		done = done + [pkgname]

		constraints = self.load_constraints(pkgname, pkgver)
		self.trace("constraints %r" % constraints)

		for k in constraints:
			try:
				domain = n_packages_to_versions[k.pkgname]
			except KeyError:
				domain = list_package_versions(get_distnet_cache(), k.pkgname)


			self.trace("constraints?")
			if not k.pkgname in done:
				todo = todo + [k.pkgname]

			self.trace("domain before %s -> %s, %r" % (pkgname, k.pkgname, domain))

			# apply the constraint
			domain = self.apply_constraint(domain, k)

			self.trace("domain after %s -> %s, %r" % (pkgname, k.pkgname, domain))

			n_packages_to_versions[k.pkgname] = domain

			# then store the constraint applied
			constraints = list(packages_to_constraints.get(k.pkgname, []))
			constraints.append((pkgname, pkgver, k))
			n_packages_to_constraints[k.pkgname] = constraints

			if not domain:
				self.trace("no domain while processing constraint %r from %r %r" % (domain, pkgname, pkgver))
				return (n_packages_to_versions, n_packages_to_constraints)

		# next package on the todo list
		if not todo:
			return (n_packages_to_versions, n_packages_to_constraints)

		n_pkgname = todo[0]
		n_pkgver = n_packages_to_versions[n_pkgname][0]
		tmp = dict(n_packages_to_versions)
		tmp[n_pkgname] = [n_pkgver]

		self.trace("fixed point %s" % n_pkgname)

		return self.solve(tmp, n_packages_to_constraints, n_pkgname, n_pkgver, todo[1:], done)

	def get_results(self):
		return '\n'.join([str(c) for c in self.constraints])

	def solution_to_constraints(self, versions, constraints):
		solution = []
		for p in versions:
			c = constraint()
			solution.append(c)

			c.pkgname = p
			if versions[p]:
				c.required_version = versions[p][0]
			else:
				c.required_version = ''
			for (from_pkgname, from_pkgver, c2) in constraints.get(p, ''):
				c.add_reason(c2.human_display(from_pkgname, from_pkgver))
		return solution

	def local_resolve(self, text):
		self.cache_constraints[(self.myproject, self.myversion)] = parse_constraints(text)
		p2v = OrderedDict({self.myproject: [self.myversion]})
		(versions, constraints) = self.solve(p2v, {}, self.myproject, self.myversion, [])
		return self.solution_to_constraints(versions, constraints)

	def download_to_file(self, pkgname, pkgver, subdir, tmp):
		data = safe_urlencode([('pkgname', pkgname), ('pkgver', pkgver), ('pkgfile', subdir)])
		req = urlopen(get_download_url(), data, timeout=TIMEOUT)
		with open(tmp, 'wb') as f:
			while True:
				buf = req.read(8192)
				if not buf:
					break
				f.write(buf)

	def extract_tar(self, subdir, pkgdir, tmpfile):
		with tarfile.open(tmpfile) as f:
			temp = tempfile.mkdtemp(dir=pkgdir)
			try:
				f.extractall(temp)
				os.rename(temp, os.path.join(pkgdir, subdir))
			finally:
				try:
					shutil.rmtree(temp)
				except Exception:
					pass

	def get_pkg_dir(self, pkgname, pkgver, subdir):
		pkgdir = os.path.join(get_distnet_cache(), pkgname, pkgver)
		if not os.path.isdir(pkgdir):
			os.makedirs(pkgdir)

		target = os.path.join(pkgdir, subdir)

		if os.path.exists(target):
			return target

		(fd, tmp) = tempfile.mkstemp(dir=pkgdir)
		try:
			os.close(fd)
			self.download_to_file(pkgname, pkgver, subdir, tmp)
			if subdir == REQUIRES:
				os.rename(tmp, target)
			else:
				self.extract_tar(subdir, pkgdir, tmp)
		finally:
			try:
				os.remove(tmp)
			except OSError:
				pass

		return target

	def __iter__(self):
		if not self.constraints:
			self.compute_dependencies()
		for x in self.constraints:
			if x.pkgname == self.myproject:
				continue
			yield x

	def execute(self):
		self.compute_dependencies()

packages = package_reader()

def load_tools(ctx, extra):
	global packages
	for c in packages:
		packages.get_pkg_dir(c.pkgname, c.required_version, extra)
		noarchdir = packages.get_pkg_dir(c.pkgname, c.required_version, 'noarch')
		for x in os.listdir(noarchdir):
			if x.startswith('waf_') and x.endswith('.py'):
				ctx.load([x.rstrip('.py')], tooldir=[noarchdir])

def options(opt):
	opt.add_option('--offline', action='store_true')
	packages.execute()
	load_tools(opt, REQUIRES)

def configure(conf):
	load_tools(conf, conf.variant)

def build(bld):
	load_tools(bld, bld.variant)

