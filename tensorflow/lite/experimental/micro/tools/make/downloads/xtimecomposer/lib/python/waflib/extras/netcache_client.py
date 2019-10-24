#! /usr/bin/env python
# encoding: utf-8
# Thomas Nagy, 2011-2015 (ita)

"""
A client for the network cache (playground/netcache/). Launch the server with:
./netcache_server, then use it for the builds by adding the following:

	def build(bld):
		bld.load('netcache_client')

The parameters should be present in the environment in the form:
	NETCACHE=host:port waf configure build

Or in a more detailed way:
	NETCACHE_PUSH=host:port NETCACHE_PULL=host:port waf configure build

where:
	host: host where the server resides, by default localhost
	port: by default push on 11001 and pull on 12001

Use the server provided in playground/netcache/Netcache.java
"""

import os, socket, time, atexit, sys
from waflib import Task, Logs, Utils, Build, Runner
from waflib.Configure import conf

BUF = 8192 * 16
HEADER_SIZE = 128
MODES = ['PUSH', 'PULL', 'PUSH_PULL']
STALE_TIME = 30 # seconds

GET = 'GET'
PUT = 'PUT'
LST = 'LST'
BYE = 'BYE'

all_sigs_in_cache = (0.0, [])

def put_data(conn, data):
	if sys.hexversion > 0x3000000:
		data = data.encode('latin-1')
	cnt = 0
	while cnt < len(data):
		sent = conn.send(data[cnt:])
		if sent == 0:
			raise RuntimeError('connection ended')
		cnt += sent

push_connections = Runner.Queue(0)
pull_connections = Runner.Queue(0)
def get_connection(push=False):
	# return a new connection... do not forget to release it!
	try:
		if push:
			ret = push_connections.get(block=False)
		else:
			ret = pull_connections.get(block=False)
	except Exception:
		ret = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		if push:
			ret.connect(Task.push_addr)
		else:
			ret.connect(Task.pull_addr)
	return ret

def release_connection(conn, msg='', push=False):
	if conn:
		if push:
			push_connections.put(conn)
		else:
			pull_connections.put(conn)

def close_connection(conn, msg=''):
	if conn:
		data = '%s,%s' % (BYE, msg)
		try:
			put_data(conn, data.ljust(HEADER_SIZE))
		except:
			pass
		try:
			conn.close()
		except:
			pass

def close_all():
	for q in (push_connections, pull_connections):
		while q.qsize():
			conn = q.get()
			try:
				close_connection(conn)
			except:
				# ignore errors when cleaning up
				pass
atexit.register(close_all)

def read_header(conn):
	cnt = 0
	buf = []
	while cnt < HEADER_SIZE:
		data = conn.recv(HEADER_SIZE - cnt)
		if not data:
			#import traceback
			#traceback.print_stack()
			raise ValueError('connection ended when reading a header %r' % buf)
		buf.append(data)
		cnt += len(data)
	if sys.hexversion > 0x3000000:
		ret = ''.encode('latin-1').join(buf)
		ret = ret.decode('latin-1')
	else:
		ret = ''.join(buf)
	return ret

def check_cache(conn, ssig):
	"""
	List the files on the server, this is an optimization because it assumes that
	concurrent builds are rare
	"""
	global all_sigs_in_cache
	if not STALE_TIME:
		return
	if time.time() - all_sigs_in_cache[0] > STALE_TIME:

		params = (LST,'')
		put_data(conn, ','.join(params).ljust(HEADER_SIZE))

		# read what is coming back
		ret = read_header(conn)
		size = int(ret.split(',')[0])

		buf = []
		cnt = 0
		while cnt < size:
			data = conn.recv(min(BUF, size-cnt))
			if not data:
				raise ValueError('connection ended %r %r' % (cnt, size))
			buf.append(data)
			cnt += len(data)

		if sys.hexversion > 0x3000000:
			ret = ''.encode('latin-1').join(buf)
			ret = ret.decode('latin-1')
		else:
			ret = ''.join(buf)

		all_sigs_in_cache = (time.time(), ret.splitlines())
		Logs.debug('netcache: server cache has %r entries', len(all_sigs_in_cache[1]))

	if not ssig in all_sigs_in_cache[1]:
		raise ValueError('no file %s in cache' % ssig)

class MissingFile(Exception):
	pass

def recv_file(conn, ssig, count, p):
	check_cache(conn, ssig)

	params = (GET, ssig, str(count))
	put_data(conn, ','.join(params).ljust(HEADER_SIZE))
	data = read_header(conn)

	size = int(data.split(',')[0])

	if size == -1:
		raise MissingFile('no file %s - %s in cache' % (ssig, count))

	# get the file, writing immediately
	# TODO a tmp file would be better
	f = open(p, 'wb')
	cnt = 0
	while cnt < size:
		data = conn.recv(min(BUF, size-cnt))
		if not data:
			raise ValueError('connection ended %r %r' % (cnt, size))
		f.write(data)
		cnt += len(data)
	f.close()

def sock_send(conn, ssig, cnt, p):
	#print "pushing %r %r %r" % (ssig, cnt, p)
	size = os.stat(p).st_size
	params = (PUT, ssig, str(cnt), str(size))
	put_data(conn, ','.join(params).ljust(HEADER_SIZE))
	f = open(p, 'rb')
	cnt = 0
	while cnt < size:
		r = f.read(min(BUF, size-cnt))
		while r:
			k = conn.send(r)
			if not k:
				raise ValueError('connection ended')
			cnt += k
			r = r[k:]

def can_retrieve_cache(self):
	if not Task.pull_addr:
		return False
	if not self.outputs:
		return False
	self.cached = False

	cnt = 0
	sig = self.signature()
	ssig = Utils.to_hex(self.uid() + sig)

	conn = None
	err = False
	try:
		try:
			conn = get_connection()
			for node in self.outputs:
				p = node.abspath()
				recv_file(conn, ssig, cnt, p)
				cnt += 1
		except MissingFile as e:
			Logs.debug('netcache: file is not in the cache %r', e)
			err = True
		except Exception as e:
			Logs.debug('netcache: could not get the files %r', self.outputs)
			if Logs.verbose > 1:
				Logs.debug('netcache: exception %r', e)
			err = True

			# broken connection? remove this one
			close_connection(conn)
			conn = None
		else:
			Logs.debug('netcache: obtained %r from cache', self.outputs)

	finally:
		release_connection(conn)
	if err:
		return False

	self.cached = True
	return True

@Utils.run_once
def put_files_cache(self):
	if not Task.push_addr:
		return
	if not self.outputs:
		return
	if getattr(self, 'cached', None):
		return

	#print "called put_files_cache", id(self)
	bld = self.generator.bld
	sig = self.signature()
	ssig = Utils.to_hex(self.uid() + sig)

	conn = None
	cnt = 0
	try:
		for node in self.outputs:
			# We could re-create the signature of the task with the signature of the outputs
			# in practice, this means hashing the output files
			# this is unnecessary
			try:
				if not conn:
					conn = get_connection(push=True)
				sock_send(conn, ssig, cnt, node.abspath())
				Logs.debug('netcache: sent %r', node)
			except Exception as e:
				Logs.debug('netcache: could not push the files %r', e)

				# broken connection? remove this one
				close_connection(conn)
				conn = None
			cnt += 1
	finally:
		release_connection(conn, push=True)

	bld.task_sigs[self.uid()] = self.cache_sig

def hash_env_vars(self, env, vars_lst):
	# reimplement so that the resulting hash does not depend on local paths
	if not env.table:
		env = env.parent
		if not env:
			return Utils.SIG_NIL

	idx = str(id(env)) + str(vars_lst)
	try:
		cache = self.cache_env
	except AttributeError:
		cache = self.cache_env = {}
	else:
		try:
			return self.cache_env[idx]
		except KeyError:
			pass

	v = str([env[a] for a in vars_lst])
	v = v.replace(self.srcnode.abspath().__repr__()[:-1], '')
	m = Utils.md5()
	m.update(v.encode())
	ret = m.digest()

	Logs.debug('envhash: %r %r', ret, v)

	cache[idx] = ret

	return ret

def uid(self):
	# reimplement so that the signature does not depend on local paths
	try:
		return self.uid_
	except AttributeError:
		m = Utils.md5()
		src = self.generator.bld.srcnode
		up = m.update
		up(self.__class__.__name__.encode())
		for x in self.inputs + self.outputs:
			up(x.path_from(src).encode())
		self.uid_ = m.digest()
		return self.uid_


def make_cached(cls):
	if getattr(cls, 'nocache', None):
		return

	m1 = cls.run
	def run(self):
		if getattr(self, 'nocache', False):
			return m1(self)
		if self.can_retrieve_cache():
			return 0
		return m1(self)
	cls.run = run

	m2 = cls.post_run
	def post_run(self):
		if getattr(self, 'nocache', False):
			return m2(self)
		bld = self.generator.bld
		ret = m2(self)
		if bld.cache_global:
			self.put_files_cache()
		if hasattr(self, 'chmod'):
			for node in self.outputs:
				os.chmod(node.abspath(), self.chmod)
		return ret
	cls.post_run = post_run

@conf
def setup_netcache(ctx, push_addr, pull_addr):
	Task.Task.can_retrieve_cache = can_retrieve_cache
	Task.Task.put_files_cache = put_files_cache
	Task.Task.uid = uid
	Task.push_addr = push_addr
	Task.pull_addr = pull_addr
	Build.BuildContext.hash_env_vars = hash_env_vars
	ctx.cache_global = True

	for x in Task.classes.values():
		make_cached(x)

def build(bld):
	if not 'NETCACHE' in os.environ and not 'NETCACHE_PULL' in os.environ and not 'NETCACHE_PUSH' in os.environ:
		Logs.warn('Setting  NETCACHE_PULL=127.0.0.1:11001 and NETCACHE_PUSH=127.0.0.1:12001')
		os.environ['NETCACHE_PULL'] = '127.0.0.1:12001'
		os.environ['NETCACHE_PUSH'] = '127.0.0.1:11001'

	if 'NETCACHE' in os.environ:
		if not 'NETCACHE_PUSH' in os.environ:
			os.environ['NETCACHE_PUSH'] = os.environ['NETCACHE']
		if not 'NETCACHE_PULL' in os.environ:
			os.environ['NETCACHE_PULL'] = os.environ['NETCACHE']

	v = os.environ['NETCACHE_PULL']
	if v:
		h, p = v.split(':')
		pull_addr = (h, int(p))
	else:
		pull_addr = None

	v = os.environ['NETCACHE_PUSH']
	if v:
		h, p = v.split(':')
		push_addr = (h, int(p))
	else:
		push_addr = None

	setup_netcache(bld, push_addr, pull_addr)

