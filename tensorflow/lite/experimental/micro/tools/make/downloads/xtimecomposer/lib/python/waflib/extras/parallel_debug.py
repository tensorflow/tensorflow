#! /usr/bin/env python
# encoding: utf-8
# Thomas Nagy, 2007-2010 (ita)

"""
Debugging helper for parallel compilation.

Copy it to your project and load it with::

	def options(opt):
		opt.load('parallel_debug', tooldir='.')
	def build(bld):
		...

The build will then output a file named pdebug.svg in the source directory.
"""

import re, sys, threading, time, traceback
try:
	from Queue import Queue
except:
	from queue import Queue
from waflib import Runner, Options, Task, Logs, Errors

SVG_TEMPLATE = """<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.0//EN" "http://www.w3.org/TR/2001/REC-SVG-20010904/DTD/svg10.dtd">
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" version="1.0"
   x="${project.x}" y="${project.y}" width="${project.width}" height="${project.height}" id="svg602" xml:space="preserve">

<style type='text/css' media='screen'>
	g.over rect { stroke:#FF0000; fill-opacity:0.4 }
</style>

<script type='text/javascript'><![CDATA[
var svg  = document.getElementsByTagName('svg')[0];

svg.addEventListener('mouseover', function(e) {
	var g = e.target.parentNode;
	var x = document.getElementById('r_' + g.id);
	if (x) {
		g.setAttribute('class', g.getAttribute('class') + ' over');
		x.setAttribute('class', x.getAttribute('class') + ' over');
		showInfo(e, g.id, e.target.attributes.tooltip.value);
	}
}, false);

svg.addEventListener('mouseout', function(e) {
		var g = e.target.parentNode;
		var x = document.getElementById('r_' + g.id);
		if (x) {
			g.setAttribute('class', g.getAttribute('class').replace(' over', ''));
			x.setAttribute('class', x.getAttribute('class').replace(' over', ''));
			hideInfo(e);
		}
}, false);

function showInfo(evt, txt, details) {
${if project.tooltip}
	tooltip = document.getElementById('tooltip');

	var t = document.getElementById('tooltiptext');
	t.firstChild.data = txt + " " + details;

	var x = evt.clientX + 9;
	if (x > 250) { x -= t.getComputedTextLength() + 16; }
	var y = evt.clientY + 20;
	tooltip.setAttribute("transform", "translate(" + x + "," + y + ")");
	tooltip.setAttributeNS(null, "visibility", "visible");

	var r = document.getElementById('tooltiprect');
	r.setAttribute('width', t.getComputedTextLength() + 6);
${endif}
}

function hideInfo(evt) {
	var tooltip = document.getElementById('tooltip');
	tooltip.setAttributeNS(null,"visibility","hidden");
}
]]></script>

<!-- inkscape requires a big rectangle or it will not export the pictures properly -->
<rect
   x='${project.x}' y='${project.y}' width='${project.width}' height='${project.height}'
   style="font-size:10;fill:#ffffff;fill-opacity:0.01;fill-rule:evenodd;stroke:#ffffff;"></rect>

${if project.title}
  <text x="${project.title_x}" y="${project.title_y}"
    style="font-size:15px; text-anchor:middle; font-style:normal;font-weight:normal;fill:#000000;fill-opacity:1;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1;font-family:Bitstream Vera Sans">${project.title}</text>
${endif}


${for cls in project.groups}
  <g id='${cls.classname}'>
    ${for rect in cls.rects}
    <rect x='${rect.x}' y='${rect.y}' width='${rect.width}' height='${rect.height}' tooltip='${rect.name}' style="font-size:10;fill:${rect.color};fill-rule:evenodd;stroke:#000000;stroke-width:0.4;" />
    ${endfor}
  </g>
${endfor}

${for info in project.infos}
  <g id='r_${info.classname}'>
   <rect x='${info.x}' y='${info.y}' width='${info.width}' height='${info.height}' style="font-size:10;fill:${info.color};fill-rule:evenodd;stroke:#000000;stroke-width:0.4;" />
   <text x="${info.text_x}" y="${info.text_y}"
       style="font-size:12px;font-style:normal;font-weight:normal;fill:#000000;fill-opacity:1;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1;font-family:Bitstream Vera Sans"
   >${info.text}</text>
  </g>
${endfor}

${if project.tooltip}
  <g transform="translate(0,0)" visibility="hidden" id="tooltip">
       <rect id="tooltiprect" y="-15" x="-3" width="1" height="20" style="stroke:black;fill:#edefc2;stroke-width:1"/>
       <text id="tooltiptext" style="font-family:Arial; font-size:12;fill:black;"> </text>
  </g>
${endif}

</svg>
"""

COMPILE_TEMPLATE = '''def f(project):
	lst = []
	def xml_escape(value):
		return value.replace("&", "&amp;").replace('"', "&quot;").replace("'", "&apos;").replace("<", "&lt;").replace(">", "&gt;")

	%s
	return ''.join(lst)
'''
reg_act = re.compile(r"(?P<backslash>\\)|(?P<dollar>\$\$)|(?P<subst>\$\{(?P<code>[^}]*?)\})", re.M)
def compile_template(line):

	extr = []
	def repl(match):
		g = match.group
		if g('dollar'):
			return "$"
		elif g('backslash'):
			return "\\"
		elif g('subst'):
			extr.append(g('code'))
			return "<<|@|>>"
		return None

	line2 = reg_act.sub(repl, line)
	params = line2.split('<<|@|>>')
	assert(extr)


	indent = 0
	buf = []
	app = buf.append

	def app(txt):
		buf.append(indent * '\t' + txt)

	for x in range(len(extr)):
		if params[x]:
			app("lst.append(%r)" % params[x])

		f = extr[x]
		if f.startswith(('if', 'for')):
			app(f + ':')
			indent += 1
		elif f.startswith('py:'):
			app(f[3:])
		elif f.startswith(('endif', 'endfor')):
			indent -= 1
		elif f.startswith(('else', 'elif')):
			indent -= 1
			app(f + ':')
			indent += 1
		elif f.startswith('xml:'):
			app('lst.append(xml_escape(%s))' % f[4:])
		else:
			#app('lst.append((%s) or "cannot find %s")' % (f, f))
			app('lst.append(str(%s))' % f)

	if extr:
		if params[-1]:
			app("lst.append(%r)" % params[-1])

	fun = COMPILE_TEMPLATE % "\n\t".join(buf)
	# uncomment the following to debug the template
	#for i, x in enumerate(fun.splitlines()):
	#	print i, x
	return Task.funex(fun)

# red   #ff4d4d
# green #4da74d
# lila  #a751ff

color2code = {
	'GREEN'  : '#4da74d',
	'YELLOW' : '#fefe44',
	'PINK'   : '#a751ff',
	'RED'    : '#cc1d1d',
	'BLUE'   : '#6687bb',
	'CYAN'   : '#34e2e2',
}

mp = {}
info = [] # list of (text,color)

def map_to_color(name):
	if name in mp:
		return mp[name]
	try:
		cls = Task.classes[name]
	except KeyError:
		return color2code['RED']
	if cls.color in mp:
		return mp[cls.color]
	if cls.color in color2code:
		return color2code[cls.color]
	return color2code['RED']

def process(self):
	m = self.generator.bld.producer
	try:
		# TODO another place for this?
		del self.generator.bld.task_sigs[self.uid()]
	except KeyError:
		pass

	self.generator.bld.producer.set_running(1, self)

	try:
		ret = self.run()
	except Exception:
		self.err_msg = traceback.format_exc()
		self.hasrun = Task.EXCEPTION

		# TODO cleanup
		m.error_handler(self)
		return

	if ret:
		self.err_code = ret
		self.hasrun = Task.CRASHED
	else:
		try:
			self.post_run()
		except Errors.WafError:
			pass
		except Exception:
			self.err_msg = traceback.format_exc()
			self.hasrun = Task.EXCEPTION
		else:
			self.hasrun = Task.SUCCESS
	if self.hasrun != Task.SUCCESS:
		m.error_handler(self)

	self.generator.bld.producer.set_running(-1, self)

Task.Task.process_back = Task.Task.process
Task.Task.process = process

old_start = Runner.Parallel.start
def do_start(self):
	try:
		Options.options.dband
	except AttributeError:
		self.bld.fatal('use def options(opt): opt.load("parallel_debug")!')

	self.taskinfo = Queue()
	old_start(self)
	if self.dirty:
		make_picture(self)
Runner.Parallel.start = do_start

lock_running = threading.Lock()
def set_running(self, by, tsk):
	with lock_running:
		try:
			cache = self.lock_cache
		except AttributeError:
			cache = self.lock_cache = {}

		i = 0
		if by > 0:
			vals = cache.values()
			for i in range(self.numjobs):
				if i not in vals:
					cache[tsk] = i
					break
		else:
			i = cache[tsk]
			del cache[tsk]

		self.taskinfo.put( (i, id(tsk), time.time(), tsk.__class__.__name__, self.processed, self.count, by, ",".join(map(str, tsk.outputs)))  )
Runner.Parallel.set_running = set_running

def name2class(name):
	return name.replace(' ', '_').replace('.', '_')

def make_picture(producer):
	# first, cast the parameters
	if not hasattr(producer.bld, 'path'):
		return

	tmp = []
	try:
		while True:
			tup = producer.taskinfo.get(False)
			tmp.append(list(tup))
	except:
		pass

	try:
		ini = float(tmp[0][2])
	except:
		return

	if not info:
		seen = []
		for x in tmp:
			name = x[3]
			if not name in seen:
				seen.append(name)
			else:
				continue

			info.append((name, map_to_color(name)))
		info.sort(key=lambda x: x[0])

	thread_count = 0
	acc = []
	for x in tmp:
		thread_count += x[6]
		acc.append("%d %d %f %r %d %d %d %s" % (x[0], x[1], x[2] - ini, x[3], x[4], x[5], thread_count, x[7]))

	data_node = producer.bld.path.make_node('pdebug.dat')
	data_node.write('\n'.join(acc))

	tmp = [lst[:2] + [float(lst[2]) - ini] + lst[3:] for lst in tmp]

	st = {}
	for l in tmp:
		if not l[0] in st:
			st[l[0]] = len(st.keys())
	tmp = [  [st[lst[0]]] + lst[1:] for lst in tmp ]
	THREAD_AMOUNT = len(st.keys())

	st = {}
	for l in tmp:
		if not l[1] in st:
			st[l[1]] = len(st.keys())
	tmp = [  [lst[0]] + [st[lst[1]]] + lst[2:] for lst in tmp ]


	BAND = Options.options.dband

	seen = {}
	acc = []
	for x in range(len(tmp)):
		line = tmp[x]
		id = line[1]

		if id in seen:
			continue
		seen[id] = True

		begin = line[2]
		thread_id = line[0]
		for y in range(x + 1, len(tmp)):
			line = tmp[y]
			if line[1] == id:
				end = line[2]
				#print id, thread_id, begin, end
				#acc.append(  ( 10*thread_id, 10*(thread_id+1), 10*begin, 10*end ) )
				acc.append( (BAND * begin, BAND*thread_id, BAND*end - BAND*begin, BAND, line[3], line[7]) )
				break

	if Options.options.dmaxtime < 0.1:
		gwidth = 1
		for x in tmp:
			m = BAND * x[2]
			if m > gwidth:
				gwidth = m
	else:
		gwidth = BAND * Options.options.dmaxtime

	ratio = float(Options.options.dwidth) / gwidth
	gwidth = Options.options.dwidth
	gheight = BAND * (THREAD_AMOUNT + len(info) + 1.5)


	# simple data model for our template
	class tobject(object):
		pass

	model = tobject()
	model.x = 0
	model.y = 0
	model.width = gwidth + 4
	model.height = gheight + 4

	model.tooltip = not Options.options.dnotooltip

	model.title = Options.options.dtitle
	model.title_x = gwidth / 2
	model.title_y = gheight + - 5

	groups = {}
	for (x, y, w, h, clsname, name) in acc:
		try:
			groups[clsname].append((x, y, w, h, name))
		except:
			groups[clsname] = [(x, y, w, h, name)]

	# groups of rectangles (else js highlighting is slow)
	model.groups = []
	for cls in groups:
		g = tobject()
		model.groups.append(g)
		g.classname = name2class(cls)
		g.rects = []
		for (x, y, w, h, name) in groups[cls]:
			r = tobject()
			g.rects.append(r)
			r.x = 2 + x * ratio
			r.y = 2 + y
			r.width = w * ratio
			r.height = h
			r.name = name
			r.color = map_to_color(cls)

	cnt = THREAD_AMOUNT

	# caption
	model.infos = []
	for (text, color) in info:
		inf = tobject()
		model.infos.append(inf)
		inf.classname = name2class(text)
		inf.x = 2 + BAND
		inf.y = 5 + (cnt + 0.5) * BAND
		inf.width = BAND/2
		inf.height = BAND/2
		inf.color = color

		inf.text = text
		inf.text_x = 2 + 2 * BAND
		inf.text_y = 5 + (cnt + 0.5) * BAND + 10

		cnt += 1

	# write the file...
	template1 = compile_template(SVG_TEMPLATE)
	txt = template1(model)

	node = producer.bld.path.make_node('pdebug.svg')
	node.write(txt)
	Logs.warn('Created the diagram %r', node)

def options(opt):
	opt.add_option('--dtitle', action='store', default='Parallel build representation for %r' % ' '.join(sys.argv),
		help='title for the svg diagram', dest='dtitle')
	opt.add_option('--dwidth', action='store', type='int', help='diagram width', default=800, dest='dwidth')
	opt.add_option('--dtime', action='store', type='float', help='recording interval in seconds', default=0.009, dest='dtime')
	opt.add_option('--dband', action='store', type='int', help='band width', default=22, dest='dband')
	opt.add_option('--dmaxtime', action='store', type='float', help='maximum time, for drawing fair comparisons', default=0, dest='dmaxtime')
	opt.add_option('--dnotooltip', action='store_true', help='disable tooltips', default=False, dest='dnotooltip')

