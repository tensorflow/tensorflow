"""A parser for XML, using the derived class as static DTD."""

# Author: Sjoerd Mullender.

import re
import string

import warnings
warnings.warn("The xmllib module is obsolete.  Use xml.sax instead.", DeprecationWarning)
del warnings

version = '0.3'

class Error(RuntimeError):
    pass

# Regular expressions used for parsing

_S = '[ \t\r\n]+'                       # white space
_opS = '[ \t\r\n]*'                     # optional white space
_Name = '[a-zA-Z_:][-a-zA-Z0-9._:]*'    # valid XML name
_QStr = "(?:'[^']*'|\"[^\"]*\")"        # quoted XML string
illegal = re.compile('[^\t\r\n -\176\240-\377]') # illegal chars in content
interesting = re.compile('[]&<]')

amp = re.compile('&')
ref = re.compile('&(' + _Name + '|#[0-9]+|#x[0-9a-fA-F]+)[^-a-zA-Z0-9._:]')
entityref = re.compile('&(?P<name>' + _Name + ')[^-a-zA-Z0-9._:]')
charref = re.compile('&#(?P<char>[0-9]+[^0-9]|x[0-9a-fA-F]+[^0-9a-fA-F])')
space = re.compile(_S + '$')
newline = re.compile('\n')

attrfind = re.compile(
    _S + '(?P<name>' + _Name + ')'
    '(' + _opS + '=' + _opS +
    '(?P<value>'+_QStr+'|[-a-zA-Z0-9.:+*%?!\(\)_#=~]+))?')
starttagopen = re.compile('<' + _Name)
starttagend = re.compile(_opS + '(?P<slash>/?)>')
starttagmatch = re.compile('<(?P<tagname>'+_Name+')'
                      '(?P<attrs>(?:'+attrfind.pattern+')*)'+
                      starttagend.pattern)
endtagopen = re.compile('</')
endbracket = re.compile(_opS + '>')
endbracketfind = re.compile('(?:[^>\'"]|'+_QStr+')*>')
tagfind = re.compile(_Name)
cdataopen = re.compile(r'<!\[CDATA\[')
cdataclose = re.compile(r'\]\]>')
# this matches one of the following:
# SYSTEM SystemLiteral
# PUBLIC PubidLiteral SystemLiteral
_SystemLiteral = '(?P<%s>'+_QStr+')'
_PublicLiteral = '(?P<%s>"[-\'\(\)+,./:=?;!*#@$_%% \n\ra-zA-Z0-9]*"|' \
                        "'[-\(\)+,./:=?;!*#@$_%% \n\ra-zA-Z0-9]*')"
_ExternalId = '(?:SYSTEM|' \
                 'PUBLIC'+_S+_PublicLiteral%'pubid'+ \
              ')'+_S+_SystemLiteral%'syslit'
doctype = re.compile('<!DOCTYPE'+_S+'(?P<name>'+_Name+')'
                     '(?:'+_S+_ExternalId+')?'+_opS)
xmldecl = re.compile('<\?xml'+_S+
                     'version'+_opS+'='+_opS+'(?P<version>'+_QStr+')'+
                     '(?:'+_S+'encoding'+_opS+'='+_opS+
                        "(?P<encoding>'[A-Za-z][-A-Za-z0-9._]*'|"
                        '"[A-Za-z][-A-Za-z0-9._]*"))?'
                     '(?:'+_S+'standalone'+_opS+'='+_opS+
                        '(?P<standalone>\'(?:yes|no)\'|"(?:yes|no)"))?'+
                     _opS+'\?>')
procopen = re.compile(r'<\?(?P<proc>' + _Name + ')' + _opS)
procclose = re.compile(_opS + r'\?>')
commentopen = re.compile('<!--')
commentclose = re.compile('-->')
doubledash = re.compile('--')
attrtrans = string.maketrans(' \r\n\t', '    ')

# definitions for XML namespaces
_NCName = '[a-zA-Z_][-a-zA-Z0-9._]*'    # XML Name, minus the ":"
ncname = re.compile(_NCName + '$')
qname = re.compile('(?:(?P<prefix>' + _NCName + '):)?' # optional prefix
                   '(?P<local>' + _NCName + ')$')

xmlns = re.compile('xmlns(?::(?P<ncname>'+_NCName+'))?$')

# XML parser base class -- find tags and call handler functions.
# Usage: p = XMLParser(); p.feed(data); ...; p.close().
# The dtd is defined by deriving a class which defines methods with
# special names to handle tags: start_foo and end_foo to handle <foo>
# and </foo>, respectively.  The data between tags is passed to the
# parser by calling self.handle_data() with some data as argument (the
# data may be split up in arbitrary chunks).

class XMLParser:
    attributes = {}                     # default, to be overridden
    elements = {}                       # default, to be overridden

    # parsing options, settable using keyword args in __init__
    __accept_unquoted_attributes = 0
    __accept_missing_endtag_name = 0
    __map_case = 0
    __accept_utf8 = 0
    __translate_attribute_references = 1

    # Interface -- initialize and reset this instance
    def __init__(self, **kw):
        self.__fixed = 0
        if 'accept_unquoted_attributes' in kw:
            self.__accept_unquoted_attributes = kw['accept_unquoted_attributes']
        if 'accept_missing_endtag_name' in kw:
            self.__accept_missing_endtag_name = kw['accept_missing_endtag_name']
        if 'map_case' in kw:
            self.__map_case = kw['map_case']
        if 'accept_utf8' in kw:
            self.__accept_utf8 = kw['accept_utf8']
        if 'translate_attribute_references' in kw:
            self.__translate_attribute_references = kw['translate_attribute_references']
        self.reset()

    def __fixelements(self):
        self.__fixed = 1
        self.elements = {}
        self.__fixdict(self.__dict__)
        self.__fixclass(self.__class__)

    def __fixclass(self, kl):
        self.__fixdict(kl.__dict__)
        for k in kl.__bases__:
            self.__fixclass(k)

    def __fixdict(self, dict):
        for key in dict.keys():
            if key[:6] == 'start_':
                tag = key[6:]
                start, end = self.elements.get(tag, (None, None))
                if start is None:
                    self.elements[tag] = getattr(self, key), end
            elif key[:4] == 'end_':
                tag = key[4:]
                start, end = self.elements.get(tag, (None, None))
                if end is None:
                    self.elements[tag] = start, getattr(self, key)

    # Interface -- reset this instance.  Loses all unprocessed data
    def reset(self):
        self.rawdata = ''
        self.stack = []
        self.nomoretags = 0
        self.literal = 0
        self.lineno = 1
        self.__at_start = 1
        self.__seen_doctype = None
        self.__seen_starttag = 0
        self.__use_namespaces = 0
        self.__namespaces = {'xml':None}   # xml is implicitly declared
        # backward compatibility hack: if elements not overridden,
        # fill it in ourselves
        if self.elements is XMLParser.elements:
            self.__fixelements()

    # For derived classes only -- enter literal mode (CDATA) till EOF
    def setnomoretags(self):
        self.nomoretags = self.literal = 1

    # For derived classes only -- enter literal mode (CDATA)
    def setliteral(self, *args):
        self.literal = 1

    # Interface -- feed some data to the parser.  Call this as
    # often as you want, with as little or as much text as you
    # want (may include '\n').  (This just saves the text, all the
    # processing is done by goahead().)
    def feed(self, data):
        self.rawdata = self.rawdata + data
        self.goahead(0)

    # Interface -- handle the remaining data
    def close(self):
        self.goahead(1)
        if self.__fixed:
            self.__fixed = 0
            # remove self.elements so that we don't leak
            del self.elements

    # Interface -- translate references
    def translate_references(self, data, all = 1):
        if not self.__translate_attribute_references:
            return data
        i = 0
        while 1:
            res = amp.search(data, i)
            if res is None:
                return data
            s = res.start(0)
            res = ref.match(data, s)
            if res is None:
                self.syntax_error("bogus `&'")
                i = s+1
                continue
            i = res.end(0)
            str = res.group(1)
            rescan = 0
            if str[0] == '#':
                if str[1] == 'x':
                    str = chr(int(str[2:], 16))
                else:
                    str = chr(int(str[1:]))
                if data[i - 1] != ';':
                    self.syntax_error("`;' missing after char reference")
                    i = i-1
            elif all:
                if str in self.entitydefs:
                    str = self.entitydefs[str]
                    rescan = 1
                elif data[i - 1] != ';':
                    self.syntax_error("bogus `&'")
                    i = s + 1 # just past the &
                    continue
                else:
                    self.syntax_error("reference to unknown entity `&%s;'" % str)
                    str = '&' + str + ';'
            elif data[i - 1] != ';':
                self.syntax_error("bogus `&'")
                i = s + 1 # just past the &
                continue

            # when we get here, str contains the translated text and i points
            # to the end of the string that is to be replaced
            data = data[:s] + str + data[i:]
            if rescan:
                i = s
            else:
                i = s + len(str)

    # Interface - return a dictionary of all namespaces currently valid
    def getnamespace(self):
        nsdict = {}
        for t, d, nst in self.stack:
            nsdict.update(d)
        return nsdict

    # Internal -- handle data as far as reasonable.  May leave state
    # and data to be processed by a subsequent call.  If 'end' is
    # true, force handling all data as if followed by EOF marker.
    def goahead(self, end):
        rawdata = self.rawdata
        i = 0
        n = len(rawdata)
        while i < n:
            if i > 0:
                self.__at_start = 0
            if self.nomoretags:
                data = rawdata[i:n]
                self.handle_data(data)
                self.lineno = self.lineno + data.count('\n')
                i = n
                break
            res = interesting.search(rawdata, i)
            if res:
                j = res.start(0)
            else:
                j = n
            if i < j:
                data = rawdata[i:j]
                if self.__at_start and space.match(data) is None:
                    self.syntax_error('illegal data at start of file')
                self.__at_start = 0
                if not self.stack and space.match(data) is None:
                    self.syntax_error('data not in content')
                if not self.__accept_utf8 and illegal.search(data):
                    self.syntax_error('illegal character in content')
                self.handle_data(data)
                self.lineno = self.lineno + data.count('\n')
            i = j
            if i == n: break
            if rawdata[i] == '<':
                if starttagopen.match(rawdata, i):
                    if self.literal:
                        data = rawdata[i]
                        self.handle_data(data)
                        self.lineno = self.lineno + data.count('\n')
                        i = i+1
                        continue
                    k = self.parse_starttag(i)
                    if k < 0: break
                    self.__seen_starttag = 1
                    self.lineno = self.lineno + rawdata[i:k].count('\n')
                    i = k
                    continue
                if endtagopen.match(rawdata, i):
                    k = self.parse_endtag(i)
                    if k < 0: break
                    self.lineno = self.lineno + rawdata[i:k].count('\n')
                    i =  k
                    continue
                if commentopen.match(rawdata, i):
                    if self.literal:
                        data = rawdata[i]
                        self.handle_data(data)
                        self.lineno = self.lineno + data.count('\n')
                        i = i+1
                        continue
                    k = self.parse_comment(i)
                    if k < 0: break
                    self.lineno = self.lineno + rawdata[i:k].count('\n')
                    i = k
                    continue
                if cdataopen.match(rawdata, i):
                    k = self.parse_cdata(i)
                    if k < 0: break
                    self.lineno = self.lineno + rawdata[i:k].count('\n')
                    i = k
                    continue
                res = xmldecl.match(rawdata, i)
                if res:
                    if not self.__at_start:
                        self.syntax_error("<?xml?> declaration not at start of document")
                    version, encoding, standalone = res.group('version',
                                                              'encoding',
                                                              'standalone')
                    if version[1:-1] != '1.0':
                        raise Error('only XML version 1.0 supported')
                    if encoding: encoding = encoding[1:-1]
                    if standalone: standalone = standalone[1:-1]
                    self.handle_xml(encoding, standalone)
                    i = res.end(0)
                    continue
                res = procopen.match(rawdata, i)
                if res:
                    k = self.parse_proc(i)
                    if k < 0: break
                    self.lineno = self.lineno + rawdata[i:k].count('\n')
                    i = k
                    continue
                res = doctype.match(rawdata, i)
                if res:
                    if self.literal:
                        data = rawdata[i]
                        self.handle_data(data)
                        self.lineno = self.lineno + data.count('\n')
                        i = i+1
                        continue
                    if self.__seen_doctype:
                        self.syntax_error('multiple DOCTYPE elements')
                    if self.__seen_starttag:
                        self.syntax_error('DOCTYPE not at beginning of document')
                    k = self.parse_doctype(res)
                    if k < 0: break
                    self.__seen_doctype = res.group('name')
                    if self.__map_case:
                        self.__seen_doctype = self.__seen_doctype.lower()
                    self.lineno = self.lineno + rawdata[i:k].count('\n')
                    i = k
                    continue
            elif rawdata[i] == '&':
                if self.literal:
                    data = rawdata[i]
                    self.handle_data(data)
                    i = i+1
                    continue
                res = charref.match(rawdata, i)
                if res is not None:
                    i = res.end(0)
                    if rawdata[i-1] != ';':
                        self.syntax_error("`;' missing in charref")
                        i = i-1
                    if not self.stack:
                        self.syntax_error('data not in content')
                    self.handle_charref(res.group('char')[:-1])
                    self.lineno = self.lineno + res.group(0).count('\n')
                    continue
                res = entityref.match(rawdata, i)
                if res is not None:
                    i = res.end(0)
                    if rawdata[i-1] != ';':
                        self.syntax_error("`;' missing in entityref")
                        i = i-1
                    name = res.group('name')
                    if self.__map_case:
                        name = name.lower()
                    if name in self.entitydefs:
                        self.rawdata = rawdata = rawdata[:res.start(0)] + self.entitydefs[name] + rawdata[i:]
                        n = len(rawdata)
                        i = res.start(0)
                    else:
                        self.unknown_entityref(name)
                    self.lineno = self.lineno + res.group(0).count('\n')
                    continue
            elif rawdata[i] == ']':
                if self.literal:
                    data = rawdata[i]
                    self.handle_data(data)
                    i = i+1
                    continue
                if n-i < 3:
                    break
                if cdataclose.match(rawdata, i):
                    self.syntax_error("bogus `]]>'")
                self.handle_data(rawdata[i])
                i = i+1
                continue
            else:
                raise Error('neither < nor & ??')
            # We get here only if incomplete matches but
            # nothing else
            break
        # end while
        if i > 0:
            self.__at_start = 0
        if end and i < n:
            data = rawdata[i]
            self.syntax_error("bogus `%s'" % data)
            if not self.__accept_utf8 and illegal.search(data):
                self.syntax_error('illegal character in content')
            self.handle_data(data)
            self.lineno = self.lineno + data.count('\n')
            self.rawdata = rawdata[i+1:]
            return self.goahead(end)
        self.rawdata = rawdata[i:]
        if end:
            if not self.__seen_starttag:
                self.syntax_error('no elements in file')
            if self.stack:
                self.syntax_error('missing end tags')
                while self.stack:
                    self.finish_endtag(self.stack[-1][0])

    # Internal -- parse comment, return length or -1 if not terminated
    def parse_comment(self, i):
        rawdata = self.rawdata
        if rawdata[i:i+4] != '<!--':
            raise Error('unexpected call to handle_comment')
        res = commentclose.search(rawdata, i+4)
        if res is None:
            return -1
        if doubledash.search(rawdata, i+4, res.start(0)):
            self.syntax_error("`--' inside comment")
        if rawdata[res.start(0)-1] == '-':
            self.syntax_error('comment cannot end in three dashes')
        if not self.__accept_utf8 and \
           illegal.search(rawdata, i+4, res.start(0)):
            self.syntax_error('illegal character in comment')
        self.handle_comment(rawdata[i+4: res.start(0)])
        return res.end(0)

    # Internal -- handle DOCTYPE tag, return length or -1 if not terminated
    def parse_doctype(self, res):
        rawdata = self.rawdata
        n = len(rawdata)
        name = res.group('name')
        if self.__map_case:
            name = name.lower()
        pubid, syslit = res.group('pubid', 'syslit')
        if pubid is not None:
            pubid = pubid[1:-1]         # remove quotes
            pubid = ' '.join(pubid.split()) # normalize
        if syslit is not None: syslit = syslit[1:-1] # remove quotes
        j = k = res.end(0)
        if k >= n:
            return -1
        if rawdata[k] == '[':
            level = 0
            k = k+1
            dq = sq = 0
            while k < n:
                c = rawdata[k]
                if not sq and c == '"':
                    dq = not dq
                elif not dq and c == "'":
                    sq = not sq
                elif sq or dq:
                    pass
                elif level <= 0 and c == ']':
                    res = endbracket.match(rawdata, k+1)
                    if res is None:
                        return -1
                    self.handle_doctype(name, pubid, syslit, rawdata[j+1:k])
                    return res.end(0)
                elif c == '<':
                    level = level + 1
                elif c == '>':
                    level = level - 1
                    if level < 0:
                        self.syntax_error("bogus `>' in DOCTYPE")
                k = k+1
        res = endbracketfind.match(rawdata, k)
        if res is None:
            return -1
        if endbracket.match(rawdata, k) is None:
            self.syntax_error('garbage in DOCTYPE')
        self.handle_doctype(name, pubid, syslit, None)
        return res.end(0)

    # Internal -- handle CDATA tag, return length or -1 if not terminated
    def parse_cdata(self, i):
        rawdata = self.rawdata
        if rawdata[i:i+9] != '<![CDATA[':
            raise Error('unexpected call to parse_cdata')
        res = cdataclose.search(rawdata, i+9)
        if res is None:
            return -1
        if not self.__accept_utf8 and \
           illegal.search(rawdata, i+9, res.start(0)):
            self.syntax_error('illegal character in CDATA')
        if not self.stack:
            self.syntax_error('CDATA not in content')
        self.handle_cdata(rawdata[i+9:res.start(0)])
        return res.end(0)

    __xml_namespace_attributes = {'ns':None, 'src':None, 'prefix':None}
    # Internal -- handle a processing instruction tag
    def parse_proc(self, i):
        rawdata = self.rawdata
        end = procclose.search(rawdata, i)
        if end is None:
            return -1
        j = end.start(0)
        if not self.__accept_utf8 and illegal.search(rawdata, i+2, j):
            self.syntax_error('illegal character in processing instruction')
        res = tagfind.match(rawdata, i+2)
        if res is None:
            raise Error('unexpected call to parse_proc')
        k = res.end(0)
        name = res.group(0)
        if self.__map_case:
            name = name.lower()
        if name == 'xml:namespace':
            self.syntax_error('old-fashioned namespace declaration')
            self.__use_namespaces = -1
            # namespace declaration
            # this must come after the <?xml?> declaration (if any)
            # and before the <!DOCTYPE> (if any).
            if self.__seen_doctype or self.__seen_starttag:
                self.syntax_error('xml:namespace declaration too late in document')
            attrdict, namespace, k = self.parse_attributes(name, k, j)
            if namespace:
                self.syntax_error('namespace declaration inside namespace declaration')
            for attrname in attrdict.keys():
                if not attrname in self.__xml_namespace_attributes:
                    self.syntax_error("unknown attribute `%s' in xml:namespace tag" % attrname)
            if not 'ns' in attrdict or not 'prefix' in attrdict:
                self.syntax_error('xml:namespace without required attributes')
            prefix = attrdict.get('prefix')
            if ncname.match(prefix) is None:
                self.syntax_error('xml:namespace illegal prefix value')
                return end.end(0)
            if prefix in self.__namespaces:
                self.syntax_error('xml:namespace prefix not unique')
            self.__namespaces[prefix] = attrdict['ns']
        else:
            if name.lower() == 'xml':
                self.syntax_error('illegal processing instruction target name')
            self.handle_proc(name, rawdata[k:j])
        return end.end(0)

    # Internal -- parse attributes between i and j
    def parse_attributes(self, tag, i, j):
        rawdata = self.rawdata
        attrdict = {}
        namespace = {}
        while i < j:
            res = attrfind.match(rawdata, i)
            if res is None:
                break
            attrname, attrvalue = res.group('name', 'value')
            if self.__map_case:
                attrname = attrname.lower()
            i = res.end(0)
            if attrvalue is None:
                self.syntax_error("no value specified for attribute `%s'" % attrname)
                attrvalue = attrname
            elif attrvalue[:1] == "'" == attrvalue[-1:] or \
                 attrvalue[:1] == '"' == attrvalue[-1:]:
                attrvalue = attrvalue[1:-1]
            elif not self.__accept_unquoted_attributes:
                self.syntax_error("attribute `%s' value not quoted" % attrname)
            res = xmlns.match(attrname)
            if res is not None:
                # namespace declaration
                ncname = res.group('ncname')
                namespace[ncname or ''] = attrvalue or None
                if not self.__use_namespaces:
                    self.__use_namespaces = len(self.stack)+1
                continue
            if '<' in attrvalue:
                self.syntax_error("`<' illegal in attribute value")
            if attrname in attrdict:
                self.syntax_error("attribute `%s' specified twice" % attrname)
            attrvalue = attrvalue.translate(attrtrans)
            attrdict[attrname] = self.translate_references(attrvalue)
        return attrdict, namespace, i

    # Internal -- handle starttag, return length or -1 if not terminated
    def parse_starttag(self, i):
        rawdata = self.rawdata
        # i points to start of tag
        end = endbracketfind.match(rawdata, i+1)
        if end is None:
            return -1
        tag = starttagmatch.match(rawdata, i)
        if tag is None or tag.end(0) != end.end(0):
            self.syntax_error('garbage in starttag')
            return end.end(0)
        nstag = tagname = tag.group('tagname')
        if self.__map_case:
            nstag = tagname = nstag.lower()
        if not self.__seen_starttag and self.__seen_doctype and \
           tagname != self.__seen_doctype:
            self.syntax_error('starttag does not match DOCTYPE')
        if self.__seen_starttag and not self.stack:
            self.syntax_error('multiple elements on top level')
        k, j = tag.span('attrs')
        attrdict, nsdict, k = self.parse_attributes(tagname, k, j)
        self.stack.append((tagname, nsdict, nstag))
        if self.__use_namespaces:
            res = qname.match(tagname)
        else:
            res = None
        if res is not None:
            prefix, nstag = res.group('prefix', 'local')
            if prefix is None:
                prefix = ''
            ns = None
            for t, d, nst in self.stack:
                if prefix in d:
                    ns = d[prefix]
            if ns is None and prefix != '':
                ns = self.__namespaces.get(prefix)
            if ns is not None:
                nstag = ns + ' ' + nstag
            elif prefix != '':
                nstag = prefix + ':' + nstag # undo split
            self.stack[-1] = tagname, nsdict, nstag
        # translate namespace of attributes
        attrnamemap = {} # map from new name to old name (used for error reporting)
        for key in attrdict.keys():
            attrnamemap[key] = key
        if self.__use_namespaces:
            nattrdict = {}
            for key, val in attrdict.items():
                okey = key
                res = qname.match(key)
                if res is not None:
                    aprefix, key = res.group('prefix', 'local')
                    if self.__map_case:
                        key = key.lower()
                    if aprefix is not None:
                        ans = None
                        for t, d, nst in self.stack:
                            if aprefix in d:
                                ans = d[aprefix]
                        if ans is None:
                            ans = self.__namespaces.get(aprefix)
                        if ans is not None:
                            key = ans + ' ' + key
                        else:
                            key = aprefix + ':' + key
                nattrdict[key] = val
                attrnamemap[key] = okey
            attrdict = nattrdict
        attributes = self.attributes.get(nstag)
        if attributes is not None:
            for key in attrdict.keys():
                if not key in attributes:
                    self.syntax_error("unknown attribute `%s' in tag `%s'" % (attrnamemap[key], tagname))
            for key, val in attributes.items():
                if val is not None and not key in attrdict:
                    attrdict[key] = val
        method = self.elements.get(nstag, (None, None))[0]
        self.finish_starttag(nstag, attrdict, method)
        if tag.group('slash') == '/':
            self.finish_endtag(tagname)
        return tag.end(0)

    # Internal -- parse endtag
    def parse_endtag(self, i):
        rawdata = self.rawdata
        end = endbracketfind.match(rawdata, i+1)
        if end is None:
            return -1
        res = tagfind.match(rawdata, i+2)
        if res is None:
            if self.literal:
                self.handle_data(rawdata[i])
                return i+1
            if not self.__accept_missing_endtag_name:
                self.syntax_error('no name specified in end tag')
            tag = self.stack[-1][0]
            k = i+2
        else:
            tag = res.group(0)
            if self.__map_case:
                tag = tag.lower()
            if self.literal:
                if not self.stack or tag != self.stack[-1][0]:
                    self.handle_data(rawdata[i])
                    return i+1
            k = res.end(0)
        if endbracket.match(rawdata, k) is None:
            self.syntax_error('garbage in end tag')
        self.finish_endtag(tag)
        return end.end(0)

    # Internal -- finish processing of start tag
    def finish_starttag(self, tagname, attrdict, method):
        if method is not None:
            self.handle_starttag(tagname, method, attrdict)
        else:
            self.unknown_starttag(tagname, attrdict)

    # Internal -- finish processing of end tag
    def finish_endtag(self, tag):
        self.literal = 0
        if not tag:
            self.syntax_error('name-less end tag')
            found = len(self.stack) - 1
            if found < 0:
                self.unknown_endtag(tag)
                return
        else:
            found = -1
            for i in range(len(self.stack)):
                if tag == self.stack[i][0]:
                    found = i
            if found == -1:
                self.syntax_error('unopened end tag')
                return
        while len(self.stack) > found:
            if found < len(self.stack) - 1:
                self.syntax_error('missing close tag for %s' % self.stack[-1][2])
            nstag = self.stack[-1][2]
            method = self.elements.get(nstag, (None, None))[1]
            if method is not None:
                self.handle_endtag(nstag, method)
            else:
                self.unknown_endtag(nstag)
            if self.__use_namespaces == len(self.stack):
                self.__use_namespaces = 0
            del self.stack[-1]

    # Overridable -- handle xml processing instruction
    def handle_xml(self, encoding, standalone):
        pass

    # Overridable -- handle DOCTYPE
    def handle_doctype(self, tag, pubid, syslit, data):
        pass

    # Overridable -- handle start tag
    def handle_starttag(self, tag, method, attrs):
        method(attrs)

    # Overridable -- handle end tag
    def handle_endtag(self, tag, method):
        method()

    # Example -- handle character reference, no need to override
    def handle_charref(self, name):
        try:
            if name[0] == 'x':
                n = int(name[1:], 16)
            else:
                n = int(name)
        except ValueError:
            self.unknown_charref(name)
            return
        if not 0 <= n <= 255:
            self.unknown_charref(name)
            return
        self.handle_data(chr(n))

    # Definition of entities -- derived classes may override
    entitydefs = {'lt': '&#60;',        # must use charref
                  'gt': '&#62;',
                  'amp': '&#38;',       # must use charref
                  'quot': '&#34;',
                  'apos': '&#39;',
                  }

    # Example -- handle data, should be overridden
    def handle_data(self, data):
        pass

    # Example -- handle cdata, could be overridden
    def handle_cdata(self, data):
        pass

    # Example -- handle comment, could be overridden
    def handle_comment(self, data):
        pass

    # Example -- handle processing instructions, could be overridden
    def handle_proc(self, name, data):
        pass

    # Example -- handle relatively harmless syntax errors, could be overridden
    def syntax_error(self, message):
        raise Error('Syntax error at line %d: %s' % (self.lineno, message))

    # To be overridden -- handlers for unknown objects
    def unknown_starttag(self, tag, attrs): pass
    def unknown_endtag(self, tag): pass
    def unknown_charref(self, ref): pass
    def unknown_entityref(self, name):
        self.syntax_error("reference to unknown entity `&%s;'" % name)


class TestXMLParser(XMLParser):

    def __init__(self, **kw):
        self.testdata = ""
        XMLParser.__init__(self, **kw)

    def handle_xml(self, encoding, standalone):
        self.flush()
        print 'xml: encoding =',encoding,'standalone =',standalone

    def handle_doctype(self, tag, pubid, syslit, data):
        self.flush()
        print 'DOCTYPE:',tag, repr(data)

    def handle_data(self, data):
        self.testdata = self.testdata + data
        if len(repr(self.testdata)) >= 70:
            self.flush()

    def flush(self):
        data = self.testdata
        if data:
            self.testdata = ""
            print 'data:', repr(data)

    def handle_cdata(self, data):
        self.flush()
        print 'cdata:', repr(data)

    def handle_proc(self, name, data):
        self.flush()
        print 'processing:',name,repr(data)

    def handle_comment(self, data):
        self.flush()
        r = repr(data)
        if len(r) > 68:
            r = r[:32] + '...' + r[-32:]
        print 'comment:', r

    def syntax_error(self, message):
        print 'error at line %d:' % self.lineno, message

    def unknown_starttag(self, tag, attrs):
        self.flush()
        if not attrs:
            print 'start tag: <' + tag + '>'
        else:
            print 'start tag: <' + tag,
            for name, value in attrs.items():
                print name + '=' + '"' + value + '"',
            print '>'

    def unknown_endtag(self, tag):
        self.flush()
        print 'end tag: </' + tag + '>'

    def unknown_entityref(self, ref):
        self.flush()
        print '*** unknown entity ref: &' + ref + ';'

    def unknown_charref(self, ref):
        self.flush()
        print '*** unknown char ref: &#' + ref + ';'

    def close(self):
        XMLParser.close(self)
        self.flush()

def test(args = None):
    import sys, getopt
    from time import time

    if not args:
        args = sys.argv[1:]

    opts, args = getopt.getopt(args, 'st')
    klass = TestXMLParser
    do_time = 0
    for o, a in opts:
        if o == '-s':
            klass = XMLParser
        elif o == '-t':
            do_time = 1

    if args:
        file = args[0]
    else:
        file = 'test.xml'

    if file == '-':
        f = sys.stdin
    else:
        try:
            f = open(file, 'r')
        except IOError, msg:
            print file, ":", msg
            sys.exit(1)

    data = f.read()
    if f is not sys.stdin:
        f.close()

    x = klass()
    t0 = time()
    try:
        if do_time:
            x.feed(data)
            x.close()
        else:
            for c in data:
                x.feed(c)
            x.close()
    except Error, msg:
        t1 = time()
        print msg
        if do_time:
            print 'total time: %g' % (t1-t0)
        sys.exit(1)
    t1 = time()
    if do_time:
        print 'total time: %g' % (t1-t0)


if __name__ == '__main__':
    test()
