""" Locale support.

    The module provides low-level access to the C lib's locale APIs
    and adds high level number formatting APIs as well as a locale
    aliasing engine to complement these.

    The aliasing engine includes support for many commonly used locale
    names and maps them to values suitable for passing to the C lib's
    setlocale() function. It also includes default encodings for all
    supported locale names.

"""

import sys, encodings, encodings.aliases

# Try importing the _locale module.
#
# If this fails, fall back on a basic 'C' locale emulation.

# Yuck:  LC_MESSAGES is non-standard:  can't tell whether it exists before
# trying the import.  So __all__ is also fiddled at the end of the file.
__all__ = ["getlocale", "getdefaultlocale", "getpreferredencoding", "Error",
           "setlocale", "resetlocale", "localeconv", "strcoll", "strxfrm",
           "str", "atof", "atoi", "format", "format_string", "currency",
           "normalize", "LC_CTYPE", "LC_COLLATE", "LC_TIME", "LC_MONETARY",
           "LC_NUMERIC", "LC_ALL", "CHAR_MAX"]

try:

    from _locale import *

except ImportError:

    # Locale emulation

    CHAR_MAX = 127
    LC_ALL = 6
    LC_COLLATE = 3
    LC_CTYPE = 0
    LC_MESSAGES = 5
    LC_MONETARY = 4
    LC_NUMERIC = 1
    LC_TIME = 2
    Error = ValueError

    def localeconv():
        """ localeconv() -> dict.
            Returns numeric and monetary locale-specific parameters.
        """
        # 'C' locale default values
        return {'grouping': [127],
                'currency_symbol': '',
                'n_sign_posn': 127,
                'p_cs_precedes': 127,
                'n_cs_precedes': 127,
                'mon_grouping': [],
                'n_sep_by_space': 127,
                'decimal_point': '.',
                'negative_sign': '',
                'positive_sign': '',
                'p_sep_by_space': 127,
                'int_curr_symbol': '',
                'p_sign_posn': 127,
                'thousands_sep': '',
                'mon_thousands_sep': '',
                'frac_digits': 127,
                'mon_decimal_point': '',
                'int_frac_digits': 127}

    def setlocale(category, value=None):
        """ setlocale(integer,string=None) -> string.
            Activates/queries locale processing.
        """
        if value not in (None, '', 'C'):
            raise Error, '_locale emulation only supports "C" locale'
        return 'C'

    def strcoll(a,b):
        """ strcoll(string,string) -> int.
            Compares two strings according to the locale.
        """
        return cmp(a,b)

    def strxfrm(s):
        """ strxfrm(string) -> string.
            Returns a string that behaves for cmp locale-aware.
        """
        return s

### Number formatting APIs

# Author: Martin von Loewis
# improved by Georg Brandl

#perform the grouping from right to left
def _group(s, monetary=False):
    conv = localeconv()
    thousands_sep = conv[monetary and 'mon_thousands_sep' or 'thousands_sep']
    grouping = conv[monetary and 'mon_grouping' or 'grouping']
    if not grouping:
        return (s, 0)
    result = ""
    seps = 0
    spaces = ""
    if s[-1] == ' ':
        sp = s.find(' ')
        spaces = s[sp:]
        s = s[:sp]
    while s and grouping:
        # if grouping is -1, we are done
        if grouping[0] == CHAR_MAX:
            break
        # 0: re-use last group ad infinitum
        elif grouping[0] != 0:
            #process last group
            group = grouping[0]
            grouping = grouping[1:]
        if result:
            result = s[-group:] + thousands_sep + result
            seps += 1
        else:
            result = s[-group:]
        s = s[:-group]
        if s and s[-1] not in "0123456789":
            # the leading string is only spaces and signs
            return s + result + spaces, seps
    if not result:
        return s + spaces, seps
    if s:
        result = s + thousands_sep + result
        seps += 1
    return result + spaces, seps

def format(percent, value, grouping=False, monetary=False, *additional):
    """Returns the locale-aware substitution of a %? specifier
    (percent).

    additional is for format strings which contain one or more
    '*' modifiers."""
    # this is only for one-percent-specifier strings and this should be checked
    if percent[0] != '%':
        raise ValueError("format() must be given exactly one %char "
                         "format specifier")
    if additional:
        formatted = percent % ((value,) + additional)
    else:
        formatted = percent % value
    # floats and decimal ints need special action!
    if percent[-1] in 'eEfFgG':
        seps = 0
        parts = formatted.split('.')
        if grouping:
            parts[0], seps = _group(parts[0], monetary=monetary)
        decimal_point = localeconv()[monetary and 'mon_decimal_point'
                                              or 'decimal_point']
        formatted = decimal_point.join(parts)
        while seps:
            sp = formatted.find(' ')
            if sp == -1: break
            formatted = formatted[:sp] + formatted[sp+1:]
            seps -= 1
    elif percent[-1] in 'diu':
        if grouping:
            formatted = _group(formatted, monetary=monetary)[0]
    return formatted

import re, operator
_percent_re = re.compile(r'%(?:\((?P<key>.*?)\))?'
                         r'(?P<modifiers>[-#0-9 +*.hlL]*?)[eEfFgGdiouxXcrs%]')

def format_string(f, val, grouping=False):
    """Formats a string in the same way that the % formatting would use,
    but takes the current locale into account.
    Grouping is applied if the third parameter is true."""
    percents = list(_percent_re.finditer(f))
    new_f = _percent_re.sub('%s', f)

    if isinstance(val, tuple):
        new_val = list(val)
        i = 0
        for perc in percents:
            starcount = perc.group('modifiers').count('*')
            new_val[i] = format(perc.group(), new_val[i], grouping, False, *new_val[i+1:i+1+starcount])
            del new_val[i+1:i+1+starcount]
            i += (1 + starcount)
        val = tuple(new_val)
    elif operator.isMappingType(val):
        for perc in percents:
            key = perc.group("key")
            val[key] = format(perc.group(), val[key], grouping)
    else:
        # val is a single value
        val = format(percents[0].group(), val, grouping)

    return new_f % val

def currency(val, symbol=True, grouping=False, international=False):
    """Formats val according to the currency settings
    in the current locale."""
    conv = localeconv()

    # check for illegal values
    digits = conv[international and 'int_frac_digits' or 'frac_digits']
    if digits == 127:
        raise ValueError("Currency formatting is not possible using "
                         "the 'C' locale.")

    s = format('%%.%if' % digits, abs(val), grouping, monetary=True)
    # '<' and '>' are markers if the sign must be inserted between symbol and value
    s = '<' + s + '>'

    if symbol:
        smb = conv[international and 'int_curr_symbol' or 'currency_symbol']
        precedes = conv[val<0 and 'n_cs_precedes' or 'p_cs_precedes']
        separated = conv[val<0 and 'n_sep_by_space' or 'p_sep_by_space']

        if precedes:
            s = smb + (separated and ' ' or '') + s
        else:
            s = s + (separated and ' ' or '') + smb

    sign_pos = conv[val<0 and 'n_sign_posn' or 'p_sign_posn']
    sign = conv[val<0 and 'negative_sign' or 'positive_sign']

    if sign_pos == 0:
        s = '(' + s + ')'
    elif sign_pos == 1:
        s = sign + s
    elif sign_pos == 2:
        s = s + sign
    elif sign_pos == 3:
        s = s.replace('<', sign)
    elif sign_pos == 4:
        s = s.replace('>', sign)
    else:
        # the default if nothing specified;
        # this should be the most fitting sign position
        s = sign + s

    return s.replace('<', '').replace('>', '')

def str(val):
    """Convert float to integer, taking the locale into account."""
    return format("%.12g", val)

def atof(string, func=float):
    "Parses a string as a float according to the locale settings."
    #First, get rid of the grouping
    ts = localeconv()['thousands_sep']
    if ts:
        string = string.replace(ts, '')
    #next, replace the decimal point with a dot
    dd = localeconv()['decimal_point']
    if dd:
        string = string.replace(dd, '.')
    #finally, parse the string
    return func(string)

def atoi(str):
    "Converts a string to an integer according to the locale settings."
    return atof(str, int)

def _test():
    setlocale(LC_ALL, "")
    #do grouping
    s1 = format("%d", 123456789,1)
    print s1, "is", atoi(s1)
    #standard formatting
    s1 = str(3.14)
    print s1, "is", atof(s1)

### Locale name aliasing engine

# Author: Marc-Andre Lemburg, mal@lemburg.com
# Various tweaks by Fredrik Lundh <fredrik@pythonware.com>

# store away the low-level version of setlocale (it's
# overridden below)
_setlocale = setlocale

def normalize(localename):

    """ Returns a normalized locale code for the given locale
        name.

        The returned locale code is formatted for use with
        setlocale().

        If normalization fails, the original name is returned
        unchanged.

        If the given encoding is not known, the function defaults to
        the default encoding for the locale code just like setlocale()
        does.

    """
    # Normalize the locale name and extract the encoding
    fullname = localename.lower()
    if ':' in fullname:
        # ':' is sometimes used as encoding delimiter.
        fullname = fullname.replace(':', '.')
    if '.' in fullname:
        langname, encoding = fullname.split('.')[:2]
        fullname = langname + '.' + encoding
    else:
        langname = fullname
        encoding = ''

    # First lookup: fullname (possibly with encoding)
    norm_encoding = encoding.replace('-', '')
    norm_encoding = norm_encoding.replace('_', '')
    lookup_name = langname + '.' + encoding
    code = locale_alias.get(lookup_name, None)
    if code is not None:
        return code
    #print 'first lookup failed'

    # Second try: langname (without encoding)
    code = locale_alias.get(langname, None)
    if code is not None:
        #print 'langname lookup succeeded'
        if '.' in code:
            langname, defenc = code.split('.')
        else:
            langname = code
            defenc = ''
        if encoding:
            # Convert the encoding to a C lib compatible encoding string
            norm_encoding = encodings.normalize_encoding(encoding)
            #print 'norm encoding: %r' % norm_encoding
            norm_encoding = encodings.aliases.aliases.get(norm_encoding,
                                                          norm_encoding)
            #print 'aliased encoding: %r' % norm_encoding
            encoding = locale_encoding_alias.get(norm_encoding,
                                                 norm_encoding)
        else:
            encoding = defenc
        #print 'found encoding %r' % encoding
        if encoding:
            return langname + '.' + encoding
        else:
            return langname

    else:
        return localename

def _parse_localename(localename):

    """ Parses the locale code for localename and returns the
        result as tuple (language code, encoding).

        The localename is normalized and passed through the locale
        alias engine. A ValueError is raised in case the locale name
        cannot be parsed.

        The language code corresponds to RFC 1766.  code and encoding
        can be None in case the values cannot be determined or are
        unknown to this implementation.

    """
    code = normalize(localename)
    if '@' in code:
        # Deal with locale modifiers
        code, modifier = code.split('@')
        if modifier == 'euro' and '.' not in code:
            # Assume Latin-9 for @euro locales. This is bogus,
            # since some systems may use other encodings for these
            # locales. Also, we ignore other modifiers.
            return code, 'iso-8859-15'

    if '.' in code:
        return tuple(code.split('.')[:2])
    elif code == 'C':
        return None, None
    raise ValueError, 'unknown locale: %s' % localename

def _build_localename(localetuple):

    """ Builds a locale code from the given tuple (language code,
        encoding).

        No aliasing or normalizing takes place.

    """
    language, encoding = localetuple
    if language is None:
        language = 'C'
    if encoding is None:
        return language
    else:
        return language + '.' + encoding

def getdefaultlocale(envvars=('LC_ALL', 'LC_CTYPE', 'LANG', 'LANGUAGE')):

    """ Tries to determine the default locale settings and returns
        them as tuple (language code, encoding).

        According to POSIX, a program which has not called
        setlocale(LC_ALL, "") runs using the portable 'C' locale.
        Calling setlocale(LC_ALL, "") lets it use the default locale as
        defined by the LANG variable. Since we don't want to interfere
        with the current locale setting we thus emulate the behavior
        in the way described above.

        To maintain compatibility with other platforms, not only the
        LANG variable is tested, but a list of variables given as
        envvars parameter. The first found to be defined will be
        used. envvars defaults to the search path used in GNU gettext;
        it must always contain the variable name 'LANG'.

        Except for the code 'C', the language code corresponds to RFC
        1766.  code and encoding can be None in case the values cannot
        be determined.

    """

    try:
        # check if it's supported by the _locale module
        import _locale
        code, encoding = _locale._getdefaultlocale()
    except (ImportError, AttributeError):
        pass
    else:
        # make sure the code/encoding values are valid
        if sys.platform == "win32" and code and code[:2] == "0x":
            # map windows language identifier to language name
            code = windows_locale.get(int(code, 0))
        # ...add other platform-specific processing here, if
        # necessary...
        return code, encoding

    # fall back on POSIX behaviour
    import os
    lookup = os.environ.get
    for variable in envvars:
        localename = lookup(variable,None)
        if localename:
            if variable == 'LANGUAGE':
                localename = localename.split(':')[0]
            break
    else:
        localename = 'C'
    return _parse_localename(localename)


def getlocale(category=LC_CTYPE):

    """ Returns the current setting for the given locale category as
        tuple (language code, encoding).

        category may be one of the LC_* value except LC_ALL. It
        defaults to LC_CTYPE.

        Except for the code 'C', the language code corresponds to RFC
        1766.  code and encoding can be None in case the values cannot
        be determined.

    """
    localename = _setlocale(category)
    if category == LC_ALL and ';' in localename:
        raise TypeError, 'category LC_ALL is not supported'
    return _parse_localename(localename)

def setlocale(category, locale=None):

    """ Set the locale for the given category.  The locale can be
        a string, a locale tuple (language code, encoding), or None.

        Locale tuples are converted to strings the locale aliasing
        engine.  Locale strings are passed directly to the C lib.

        category may be given as one of the LC_* values.

    """
    if locale and type(locale) is not type(""):
        # convert to string
        locale = normalize(_build_localename(locale))
    return _setlocale(category, locale)

def resetlocale(category=LC_ALL):

    """ Sets the locale for category to the default setting.

        The default setting is determined by calling
        getdefaultlocale(). category defaults to LC_ALL.

    """
    _setlocale(category, _build_localename(getdefaultlocale()))

if sys.platform in ('win32', 'darwin', 'mac'):
    # On Win32, this will return the ANSI code page
    # On the Mac, it should return the system encoding;
    # it might return "ascii" instead
    def getpreferredencoding(do_setlocale = True):
        """Return the charset that the user is likely using."""
        import _locale
        return _locale._getdefaultlocale()[1]
else:
    # On Unix, if CODESET is available, use that.
    try:
        CODESET
    except NameError:
        # Fall back to parsing environment variables :-(
        def getpreferredencoding(do_setlocale = True):
            """Return the charset that the user is likely using,
            by looking at environment variables."""
            return getdefaultlocale()[1]
    else:
        def getpreferredencoding(do_setlocale = True):
            """Return the charset that the user is likely using,
            according to the system configuration."""
            if do_setlocale:
                oldloc = setlocale(LC_CTYPE)
                setlocale(LC_CTYPE, "")
                result = nl_langinfo(CODESET)
                setlocale(LC_CTYPE, oldloc)
                return result
            else:
                return nl_langinfo(CODESET)


### Database
#
# The following data was extracted from the locale.alias file which
# comes with X11 and then hand edited removing the explicit encoding
# definitions and adding some more aliases. The file is usually
# available as /usr/lib/X11/locale/locale.alias.
#

#
# The local_encoding_alias table maps lowercase encoding alias names
# to C locale encoding names (case-sensitive). Note that normalize()
# first looks up the encoding in the encodings.aliases dictionary and
# then applies this mapping to find the correct C lib name for the
# encoding.
#
locale_encoding_alias = {

    # Mappings for non-standard encoding names used in locale names
    '437':                          'C',
    'c':                            'C',
    'en':                           'ISO8859-1',
    'jis':                          'JIS7',
    'jis7':                         'JIS7',
    'ajec':                         'eucJP',

    # Mappings from Python codec names to C lib encoding names
    'ascii':                        'ISO8859-1',
    'latin_1':                      'ISO8859-1',
    'iso8859_1':                    'ISO8859-1',
    'iso8859_10':                   'ISO8859-10',
    'iso8859_11':                   'ISO8859-11',
    'iso8859_13':                   'ISO8859-13',
    'iso8859_14':                   'ISO8859-14',
    'iso8859_15':                   'ISO8859-15',
    'iso8859_2':                    'ISO8859-2',
    'iso8859_3':                    'ISO8859-3',
    'iso8859_4':                    'ISO8859-4',
    'iso8859_5':                    'ISO8859-5',
    'iso8859_6':                    'ISO8859-6',
    'iso8859_7':                    'ISO8859-7',
    'iso8859_8':                    'ISO8859-8',
    'iso8859_9':                    'ISO8859-9',
    'iso2022_jp':                   'JIS7',
    'shift_jis':                    'SJIS',
    'tactis':                       'TACTIS',
    'euc_jp':                       'eucJP',
    'euc_kr':                       'eucKR',
    'utf_8':                        'UTF8',
    'koi8_r':                       'KOI8-R',
    'koi8_u':                       'KOI8-U',
    # XXX This list is still incomplete. If you know more
    # mappings, please file a bug report. Thanks.
}

#
# The locale_alias table maps lowercase alias names to C locale names
# (case-sensitive). Encodings are always separated from the locale
# name using a dot ('.'); they should only be given in case the
# language name is needed to interpret the given encoding alias
# correctly (CJK codes often have this need).
#
# Note that the normalize() function which uses this tables
# removes '_' and '-' characters from the encoding part of the
# locale name before doing the lookup. This saves a lot of
# space in the table.
#
# MAL 2004-12-10:
# Updated alias mapping to most recent locale.alias file
# from X.org distribution using makelocalealias.py.
#
# These are the differences compared to the old mapping (Python 2.4
# and older):
#
#    updated 'bg' -> 'bg_BG.ISO8859-5' to 'bg_BG.CP1251'
#    updated 'bg_bg' -> 'bg_BG.ISO8859-5' to 'bg_BG.CP1251'
#    updated 'bulgarian' -> 'bg_BG.ISO8859-5' to 'bg_BG.CP1251'
#    updated 'cz' -> 'cz_CZ.ISO8859-2' to 'cs_CZ.ISO8859-2'
#    updated 'cz_cz' -> 'cz_CZ.ISO8859-2' to 'cs_CZ.ISO8859-2'
#    updated 'czech' -> 'cs_CS.ISO8859-2' to 'cs_CZ.ISO8859-2'
#    updated 'dutch' -> 'nl_BE.ISO8859-1' to 'nl_NL.ISO8859-1'
#    updated 'et' -> 'et_EE.ISO8859-4' to 'et_EE.ISO8859-15'
#    updated 'et_ee' -> 'et_EE.ISO8859-4' to 'et_EE.ISO8859-15'
#    updated 'fi' -> 'fi_FI.ISO8859-1' to 'fi_FI.ISO8859-15'
#    updated 'fi_fi' -> 'fi_FI.ISO8859-1' to 'fi_FI.ISO8859-15'
#    updated 'iw' -> 'iw_IL.ISO8859-8' to 'he_IL.ISO8859-8'
#    updated 'iw_il' -> 'iw_IL.ISO8859-8' to 'he_IL.ISO8859-8'
#    updated 'japanese' -> 'ja_JP.SJIS' to 'ja_JP.eucJP'
#    updated 'lt' -> 'lt_LT.ISO8859-4' to 'lt_LT.ISO8859-13'
#    updated 'lv' -> 'lv_LV.ISO8859-4' to 'lv_LV.ISO8859-13'
#    updated 'sl' -> 'sl_CS.ISO8859-2' to 'sl_SI.ISO8859-2'
#    updated 'slovene' -> 'sl_CS.ISO8859-2' to 'sl_SI.ISO8859-2'
#    updated 'th_th' -> 'th_TH.TACTIS' to 'th_TH.ISO8859-11'
#    updated 'zh_cn' -> 'zh_CN.eucCN' to 'zh_CN.gb2312'
#    updated 'zh_cn.big5' -> 'zh_TW.eucTW' to 'zh_TW.big5'
#    updated 'zh_tw' -> 'zh_TW.eucTW' to 'zh_TW.big5'
#
locale_alias = {
    'a3':                                   'a3_AZ.KOI8-C',
    'a3_az':                                'a3_AZ.KOI8-C',
    'a3_az.koi8c':                          'a3_AZ.KOI8-C',
    'af':                                   'af_ZA.ISO8859-1',
    'af_za':                                'af_ZA.ISO8859-1',
    'af_za.iso88591':                       'af_ZA.ISO8859-1',
    'am':                                   'am_ET.UTF-8',
    'american':                             'en_US.ISO8859-1',
    'american.iso88591':                    'en_US.ISO8859-1',
    'ar':                                   'ar_AA.ISO8859-6',
    'ar_aa':                                'ar_AA.ISO8859-6',
    'ar_aa.iso88596':                       'ar_AA.ISO8859-6',
    'ar_ae':                                'ar_AE.ISO8859-6',
    'ar_bh':                                'ar_BH.ISO8859-6',
    'ar_dz':                                'ar_DZ.ISO8859-6',
    'ar_eg':                                'ar_EG.ISO8859-6',
    'ar_eg.iso88596':                       'ar_EG.ISO8859-6',
    'ar_iq':                                'ar_IQ.ISO8859-6',
    'ar_jo':                                'ar_JO.ISO8859-6',
    'ar_kw':                                'ar_KW.ISO8859-6',
    'ar_lb':                                'ar_LB.ISO8859-6',
    'ar_ly':                                'ar_LY.ISO8859-6',
    'ar_ma':                                'ar_MA.ISO8859-6',
    'ar_om':                                'ar_OM.ISO8859-6',
    'ar_qa':                                'ar_QA.ISO8859-6',
    'ar_sa':                                'ar_SA.ISO8859-6',
    'ar_sa.iso88596':                       'ar_SA.ISO8859-6',
    'ar_sd':                                'ar_SD.ISO8859-6',
    'ar_sy':                                'ar_SY.ISO8859-6',
    'ar_tn':                                'ar_TN.ISO8859-6',
    'ar_ye':                                'ar_YE.ISO8859-6',
    'arabic':                               'ar_AA.ISO8859-6',
    'arabic.iso88596':                      'ar_AA.ISO8859-6',
    'az':                                   'az_AZ.ISO8859-9E',
    'az_az':                                'az_AZ.ISO8859-9E',
    'az_az.iso88599e':                      'az_AZ.ISO8859-9E',
    'be':                                   'be_BY.CP1251',
    'be_by':                                'be_BY.CP1251',
    'be_by.cp1251':                         'be_BY.CP1251',
    'be_by.microsoftcp1251':                'be_BY.CP1251',
    'bg':                                   'bg_BG.CP1251',
    'bg_bg':                                'bg_BG.CP1251',
    'bg_bg.cp1251':                         'bg_BG.CP1251',
    'bg_bg.iso88595':                       'bg_BG.ISO8859-5',
    'bg_bg.koi8r':                          'bg_BG.KOI8-R',
    'bg_bg.microsoftcp1251':                'bg_BG.CP1251',
    'bokmal':                               'nb_NO.ISO8859-1',
    'bokm\xe5l':                            'nb_NO.ISO8859-1',
    'br':                                   'br_FR.ISO8859-1',
    'br_fr':                                'br_FR.ISO8859-1',
    'br_fr.iso88591':                       'br_FR.ISO8859-1',
    'br_fr.iso885914':                      'br_FR.ISO8859-14',
    'br_fr.iso885915':                      'br_FR.ISO8859-15',
    'br_fr@euro':                           'br_FR.ISO8859-15',
    'bulgarian':                            'bg_BG.CP1251',
    'c':                                    'C',
    'c-french':                             'fr_CA.ISO8859-1',
    'c-french.iso88591':                    'fr_CA.ISO8859-1',
    'c.en':                                 'C',
    'c.iso88591':                           'en_US.ISO8859-1',
    'c_c':                                  'C',
    'c_c.c':                                'C',
    'ca':                                   'ca_ES.ISO8859-1',
    'ca_es':                                'ca_ES.ISO8859-1',
    'ca_es.iso88591':                       'ca_ES.ISO8859-1',
    'ca_es.iso885915':                      'ca_ES.ISO8859-15',
    'ca_es@euro':                           'ca_ES.ISO8859-15',
    'catalan':                              'ca_ES.ISO8859-1',
    'cextend':                              'en_US.ISO8859-1',
    'cextend.en':                           'en_US.ISO8859-1',
    'chinese-s':                            'zh_CN.eucCN',
    'chinese-t':                            'zh_TW.eucTW',
    'croatian':                             'hr_HR.ISO8859-2',
    'cs':                                   'cs_CZ.ISO8859-2',
    'cs_cs':                                'cs_CZ.ISO8859-2',
    'cs_cs.iso88592':                       'cs_CZ.ISO8859-2',
    'cs_cz':                                'cs_CZ.ISO8859-2',
    'cs_cz.iso88592':                       'cs_CZ.ISO8859-2',
    'cy':                                   'cy_GB.ISO8859-1',
    'cy_gb':                                'cy_GB.ISO8859-1',
    'cy_gb.iso88591':                       'cy_GB.ISO8859-1',
    'cy_gb.iso885914':                      'cy_GB.ISO8859-14',
    'cy_gb.iso885915':                      'cy_GB.ISO8859-15',
    'cy_gb@euro':                           'cy_GB.ISO8859-15',
    'cz':                                   'cs_CZ.ISO8859-2',
    'cz_cz':                                'cs_CZ.ISO8859-2',
    'czech':                                'cs_CZ.ISO8859-2',
    'da':                                   'da_DK.ISO8859-1',
    'da_dk':                                'da_DK.ISO8859-1',
    'da_dk.88591':                          'da_DK.ISO8859-1',
    'da_dk.885915':                         'da_DK.ISO8859-15',
    'da_dk.iso88591':                       'da_DK.ISO8859-1',
    'da_dk.iso885915':                      'da_DK.ISO8859-15',
    'da_dk@euro':                           'da_DK.ISO8859-15',
    'danish':                               'da_DK.ISO8859-1',
    'danish.iso88591':                      'da_DK.ISO8859-1',
    'dansk':                                'da_DK.ISO8859-1',
    'de':                                   'de_DE.ISO8859-1',
    'de_at':                                'de_AT.ISO8859-1',
    'de_at.iso88591':                       'de_AT.ISO8859-1',
    'de_at.iso885915':                      'de_AT.ISO8859-15',
    'de_at@euro':                           'de_AT.ISO8859-15',
    'de_be':                                'de_BE.ISO8859-1',
    'de_be.iso88591':                       'de_BE.ISO8859-1',
    'de_be.iso885915':                      'de_BE.ISO8859-15',
    'de_be@euro':                           'de_BE.ISO8859-15',
    'de_ch':                                'de_CH.ISO8859-1',
    'de_ch.iso88591':                       'de_CH.ISO8859-1',
    'de_ch.iso885915':                      'de_CH.ISO8859-15',
    'de_ch@euro':                           'de_CH.ISO8859-15',
    'de_de':                                'de_DE.ISO8859-1',
    'de_de.88591':                          'de_DE.ISO8859-1',
    'de_de.885915':                         'de_DE.ISO8859-15',
    'de_de.885915@euro':                    'de_DE.ISO8859-15',
    'de_de.iso88591':                       'de_DE.ISO8859-1',
    'de_de.iso885915':                      'de_DE.ISO8859-15',
    'de_de@euro':                           'de_DE.ISO8859-15',
    'de_lu':                                'de_LU.ISO8859-1',
    'de_lu.iso88591':                       'de_LU.ISO8859-1',
    'de_lu.iso885915':                      'de_LU.ISO8859-15',
    'de_lu@euro':                           'de_LU.ISO8859-15',
    'deutsch':                              'de_DE.ISO8859-1',
    'dutch':                                'nl_NL.ISO8859-1',
    'dutch.iso88591':                       'nl_BE.ISO8859-1',
    'ee':                                   'ee_EE.ISO8859-4',
    'ee_ee':                                'ee_EE.ISO8859-4',
    'ee_ee.iso88594':                       'ee_EE.ISO8859-4',
    'eesti':                                'et_EE.ISO8859-1',
    'el':                                   'el_GR.ISO8859-7',
    'el_gr':                                'el_GR.ISO8859-7',
    'el_gr.iso88597':                       'el_GR.ISO8859-7',
    'el_gr@euro':                           'el_GR.ISO8859-15',
    'en':                                   'en_US.ISO8859-1',
    'en.iso88591':                          'en_US.ISO8859-1',
    'en_au':                                'en_AU.ISO8859-1',
    'en_au.iso88591':                       'en_AU.ISO8859-1',
    'en_be':                                'en_BE.ISO8859-1',
    'en_be@euro':                           'en_BE.ISO8859-15',
    'en_bw':                                'en_BW.ISO8859-1',
    'en_ca':                                'en_CA.ISO8859-1',
    'en_ca.iso88591':                       'en_CA.ISO8859-1',
    'en_gb':                                'en_GB.ISO8859-1',
    'en_gb.88591':                          'en_GB.ISO8859-1',
    'en_gb.iso88591':                       'en_GB.ISO8859-1',
    'en_gb.iso885915':                      'en_GB.ISO8859-15',
    'en_gb@euro':                           'en_GB.ISO8859-15',
    'en_hk':                                'en_HK.ISO8859-1',
    'en_ie':                                'en_IE.ISO8859-1',
    'en_ie.iso88591':                       'en_IE.ISO8859-1',
    'en_ie.iso885915':                      'en_IE.ISO8859-15',
    'en_ie@euro':                           'en_IE.ISO8859-15',
    'en_in':                                'en_IN.ISO8859-1',
    'en_nz':                                'en_NZ.ISO8859-1',
    'en_nz.iso88591':                       'en_NZ.ISO8859-1',
    'en_ph':                                'en_PH.ISO8859-1',
    'en_sg':                                'en_SG.ISO8859-1',
    'en_uk':                                'en_GB.ISO8859-1',
    'en_us':                                'en_US.ISO8859-1',
    'en_us.88591':                          'en_US.ISO8859-1',
    'en_us.885915':                         'en_US.ISO8859-15',
    'en_us.iso88591':                       'en_US.ISO8859-1',
    'en_us.iso885915':                      'en_US.ISO8859-15',
    'en_us.iso885915@euro':                 'en_US.ISO8859-15',
    'en_us@euro':                           'en_US.ISO8859-15',
    'en_us@euro@euro':                      'en_US.ISO8859-15',
    'en_za':                                'en_ZA.ISO8859-1',
    'en_za.88591':                          'en_ZA.ISO8859-1',
    'en_za.iso88591':                       'en_ZA.ISO8859-1',
    'en_za.iso885915':                      'en_ZA.ISO8859-15',
    'en_za@euro':                           'en_ZA.ISO8859-15',
    'en_zw':                                'en_ZW.ISO8859-1',
    'eng_gb':                               'en_GB.ISO8859-1',
    'eng_gb.8859':                          'en_GB.ISO8859-1',
    'english':                              'en_EN.ISO8859-1',
    'english.iso88591':                     'en_EN.ISO8859-1',
    'english_uk':                           'en_GB.ISO8859-1',
    'english_uk.8859':                      'en_GB.ISO8859-1',
    'english_united-states':                'en_US.ISO8859-1',
    'english_united-states.437':            'C',
    'english_us':                           'en_US.ISO8859-1',
    'english_us.8859':                      'en_US.ISO8859-1',
    'english_us.ascii':                     'en_US.ISO8859-1',
    'eo':                                   'eo_XX.ISO8859-3',
    'eo_eo':                                'eo_EO.ISO8859-3',
    'eo_eo.iso88593':                       'eo_EO.ISO8859-3',
    'eo_xx':                                'eo_XX.ISO8859-3',
    'eo_xx.iso88593':                       'eo_XX.ISO8859-3',
    'es':                                   'es_ES.ISO8859-1',
    'es_ar':                                'es_AR.ISO8859-1',
    'es_ar.iso88591':                       'es_AR.ISO8859-1',
    'es_bo':                                'es_BO.ISO8859-1',
    'es_bo.iso88591':                       'es_BO.ISO8859-1',
    'es_cl':                                'es_CL.ISO8859-1',
    'es_cl.iso88591':                       'es_CL.ISO8859-1',
    'es_co':                                'es_CO.ISO8859-1',
    'es_co.iso88591':                       'es_CO.ISO8859-1',
    'es_cr':                                'es_CR.ISO8859-1',
    'es_cr.iso88591':                       'es_CR.ISO8859-1',
    'es_do':                                'es_DO.ISO8859-1',
    'es_do.iso88591':                       'es_DO.ISO8859-1',
    'es_ec':                                'es_EC.ISO8859-1',
    'es_ec.iso88591':                       'es_EC.ISO8859-1',
    'es_es':                                'es_ES.ISO8859-1',
    'es_es.88591':                          'es_ES.ISO8859-1',
    'es_es.iso88591':                       'es_ES.ISO8859-1',
    'es_es.iso885915':                      'es_ES.ISO8859-15',
    'es_es@euro':                           'es_ES.ISO8859-15',
    'es_gt':                                'es_GT.ISO8859-1',
    'es_gt.iso88591':                       'es_GT.ISO8859-1',
    'es_hn':                                'es_HN.ISO8859-1',
    'es_hn.iso88591':                       'es_HN.ISO8859-1',
    'es_mx':                                'es_MX.ISO8859-1',
    'es_mx.iso88591':                       'es_MX.ISO8859-1',
    'es_ni':                                'es_NI.ISO8859-1',
    'es_ni.iso88591':                       'es_NI.ISO8859-1',
    'es_pa':                                'es_PA.ISO8859-1',
    'es_pa.iso88591':                       'es_PA.ISO8859-1',
    'es_pa.iso885915':                      'es_PA.ISO8859-15',
    'es_pa@euro':                           'es_PA.ISO8859-15',
    'es_pe':                                'es_PE.ISO8859-1',
    'es_pe.iso88591':                       'es_PE.ISO8859-1',
    'es_pe.iso885915':                      'es_PE.ISO8859-15',
    'es_pe@euro':                           'es_PE.ISO8859-15',
    'es_pr':                                'es_PR.ISO8859-1',
    'es_pr.iso88591':                       'es_PR.ISO8859-1',
    'es_py':                                'es_PY.ISO8859-1',
    'es_py.iso88591':                       'es_PY.ISO8859-1',
    'es_py.iso885915':                      'es_PY.ISO8859-15',
    'es_py@euro':                           'es_PY.ISO8859-15',
    'es_sv':                                'es_SV.ISO8859-1',
    'es_sv.iso88591':                       'es_SV.ISO8859-1',
    'es_sv.iso885915':                      'es_SV.ISO8859-15',
    'es_sv@euro':                           'es_SV.ISO8859-15',
    'es_us':                                'es_US.ISO8859-1',
    'es_uy':                                'es_UY.ISO8859-1',
    'es_uy.iso88591':                       'es_UY.ISO8859-1',
    'es_uy.iso885915':                      'es_UY.ISO8859-15',
    'es_uy@euro':                           'es_UY.ISO8859-15',
    'es_ve':                                'es_VE.ISO8859-1',
    'es_ve.iso88591':                       'es_VE.ISO8859-1',
    'es_ve.iso885915':                      'es_VE.ISO8859-15',
    'es_ve@euro':                           'es_VE.ISO8859-15',
    'estonian':                             'et_EE.ISO8859-1',
    'et':                                   'et_EE.ISO8859-15',
    'et_ee':                                'et_EE.ISO8859-15',
    'et_ee.iso88591':                       'et_EE.ISO8859-1',
    'et_ee.iso885913':                      'et_EE.ISO8859-13',
    'et_ee.iso885915':                      'et_EE.ISO8859-15',
    'et_ee.iso88594':                       'et_EE.ISO8859-4',
    'et_ee@euro':                           'et_EE.ISO8859-15',
    'eu':                                   'eu_ES.ISO8859-1',
    'eu_es':                                'eu_ES.ISO8859-1',
    'eu_es.iso88591':                       'eu_ES.ISO8859-1',
    'eu_es.iso885915':                      'eu_ES.ISO8859-15',
    'eu_es@euro':                           'eu_ES.ISO8859-15',
    'fa':                                   'fa_IR.UTF-8',
    'fa_ir':                                'fa_IR.UTF-8',
    'fa_ir.isiri3342':                      'fa_IR.ISIRI-3342',
    'fi':                                   'fi_FI.ISO8859-15',
    'fi_fi':                                'fi_FI.ISO8859-15',
    'fi_fi.88591':                          'fi_FI.ISO8859-1',
    'fi_fi.iso88591':                       'fi_FI.ISO8859-1',
    'fi_fi.iso885915':                      'fi_FI.ISO8859-15',
    'fi_fi.utf8@euro':                      'fi_FI.UTF-8',
    'fi_fi@euro':                           'fi_FI.ISO8859-15',
    'finnish':                              'fi_FI.ISO8859-1',
    'finnish.iso88591':                     'fi_FI.ISO8859-1',
    'fo':                                   'fo_FO.ISO8859-1',
    'fo_fo':                                'fo_FO.ISO8859-1',
    'fo_fo.iso88591':                       'fo_FO.ISO8859-1',
    'fo_fo.iso885915':                      'fo_FO.ISO8859-15',
    'fo_fo@euro':                           'fo_FO.ISO8859-15',
    'fr':                                   'fr_FR.ISO8859-1',
    'fr_be':                                'fr_BE.ISO8859-1',
    'fr_be.88591':                          'fr_BE.ISO8859-1',
    'fr_be.iso88591':                       'fr_BE.ISO8859-1',
    'fr_be.iso885915':                      'fr_BE.ISO8859-15',
    'fr_be@euro':                           'fr_BE.ISO8859-15',
    'fr_ca':                                'fr_CA.ISO8859-1',
    'fr_ca.88591':                          'fr_CA.ISO8859-1',
    'fr_ca.iso88591':                       'fr_CA.ISO8859-1',
    'fr_ca.iso885915':                      'fr_CA.ISO8859-15',
    'fr_ca@euro':                           'fr_CA.ISO8859-15',
    'fr_ch':                                'fr_CH.ISO8859-1',
    'fr_ch.88591':                          'fr_CH.ISO8859-1',
    'fr_ch.iso88591':                       'fr_CH.ISO8859-1',
    'fr_ch.iso885915':                      'fr_CH.ISO8859-15',
    'fr_ch@euro':                           'fr_CH.ISO8859-15',
    'fr_fr':                                'fr_FR.ISO8859-1',
    'fr_fr.88591':                          'fr_FR.ISO8859-1',
    'fr_fr.iso88591':                       'fr_FR.ISO8859-1',
    'fr_fr.iso885915':                      'fr_FR.ISO8859-15',
    'fr_fr@euro':                           'fr_FR.ISO8859-15',
    'fr_lu':                                'fr_LU.ISO8859-1',
    'fr_lu.88591':                          'fr_LU.ISO8859-1',
    'fr_lu.iso88591':                       'fr_LU.ISO8859-1',
    'fr_lu.iso885915':                      'fr_LU.ISO8859-15',
    'fr_lu@euro':                           'fr_LU.ISO8859-15',
    'fran\xe7ais':                          'fr_FR.ISO8859-1',
    'fre_fr':                               'fr_FR.ISO8859-1',
    'fre_fr.8859':                          'fr_FR.ISO8859-1',
    'french':                               'fr_FR.ISO8859-1',
    'french.iso88591':                      'fr_CH.ISO8859-1',
    'french_france':                        'fr_FR.ISO8859-1',
    'french_france.8859':                   'fr_FR.ISO8859-1',
    'ga':                                   'ga_IE.ISO8859-1',
    'ga_ie':                                'ga_IE.ISO8859-1',
    'ga_ie.iso88591':                       'ga_IE.ISO8859-1',
    'ga_ie.iso885914':                      'ga_IE.ISO8859-14',
    'ga_ie.iso885915':                      'ga_IE.ISO8859-15',
    'ga_ie@euro':                           'ga_IE.ISO8859-15',
    'galego':                               'gl_ES.ISO8859-1',
    'galician':                             'gl_ES.ISO8859-1',
    'gd':                                   'gd_GB.ISO8859-1',
    'gd_gb':                                'gd_GB.ISO8859-1',
    'gd_gb.iso88591':                       'gd_GB.ISO8859-1',
    'gd_gb.iso885914':                      'gd_GB.ISO8859-14',
    'gd_gb.iso885915':                      'gd_GB.ISO8859-15',
    'gd_gb@euro':                           'gd_GB.ISO8859-15',
    'ger_de':                               'de_DE.ISO8859-1',
    'ger_de.8859':                          'de_DE.ISO8859-1',
    'german':                               'de_DE.ISO8859-1',
    'german.iso88591':                      'de_CH.ISO8859-1',
    'german_germany':                       'de_DE.ISO8859-1',
    'german_germany.8859':                  'de_DE.ISO8859-1',
    'gl':                                   'gl_ES.ISO8859-1',
    'gl_es':                                'gl_ES.ISO8859-1',
    'gl_es.iso88591':                       'gl_ES.ISO8859-1',
    'gl_es.iso885915':                      'gl_ES.ISO8859-15',
    'gl_es@euro':                           'gl_ES.ISO8859-15',
    'greek':                                'el_GR.ISO8859-7',
    'greek.iso88597':                       'el_GR.ISO8859-7',
    'gv':                                   'gv_GB.ISO8859-1',
    'gv_gb':                                'gv_GB.ISO8859-1',
    'gv_gb.iso88591':                       'gv_GB.ISO8859-1',
    'gv_gb.iso885914':                      'gv_GB.ISO8859-14',
    'gv_gb.iso885915':                      'gv_GB.ISO8859-15',
    'gv_gb@euro':                           'gv_GB.ISO8859-15',
    'he':                                   'he_IL.ISO8859-8',
    'he_il':                                'he_IL.ISO8859-8',
    'he_il.cp1255':                         'he_IL.CP1255',
    'he_il.iso88598':                       'he_IL.ISO8859-8',
    'he_il.microsoftcp1255':                'he_IL.CP1255',
    'hebrew':                               'iw_IL.ISO8859-8',
    'hebrew.iso88598':                      'iw_IL.ISO8859-8',
    'hi':                                   'hi_IN.ISCII-DEV',
    'hi_in':                                'hi_IN.ISCII-DEV',
    'hi_in.isciidev':                       'hi_IN.ISCII-DEV',
    'hr':                                   'hr_HR.ISO8859-2',
    'hr_hr':                                'hr_HR.ISO8859-2',
    'hr_hr.iso88592':                       'hr_HR.ISO8859-2',
    'hrvatski':                             'hr_HR.ISO8859-2',
    'hu':                                   'hu_HU.ISO8859-2',
    'hu_hu':                                'hu_HU.ISO8859-2',
    'hu_hu.iso88592':                       'hu_HU.ISO8859-2',
    'hungarian':                            'hu_HU.ISO8859-2',
    'icelandic':                            'is_IS.ISO8859-1',
    'icelandic.iso88591':                   'is_IS.ISO8859-1',
    'id':                                   'id_ID.ISO8859-1',
    'id_id':                                'id_ID.ISO8859-1',
    'in':                                   'id_ID.ISO8859-1',
    'in_id':                                'id_ID.ISO8859-1',
    'is':                                   'is_IS.ISO8859-1',
    'is_is':                                'is_IS.ISO8859-1',
    'is_is.iso88591':                       'is_IS.ISO8859-1',
    'is_is.iso885915':                      'is_IS.ISO8859-15',
    'is_is@euro':                           'is_IS.ISO8859-15',
    'iso-8859-1':                           'en_US.ISO8859-1',
    'iso-8859-15':                          'en_US.ISO8859-15',
    'iso8859-1':                            'en_US.ISO8859-1',
    'iso8859-15':                           'en_US.ISO8859-15',
    'iso_8859_1':                           'en_US.ISO8859-1',
    'iso_8859_15':                          'en_US.ISO8859-15',
    'it':                                   'it_IT.ISO8859-1',
    'it_ch':                                'it_CH.ISO8859-1',
    'it_ch.iso88591':                       'it_CH.ISO8859-1',
    'it_ch.iso885915':                      'it_CH.ISO8859-15',
    'it_ch@euro':                           'it_CH.ISO8859-15',
    'it_it':                                'it_IT.ISO8859-1',
    'it_it.88591':                          'it_IT.ISO8859-1',
    'it_it.iso88591':                       'it_IT.ISO8859-1',
    'it_it.iso885915':                      'it_IT.ISO8859-15',
    'it_it@euro':                           'it_IT.ISO8859-15',
    'italian':                              'it_IT.ISO8859-1',
    'italian.iso88591':                     'it_IT.ISO8859-1',
    'iu':                                   'iu_CA.NUNACOM-8',
    'iu_ca':                                'iu_CA.NUNACOM-8',
    'iu_ca.nunacom8':                       'iu_CA.NUNACOM-8',
    'iw':                                   'he_IL.ISO8859-8',
    'iw_il':                                'he_IL.ISO8859-8',
    'iw_il.iso88598':                       'he_IL.ISO8859-8',
    'ja':                                   'ja_JP.eucJP',
    'ja.jis':                               'ja_JP.JIS7',
    'ja.sjis':                              'ja_JP.SJIS',
    'ja_jp':                                'ja_JP.eucJP',
    'ja_jp.ajec':                           'ja_JP.eucJP',
    'ja_jp.euc':                            'ja_JP.eucJP',
    'ja_jp.eucjp':                          'ja_JP.eucJP',
    'ja_jp.iso-2022-jp':                    'ja_JP.JIS7',
    'ja_jp.iso2022jp':                      'ja_JP.JIS7',
    'ja_jp.jis':                            'ja_JP.JIS7',
    'ja_jp.jis7':                           'ja_JP.JIS7',
    'ja_jp.mscode':                         'ja_JP.SJIS',
    'ja_jp.sjis':                           'ja_JP.SJIS',
    'ja_jp.ujis':                           'ja_JP.eucJP',
    'japan':                                'ja_JP.eucJP',
    'japanese':                             'ja_JP.eucJP',
    'japanese-euc':                         'ja_JP.eucJP',
    'japanese.euc':                         'ja_JP.eucJP',
    'japanese.sjis':                        'ja_JP.SJIS',
    'jp_jp':                                'ja_JP.eucJP',
    'ka':                                   'ka_GE.GEORGIAN-ACADEMY',
    'ka_ge':                                'ka_GE.GEORGIAN-ACADEMY',
    'ka_ge.georgianacademy':                'ka_GE.GEORGIAN-ACADEMY',
    'ka_ge.georgianps':                     'ka_GE.GEORGIAN-PS',
    'ka_ge.georgianrs':                     'ka_GE.GEORGIAN-ACADEMY',
    'kl':                                   'kl_GL.ISO8859-1',
    'kl_gl':                                'kl_GL.ISO8859-1',
    'kl_gl.iso88591':                       'kl_GL.ISO8859-1',
    'kl_gl.iso885915':                      'kl_GL.ISO8859-15',
    'kl_gl@euro':                           'kl_GL.ISO8859-15',
    'ko':                                   'ko_KR.eucKR',
    'ko_kr':                                'ko_KR.eucKR',
    'ko_kr.euc':                            'ko_KR.eucKR',
    'ko_kr.euckr':                          'ko_KR.eucKR',
    'korean':                               'ko_KR.eucKR',
    'korean.euc':                           'ko_KR.eucKR',
    'kw':                                   'kw_GB.ISO8859-1',
    'kw_gb':                                'kw_GB.ISO8859-1',
    'kw_gb.iso88591':                       'kw_GB.ISO8859-1',
    'kw_gb.iso885914':                      'kw_GB.ISO8859-14',
    'kw_gb.iso885915':                      'kw_GB.ISO8859-15',
    'kw_gb@euro':                           'kw_GB.ISO8859-15',
    'lithuanian':                           'lt_LT.ISO8859-13',
    'lo':                                   'lo_LA.MULELAO-1',
    'lo_la':                                'lo_LA.MULELAO-1',
    'lo_la.cp1133':                         'lo_LA.IBM-CP1133',
    'lo_la.ibmcp1133':                      'lo_LA.IBM-CP1133',
    'lo_la.mulelao1':                       'lo_LA.MULELAO-1',
    'lt':                                   'lt_LT.ISO8859-13',
    'lt_lt':                                'lt_LT.ISO8859-13',
    'lt_lt.iso885913':                      'lt_LT.ISO8859-13',
    'lt_lt.iso88594':                       'lt_LT.ISO8859-4',
    'lv':                                   'lv_LV.ISO8859-13',
    'lv_lv':                                'lv_LV.ISO8859-13',
    'lv_lv.iso885913':                      'lv_LV.ISO8859-13',
    'lv_lv.iso88594':                       'lv_LV.ISO8859-4',
    'mi':                                   'mi_NZ.ISO8859-1',
    'mi_nz':                                'mi_NZ.ISO8859-1',
    'mi_nz.iso88591':                       'mi_NZ.ISO8859-1',
    'mk':                                   'mk_MK.ISO8859-5',
    'mk_mk':                                'mk_MK.ISO8859-5',
    'mk_mk.cp1251':                         'mk_MK.CP1251',
    'mk_mk.iso88595':                       'mk_MK.ISO8859-5',
    'mk_mk.microsoftcp1251':                'mk_MK.CP1251',
    'ms':                                   'ms_MY.ISO8859-1',
    'ms_my':                                'ms_MY.ISO8859-1',
    'ms_my.iso88591':                       'ms_MY.ISO8859-1',
    'mt':                                   'mt_MT.ISO8859-3',
    'mt_mt':                                'mt_MT.ISO8859-3',
    'mt_mt.iso88593':                       'mt_MT.ISO8859-3',
    'nb':                                   'nb_NO.ISO8859-1',
    'nb_no':                                'nb_NO.ISO8859-1',
    'nb_no.88591':                          'nb_NO.ISO8859-1',
    'nb_no.iso88591':                       'nb_NO.ISO8859-1',
    'nb_no.iso885915':                      'nb_NO.ISO8859-15',
    'nb_no@euro':                           'nb_NO.ISO8859-15',
    'nl':                                   'nl_NL.ISO8859-1',
    'nl_be':                                'nl_BE.ISO8859-1',
    'nl_be.88591':                          'nl_BE.ISO8859-1',
    'nl_be.iso88591':                       'nl_BE.ISO8859-1',
    'nl_be.iso885915':                      'nl_BE.ISO8859-15',
    'nl_be@euro':                           'nl_BE.ISO8859-15',
    'nl_nl':                                'nl_NL.ISO8859-1',
    'nl_nl.88591':                          'nl_NL.ISO8859-1',
    'nl_nl.iso88591':                       'nl_NL.ISO8859-1',
    'nl_nl.iso885915':                      'nl_NL.ISO8859-15',
    'nl_nl@euro':                           'nl_NL.ISO8859-15',
    'nn':                                   'nn_NO.ISO8859-1',
    'nn_no':                                'nn_NO.ISO8859-1',
    'nn_no.88591':                          'nn_NO.ISO8859-1',
    'nn_no.iso88591':                       'nn_NO.ISO8859-1',
    'nn_no.iso885915':                      'nn_NO.ISO8859-15',
    'nn_no@euro':                           'nn_NO.ISO8859-15',
    'no':                                   'no_NO.ISO8859-1',
    'no@nynorsk':                           'ny_NO.ISO8859-1',
    'no_no':                                'no_NO.ISO8859-1',
    'no_no.88591':                          'no_NO.ISO8859-1',
    'no_no.iso88591':                       'no_NO.ISO8859-1',
    'no_no.iso885915':                      'no_NO.ISO8859-15',
    'no_no@euro':                           'no_NO.ISO8859-15',
    'norwegian':                            'no_NO.ISO8859-1',
    'norwegian.iso88591':                   'no_NO.ISO8859-1',
    'ny':                                   'ny_NO.ISO8859-1',
    'ny_no':                                'ny_NO.ISO8859-1',
    'ny_no.88591':                          'ny_NO.ISO8859-1',
    'ny_no.iso88591':                       'ny_NO.ISO8859-1',
    'ny_no.iso885915':                      'ny_NO.ISO8859-15',
    'ny_no@euro':                           'ny_NO.ISO8859-15',
    'nynorsk':                              'nn_NO.ISO8859-1',
    'oc':                                   'oc_FR.ISO8859-1',
    'oc_fr':                                'oc_FR.ISO8859-1',
    'oc_fr.iso88591':                       'oc_FR.ISO8859-1',
    'oc_fr.iso885915':                      'oc_FR.ISO8859-15',
    'oc_fr@euro':                           'oc_FR.ISO8859-15',
    'pd':                                   'pd_US.ISO8859-1',
    'pd_de':                                'pd_DE.ISO8859-1',
    'pd_de.iso88591':                       'pd_DE.ISO8859-1',
    'pd_de.iso885915':                      'pd_DE.ISO8859-15',
    'pd_de@euro':                           'pd_DE.ISO8859-15',
    'pd_us':                                'pd_US.ISO8859-1',
    'pd_us.iso88591':                       'pd_US.ISO8859-1',
    'pd_us.iso885915':                      'pd_US.ISO8859-15',
    'pd_us@euro':                           'pd_US.ISO8859-15',
    'ph':                                   'ph_PH.ISO8859-1',
    'ph_ph':                                'ph_PH.ISO8859-1',
    'ph_ph.iso88591':                       'ph_PH.ISO8859-1',
    'pl':                                   'pl_PL.ISO8859-2',
    'pl_pl':                                'pl_PL.ISO8859-2',
    'pl_pl.iso88592':                       'pl_PL.ISO8859-2',
    'polish':                               'pl_PL.ISO8859-2',
    'portuguese':                           'pt_PT.ISO8859-1',
    'portuguese.iso88591':                  'pt_PT.ISO8859-1',
    'portuguese_brazil':                    'pt_BR.ISO8859-1',
    'portuguese_brazil.8859':               'pt_BR.ISO8859-1',
    'posix':                                'C',
    'posix-utf2':                           'C',
    'pp':                                   'pp_AN.ISO8859-1',
    'pp_an':                                'pp_AN.ISO8859-1',
    'pp_an.iso88591':                       'pp_AN.ISO8859-1',
    'pt':                                   'pt_PT.ISO8859-1',
    'pt_br':                                'pt_BR.ISO8859-1',
    'pt_br.88591':                          'pt_BR.ISO8859-1',
    'pt_br.iso88591':                       'pt_BR.ISO8859-1',
    'pt_br.iso885915':                      'pt_BR.ISO8859-15',
    'pt_br@euro':                           'pt_BR.ISO8859-15',
    'pt_pt':                                'pt_PT.ISO8859-1',
    'pt_pt.88591':                          'pt_PT.ISO8859-1',
    'pt_pt.iso88591':                       'pt_PT.ISO8859-1',
    'pt_pt.iso885915':                      'pt_PT.ISO8859-15',
    'pt_pt.utf8@euro':                      'pt_PT.UTF-8',
    'pt_pt@euro':                           'pt_PT.ISO8859-15',
    'ro':                                   'ro_RO.ISO8859-2',
    'ro_ro':                                'ro_RO.ISO8859-2',
    'ro_ro.iso88592':                       'ro_RO.ISO8859-2',
    'romanian':                             'ro_RO.ISO8859-2',
    'ru':                                   'ru_RU.ISO8859-5',
    'ru_ru':                                'ru_RU.ISO8859-5',
    'ru_ru.cp1251':                         'ru_RU.CP1251',
    'ru_ru.iso88595':                       'ru_RU.ISO8859-5',
    'ru_ru.koi8r':                          'ru_RU.KOI8-R',
    'ru_ru.microsoftcp1251':                'ru_RU.CP1251',
    'ru_ua':                                'ru_UA.KOI8-U',
    'ru_ua.cp1251':                         'ru_UA.CP1251',
    'ru_ua.koi8u':                          'ru_UA.KOI8-U',
    'ru_ua.microsoftcp1251':                'ru_UA.CP1251',
    'rumanian':                             'ro_RO.ISO8859-2',
    'russian':                              'ru_RU.ISO8859-5',
    'se_no':                                'se_NO.UTF-8',
    'serbocroatian':                        'sh_YU.ISO8859-2',
    'sh':                                   'sh_YU.ISO8859-2',
    'sh_hr':                                'sh_HR.ISO8859-2',
    'sh_hr.iso88592':                       'sh_HR.ISO8859-2',
    'sh_sp':                                'sh_YU.ISO8859-2',
    'sh_yu':                                'sh_YU.ISO8859-2',
    'sk':                                   'sk_SK.ISO8859-2',
    'sk_sk':                                'sk_SK.ISO8859-2',
    'sk_sk.iso88592':                       'sk_SK.ISO8859-2',
    'sl':                                   'sl_SI.ISO8859-2',
    'sl_cs':                                'sl_CS.ISO8859-2',
    'sl_si':                                'sl_SI.ISO8859-2',
    'sl_si.iso88592':                       'sl_SI.ISO8859-2',
    'slovak':                               'sk_SK.ISO8859-2',
    'slovene':                              'sl_SI.ISO8859-2',
    'slovenian':                            'sl_SI.ISO8859-2',
    'sp':                                   'sp_YU.ISO8859-5',
    'sp_yu':                                'sp_YU.ISO8859-5',
    'spanish':                              'es_ES.ISO8859-1',
    'spanish.iso88591':                     'es_ES.ISO8859-1',
    'spanish_spain':                        'es_ES.ISO8859-1',
    'spanish_spain.8859':                   'es_ES.ISO8859-1',
    'sq':                                   'sq_AL.ISO8859-2',
    'sq_al':                                'sq_AL.ISO8859-2',
    'sq_al.iso88592':                       'sq_AL.ISO8859-2',
    'sr':                                   'sr_YU.ISO8859-5',
    'sr@cyrillic':                          'sr_YU.ISO8859-5',
    'sr_sp':                                'sr_SP.ISO8859-2',
    'sr_yu':                                'sr_YU.ISO8859-5',
    'sr_yu.cp1251@cyrillic':                'sr_YU.CP1251',
    'sr_yu.iso88592':                       'sr_YU.ISO8859-2',
    'sr_yu.iso88595':                       'sr_YU.ISO8859-5',
    'sr_yu.iso88595@cyrillic':              'sr_YU.ISO8859-5',
    'sr_yu.microsoftcp1251@cyrillic':       'sr_YU.CP1251',
    'sr_yu.utf8@cyrillic':                  'sr_YU.UTF-8',
    'sr_yu@cyrillic':                       'sr_YU.ISO8859-5',
    'sv':                                   'sv_SE.ISO8859-1',
    'sv_fi':                                'sv_FI.ISO8859-1',
    'sv_fi.iso88591':                       'sv_FI.ISO8859-1',
    'sv_fi.iso885915':                      'sv_FI.ISO8859-15',
    'sv_fi@euro':                           'sv_FI.ISO8859-15',
    'sv_se':                                'sv_SE.ISO8859-1',
    'sv_se.88591':                          'sv_SE.ISO8859-1',
    'sv_se.iso88591':                       'sv_SE.ISO8859-1',
    'sv_se.iso885915':                      'sv_SE.ISO8859-15',
    'sv_se@euro':                           'sv_SE.ISO8859-15',
    'swedish':                              'sv_SE.ISO8859-1',
    'swedish.iso88591':                     'sv_SE.ISO8859-1',
    'ta':                                   'ta_IN.TSCII-0',
    'ta_in':                                'ta_IN.TSCII-0',
    'ta_in.tscii':                          'ta_IN.TSCII-0',
    'ta_in.tscii0':                         'ta_IN.TSCII-0',
    'tg':                                   'tg_TJ.KOI8-C',
    'tg_tj':                                'tg_TJ.KOI8-C',
    'tg_tj.koi8c':                          'tg_TJ.KOI8-C',
    'th':                                   'th_TH.ISO8859-11',
    'th_th':                                'th_TH.ISO8859-11',
    'th_th.iso885911':                      'th_TH.ISO8859-11',
    'th_th.tactis':                         'th_TH.TIS620',
    'th_th.tis620':                         'th_TH.TIS620',
    'thai':                                 'th_TH.ISO8859-11',
    'tl':                                   'tl_PH.ISO8859-1',
    'tl_ph':                                'tl_PH.ISO8859-1',
    'tl_ph.iso88591':                       'tl_PH.ISO8859-1',
    'tr':                                   'tr_TR.ISO8859-9',
    'tr_tr':                                'tr_TR.ISO8859-9',
    'tr_tr.iso88599':                       'tr_TR.ISO8859-9',
    'tt':                                   'tt_RU.TATAR-CYR',
    'tt_ru':                                'tt_RU.TATAR-CYR',
    'tt_ru.koi8c':                          'tt_RU.KOI8-C',
    'tt_ru.tatarcyr':                       'tt_RU.TATAR-CYR',
    'turkish':                              'tr_TR.ISO8859-9',
    'turkish.iso88599':                     'tr_TR.ISO8859-9',
    'uk':                                   'uk_UA.KOI8-U',
    'uk_ua':                                'uk_UA.KOI8-U',
    'uk_ua.cp1251':                         'uk_UA.CP1251',
    'uk_ua.iso88595':                       'uk_UA.ISO8859-5',
    'uk_ua.koi8u':                          'uk_UA.KOI8-U',
    'uk_ua.microsoftcp1251':                'uk_UA.CP1251',
    'univ':                                 'en_US.utf',
    'universal':                            'en_US.utf',
    'universal.utf8@ucs4':                  'en_US.UTF-8',
    'ur':                                   'ur_PK.CP1256',
    'ur_pk':                                'ur_PK.CP1256',
    'ur_pk.cp1256':                         'ur_PK.CP1256',
    'ur_pk.microsoftcp1256':                'ur_PK.CP1256',
    'uz':                                   'uz_UZ.UTF-8',
    'uz_uz':                                'uz_UZ.UTF-8',
    'vi':                                   'vi_VN.TCVN',
    'vi_vn':                                'vi_VN.TCVN',
    'vi_vn.tcvn':                           'vi_VN.TCVN',
    'vi_vn.tcvn5712':                       'vi_VN.TCVN',
    'vi_vn.viscii':                         'vi_VN.VISCII',
    'vi_vn.viscii111':                      'vi_VN.VISCII',
    'wa':                                   'wa_BE.ISO8859-1',
    'wa_be':                                'wa_BE.ISO8859-1',
    'wa_be.iso88591':                       'wa_BE.ISO8859-1',
    'wa_be.iso885915':                      'wa_BE.ISO8859-15',
    'wa_be@euro':                           'wa_BE.ISO8859-15',
    'yi':                                   'yi_US.CP1255',
    'yi_us':                                'yi_US.CP1255',
    'yi_us.cp1255':                         'yi_US.CP1255',
    'yi_us.microsoftcp1255':                'yi_US.CP1255',
    'zh':                                   'zh_CN.eucCN',
    'zh_cn':                                'zh_CN.gb2312',
    'zh_cn.big5':                           'zh_TW.big5',
    'zh_cn.euc':                            'zh_CN.eucCN',
    'zh_cn.gb18030':                        'zh_CN.gb18030',
    'zh_cn.gb2312':                         'zh_CN.gb2312',
    'zh_cn.gbk':                            'zh_CN.gbk',
    'zh_hk':                                'zh_HK.big5hkscs',
    'zh_hk.big5':                           'zh_HK.big5',
    'zh_hk.big5hkscs':                      'zh_HK.big5hkscs',
    'zh_tw':                                'zh_TW.big5',
    'zh_tw.big5':                           'zh_TW.big5',
    'zh_tw.euc':                            'zh_TW.eucTW',
}

#
# This maps Windows language identifiers to locale strings.
#
# This list has been updated from
# http://msdn.microsoft.com/library/default.asp?url=/library/en-us/intl/nls_238z.asp
# to include every locale up to Windows XP.
#
# NOTE: this mapping is incomplete.  If your language is missing, please
# submit a bug report to Python bug manager, which you can find via:
#     http://www.python.org/dev/
# Make sure you include the missing language identifier and the suggested
# locale code.
#

windows_locale = {
    0x0436: "af_ZA", # Afrikaans
    0x041c: "sq_AL", # Albanian
    0x0401: "ar_SA", # Arabic - Saudi Arabia
    0x0801: "ar_IQ", # Arabic - Iraq
    0x0c01: "ar_EG", # Arabic - Egypt
    0x1001: "ar_LY", # Arabic - Libya
    0x1401: "ar_DZ", # Arabic - Algeria
    0x1801: "ar_MA", # Arabic - Morocco
    0x1c01: "ar_TN", # Arabic - Tunisia
    0x2001: "ar_OM", # Arabic - Oman
    0x2401: "ar_YE", # Arabic - Yemen
    0x2801: "ar_SY", # Arabic - Syria
    0x2c01: "ar_JO", # Arabic - Jordan
    0x3001: "ar_LB", # Arabic - Lebanon
    0x3401: "ar_KW", # Arabic - Kuwait
    0x3801: "ar_AE", # Arabic - United Arab Emirates
    0x3c01: "ar_BH", # Arabic - Bahrain
    0x4001: "ar_QA", # Arabic - Qatar
    0x042b: "hy_AM", # Armenian
    0x042c: "az_AZ", # Azeri Latin
    0x082c: "az_AZ", # Azeri - Cyrillic
    0x042d: "eu_ES", # Basque
    0x0423: "be_BY", # Belarusian
    0x0445: "bn_IN", # Begali
    0x201a: "bs_BA", # Bosnian
    0x141a: "bs_BA", # Bosnian - Cyrillic
    0x047e: "br_FR", # Breton - France
    0x0402: "bg_BG", # Bulgarian
    0x0403: "ca_ES", # Catalan
    0x0004: "zh_CHS",# Chinese - Simplified
    0x0404: "zh_TW", # Chinese - Taiwan
    0x0804: "zh_CN", # Chinese - PRC
    0x0c04: "zh_HK", # Chinese - Hong Kong S.A.R.
    0x1004: "zh_SG", # Chinese - Singapore
    0x1404: "zh_MO", # Chinese - Macao S.A.R.
    0x7c04: "zh_CHT",# Chinese - Traditional
    0x041a: "hr_HR", # Croatian
    0x101a: "hr_BA", # Croatian - Bosnia
    0x0405: "cs_CZ", # Czech
    0x0406: "da_DK", # Danish
    0x048c: "gbz_AF",# Dari - Afghanistan
    0x0465: "div_MV",# Divehi - Maldives
    0x0413: "nl_NL", # Dutch - The Netherlands
    0x0813: "nl_BE", # Dutch - Belgium
    0x0409: "en_US", # English - United States
    0x0809: "en_GB", # English - United Kingdom
    0x0c09: "en_AU", # English - Australia
    0x1009: "en_CA", # English - Canada
    0x1409: "en_NZ", # English - New Zealand
    0x1809: "en_IE", # English - Ireland
    0x1c09: "en_ZA", # English - South Africa
    0x2009: "en_JA", # English - Jamaica
    0x2409: "en_CB", # English - Carribbean
    0x2809: "en_BZ", # English - Belize
    0x2c09: "en_TT", # English - Trinidad
    0x3009: "en_ZW", # English - Zimbabwe
    0x3409: "en_PH", # English - Phillippines
    0x0425: "et_EE", # Estonian
    0x0438: "fo_FO", # Faroese
    0x0464: "fil_PH",# Filipino
    0x040b: "fi_FI", # Finnish
    0x040c: "fr_FR", # French - France
    0x080c: "fr_BE", # French - Belgium
    0x0c0c: "fr_CA", # French - Canada
    0x100c: "fr_CH", # French - Switzerland
    0x140c: "fr_LU", # French - Luxembourg
    0x180c: "fr_MC", # French - Monaco
    0x0462: "fy_NL", # Frisian - Netherlands
    0x0456: "gl_ES", # Galician
    0x0437: "ka_GE", # Georgian
    0x0407: "de_DE", # German - Germany
    0x0807: "de_CH", # German - Switzerland
    0x0c07: "de_AT", # German - Austria
    0x1007: "de_LU", # German - Luxembourg
    0x1407: "de_LI", # German - Liechtenstein
    0x0408: "el_GR", # Greek
    0x0447: "gu_IN", # Gujarati
    0x040d: "he_IL", # Hebrew
    0x0439: "hi_IN", # Hindi
    0x040e: "hu_HU", # Hungarian
    0x040f: "is_IS", # Icelandic
    0x0421: "id_ID", # Indonesian
    0x045d: "iu_CA", # Inuktitut
    0x085d: "iu_CA", # Inuktitut - Latin
    0x083c: "ga_IE", # Irish - Ireland
    0x0434: "xh_ZA", # Xhosa - South Africa
    0x0435: "zu_ZA", # Zulu
    0x0410: "it_IT", # Italian - Italy
    0x0810: "it_CH", # Italian - Switzerland
    0x0411: "ja_JP", # Japanese
    0x044b: "kn_IN", # Kannada - India
    0x043f: "kk_KZ", # Kazakh
    0x0457: "kok_IN",# Konkani
    0x0412: "ko_KR", # Korean
    0x0440: "ky_KG", # Kyrgyz
    0x0426: "lv_LV", # Latvian
    0x0427: "lt_LT", # Lithuanian
    0x046e: "lb_LU", # Luxembourgish
    0x042f: "mk_MK", # FYRO Macedonian
    0x043e: "ms_MY", # Malay - Malaysia
    0x083e: "ms_BN", # Malay - Brunei
    0x044c: "ml_IN", # Malayalam - India
    0x043a: "mt_MT", # Maltese
    0x0481: "mi_NZ", # Maori
    0x047a: "arn_CL",# Mapudungun
    0x044e: "mr_IN", # Marathi
    0x047c: "moh_CA",# Mohawk - Canada
    0x0450: "mn_MN", # Mongolian
    0x0461: "ne_NP", # Nepali
    0x0414: "nb_NO", # Norwegian - Bokmal
    0x0814: "nn_NO", # Norwegian - Nynorsk
    0x0482: "oc_FR", # Occitan - France
    0x0448: "or_IN", # Oriya - India
    0x0463: "ps_AF", # Pashto - Afghanistan
    0x0429: "fa_IR", # Persian
    0x0415: "pl_PL", # Polish
    0x0416: "pt_BR", # Portuguese - Brazil
    0x0816: "pt_PT", # Portuguese - Portugal
    0x0446: "pa_IN", # Punjabi
    0x046b: "quz_BO",# Quechua (Bolivia)
    0x086b: "quz_EC",# Quechua (Ecuador)
    0x0c6b: "quz_PE",# Quechua (Peru)
    0x0418: "ro_RO", # Romanian - Romania
    0x0417: "rm_CH", # Raeto-Romanese
    0x0419: "ru_RU", # Russian
    0x243b: "smn_FI",# Sami Finland
    0x103b: "smj_NO",# Sami Norway
    0x143b: "smj_SE",# Sami Sweden
    0x043b: "se_NO", # Sami Northern Norway
    0x083b: "se_SE", # Sami Northern Sweden
    0x0c3b: "se_FI", # Sami Northern Finland
    0x203b: "sms_FI",# Sami Skolt
    0x183b: "sma_NO",# Sami Southern Norway
    0x1c3b: "sma_SE",# Sami Southern Sweden
    0x044f: "sa_IN", # Sanskrit
    0x0c1a: "sr_SP", # Serbian - Cyrillic
    0x1c1a: "sr_BA", # Serbian - Bosnia Cyrillic
    0x081a: "sr_SP", # Serbian - Latin
    0x181a: "sr_BA", # Serbian - Bosnia Latin
    0x046c: "ns_ZA", # Northern Sotho
    0x0432: "tn_ZA", # Setswana - Southern Africa
    0x041b: "sk_SK", # Slovak
    0x0424: "sl_SI", # Slovenian
    0x040a: "es_ES", # Spanish - Spain
    0x080a: "es_MX", # Spanish - Mexico
    0x0c0a: "es_ES", # Spanish - Spain (Modern)
    0x100a: "es_GT", # Spanish - Guatemala
    0x140a: "es_CR", # Spanish - Costa Rica
    0x180a: "es_PA", # Spanish - Panama
    0x1c0a: "es_DO", # Spanish - Dominican Republic
    0x200a: "es_VE", # Spanish - Venezuela
    0x240a: "es_CO", # Spanish - Colombia
    0x280a: "es_PE", # Spanish - Peru
    0x2c0a: "es_AR", # Spanish - Argentina
    0x300a: "es_EC", # Spanish - Ecuador
    0x340a: "es_CL", # Spanish - Chile
    0x380a: "es_UR", # Spanish - Uruguay
    0x3c0a: "es_PY", # Spanish - Paraguay
    0x400a: "es_BO", # Spanish - Bolivia
    0x440a: "es_SV", # Spanish - El Salvador
    0x480a: "es_HN", # Spanish - Honduras
    0x4c0a: "es_NI", # Spanish - Nicaragua
    0x500a: "es_PR", # Spanish - Puerto Rico
    0x0441: "sw_KE", # Swahili
    0x041d: "sv_SE", # Swedish - Sweden
    0x081d: "sv_FI", # Swedish - Finland
    0x045a: "syr_SY",# Syriac
    0x0449: "ta_IN", # Tamil
    0x0444: "tt_RU", # Tatar
    0x044a: "te_IN", # Telugu
    0x041e: "th_TH", # Thai
    0x041f: "tr_TR", # Turkish
    0x0422: "uk_UA", # Ukrainian
    0x0420: "ur_PK", # Urdu
    0x0820: "ur_IN", # Urdu - India
    0x0443: "uz_UZ", # Uzbek - Latin
    0x0843: "uz_UZ", # Uzbek - Cyrillic
    0x042a: "vi_VN", # Vietnamese
    0x0452: "cy_GB", # Welsh
}

def _print_locale():

    """ Test function.
    """
    categories = {}
    def _init_categories(categories=categories):
        for k,v in globals().items():
            if k[:3] == 'LC_':
                categories[k] = v
    _init_categories()
    del categories['LC_ALL']

    print 'Locale defaults as determined by getdefaultlocale():'
    print '-'*72
    lang, enc = getdefaultlocale()
    print 'Language: ', lang or '(undefined)'
    print 'Encoding: ', enc or '(undefined)'
    print

    print 'Locale settings on startup:'
    print '-'*72
    for name,category in categories.items():
        print name, '...'
        lang, enc = getlocale(category)
        print '   Language: ', lang or '(undefined)'
        print '   Encoding: ', enc or '(undefined)'
        print

    print
    print 'Locale settings after calling resetlocale():'
    print '-'*72
    resetlocale()
    for name,category in categories.items():
        print name, '...'
        lang, enc = getlocale(category)
        print '   Language: ', lang or '(undefined)'
        print '   Encoding: ', enc or '(undefined)'
        print

    try:
        setlocale(LC_ALL, "")
    except:
        print 'NOTE:'
        print 'setlocale(LC_ALL, "") does not support the default locale'
        print 'given in the OS environment variables.'
    else:
        print
        print 'Locale settings after calling setlocale(LC_ALL, ""):'
        print '-'*72
        for name,category in categories.items():
            print name, '...'
            lang, enc = getlocale(category)
            print '   Language: ', lang or '(undefined)'
            print '   Encoding: ', enc or '(undefined)'
            print

###

try:
    LC_MESSAGES
except NameError:
    pass
else:
    __all__.append("LC_MESSAGES")

if __name__=='__main__':
    print 'Locale aliasing:'
    print
    _print_locale()
    print
    print 'Number formatting:'
    print
    _test()
