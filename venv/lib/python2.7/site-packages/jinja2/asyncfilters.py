from functools import wraps

from jinja2.asyncsupport import auto_aiter
from jinja2 import filters


async def auto_to_seq(value):
    seq = []
    if hasattr(value, '__aiter__'):
        async for item in value:
            seq.append(item)
    else:
        for item in value:
            seq.append(item)
    return seq


async def async_select_or_reject(args, kwargs, modfunc, lookup_attr):
    seq, func = filters.prepare_select_or_reject(
        args, kwargs, modfunc, lookup_attr)
    if seq:
        async for item in auto_aiter(seq):
            if func(item):
                yield item


def dualfilter(normal_filter, async_filter):
    wrap_evalctx = False
    if getattr(normal_filter, 'environmentfilter', False):
        is_async = lambda args: args[0].is_async
        wrap_evalctx = False
    else:
        if not getattr(normal_filter, 'evalcontextfilter', False) and \
           not getattr(normal_filter, 'contextfilter', False):
            wrap_evalctx = True
        is_async = lambda args: args[0].environment.is_async

    @wraps(normal_filter)
    def wrapper(*args, **kwargs):
        b = is_async(args)
        if wrap_evalctx:
            args = args[1:]
        if b:
            return async_filter(*args, **kwargs)
        return normal_filter(*args, **kwargs)

    if wrap_evalctx:
        wrapper.evalcontextfilter = True

    wrapper.asyncfiltervariant = True

    return wrapper


def asyncfiltervariant(original):
    def decorator(f):
        return dualfilter(original, f)
    return decorator


@asyncfiltervariant(filters.do_first)
async def do_first(environment, seq):
    try:
        return await auto_aiter(seq).__anext__()
    except StopAsyncIteration:
        return environment.undefined('No first item, sequence was empty.')


@asyncfiltervariant(filters.do_groupby)
async def do_groupby(environment, value, attribute):
    expr = filters.make_attrgetter(environment, attribute)
    return [filters._GroupTuple(key, await auto_to_seq(values))
            for key, values in filters.groupby(sorted(
                await auto_to_seq(value), key=expr), expr)]


@asyncfiltervariant(filters.do_join)
async def do_join(eval_ctx, value, d=u'', attribute=None):
    return filters.do_join(eval_ctx, await auto_to_seq(value), d, attribute)


@asyncfiltervariant(filters.do_list)
async def do_list(value):
    return await auto_to_seq(value)


@asyncfiltervariant(filters.do_reject)
async def do_reject(*args, **kwargs):
    return async_select_or_reject(args, kwargs, lambda x: not x, False)


@asyncfiltervariant(filters.do_rejectattr)
async def do_rejectattr(*args, **kwargs):
    return async_select_or_reject(args, kwargs, lambda x: not x, True)


@asyncfiltervariant(filters.do_select)
async def do_select(*args, **kwargs):
    return async_select_or_reject(args, kwargs, lambda x: x, False)


@asyncfiltervariant(filters.do_selectattr)
async def do_selectattr(*args, **kwargs):
    return async_select_or_reject(args, kwargs, lambda x: x, True)


@asyncfiltervariant(filters.do_map)
async def do_map(*args, **kwargs):
    seq, func = filters.prepare_map(args, kwargs)
    if seq:
        async for item in auto_aiter(seq):
            yield func(item)


@asyncfiltervariant(filters.do_sum)
async def do_sum(environment, iterable, attribute=None, start=0):
    rv = start
    if attribute is not None:
        func = filters.make_attrgetter(environment, attribute)
    else:
        func = lambda x: x
    async for item in auto_aiter(iterable):
        rv += func(item)
    return rv


@asyncfiltervariant(filters.do_slice)
async def do_slice(value, slices, fill_with=None):
    return filters.do_slice(await auto_to_seq(value), slices, fill_with)


ASYNC_FILTERS = {
    'first':        do_first,
    'groupby':      do_groupby,
    'join':         do_join,
    'list':         do_list,
    # we intentionally do not support do_last because that would be
    # ridiculous
    'reject':       do_reject,
    'rejectattr':   do_rejectattr,
    'map':          do_map,
    'select':       do_select,
    'selectattr':   do_selectattr,
    'sum':          do_sum,
    'slice':        do_slice,
}
