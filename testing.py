from functools import wraps

from utils import *

# "沪市（主板、科创板、基金）、深市（主板、中小板、创业板、基金）", guided by Maple
# http://www.szse.cn/
# http://www.sse.com.cn/
_test_stock_code_CN = ['600000', '688111', '510010', '000001', '002001', '300001', '159001']
_test_stock_HK = ["腾讯控股", "美团", "小米集团", "舜宇光学科技", "华虹半导体", "思摩尔国际", "海底捞"]

def timing(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print('{}.time'.format(func.__name__))
        start = time.perf_counter()
        func_return = func(*args, **kwargs)
        end = time.perf_counter()
        print("runtime: {}".format(end - start))
        return func_return
    return wrapper

def timing_async(func):
    async def helper(func, *args, **params):
        if asyncio.iscoroutinefunction(func):
            print("coroutine function")
            return await func(*args, **params)
        else:
            print("not a coroutine function")
            return func(*args, **params)

    @wraps(func)
    async def wrapper(*args, **params):
        print('{}.time'.format(func.__name__))
        start = time.time()
        func_return = await helper(func, *args, **params)
        print("runtime: {}".format(time.time() - start))
        return func_return
    return wrapper

def _query_test(stock_list):
    """
    Leave test 
    """
    for stock in stock_list:
        try:
            stock_query(stock, echo=True)
        except QueryError:
            print(f"QueryError on {stock}")