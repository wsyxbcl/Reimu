import re
import time

import requests
import mplfinance as mpf
import matplotlib.pyplot as plt
import pandas as pd

#TODO API for split-adjusted share prices
eastmoney_base = "http://push2his.eastmoney.com/api/qt/stock/kline/get?secid={market}.{bench_code}&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58&klt=101&fqt=0&beg={time_begin}&end={time_end}"

#TODO US/HK market

#TODO still need a check
# "沪市（主板、科创板、基金）、深市（主板、中小板、创业板、基金）", guided by Maple
# http://www.szse.cn/
# http://www.sse.com.cn/
_test_stock_code = ['600000', '688111', '510010', '000001', '002001', '300001', '159001']

# For filtering eastmoney searchapi
# "TypeUS" seems to be a strong factor, but with uncertain meaning 
# MktNum: MarketName
stock_market =  {'0': "SZ", '1': "SH"}
# SecurityType: SecurityTypeName
stock_type = {'1': "沪A", 
              '25': "科创板", 
              '2': "深A", 
              '8': "基金", 
              '5': "指数"}

class QueryError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message

class Stock:
    #TODO refined Stock class
    def __init__(self, code, name, market_id, type_id):
        self.code = code
        self.name = name
        self.market_id = market_id
        self.type_id = type_id
    @property
    def stock_market(self):
        return self.stock_market[market_id]
    @property
    def stock_type(self):
        return self.stock_type[type_id]

    def __repr__(self):
        return "<Stock code={0.code!r} name={0.name!r} market_id={0.market_id!r} type_id={0.type_id!r}>".format(self)
    def __str__(self):
        return "{0.name!s}({0.code!s})".format(self)


def _query_test(stock_list):
    """
    Leave test 
    """
    for stock in stock_list:
        try:
            stock_query(stock, echo=True)
        except QueryError:
            print(f"QueryError on {stock}")

#TODO minus plot -> /compare command
#TODO real time price -> /realtime command
def stock_query(keyword, echo=False):
    """
    borrowed from https://github.com/pengnanxiaomeimei/stock_data_analysis/
    Not ideal but works.
    """
    if keyword.isspace() or not keyword:
        raise QueryError("Empty query")
    # configure search API
    query_url = 'http://searchapi.eastmoney.com/api/suggest/get'
    cb_param_pre = 'jQuery112406497239864696334_'
    token = 'D43BF722C8E33BDC906FB84D85E326E8'
    time_stamp = int(round(time.time() * 1000))
    str_parameter = '?cb=' + cb_param_pre + str(time_stamp)
    str_parameter += '&input=' + keyword
    str_parameter += '&token=' + token
    str_parameter += '&type=14' # for Securities entry
    str_parameter += '&count=5'
    str_parameter += '&_=' + str(time_stamp)
    query_url = query_url + str_parameter

    r = requests.get(query_url)
    p2 = re.compile(r'[(](.*)[)]', re.S)
    result = re.findall(p2, r.content.decode('utf-8'))[0]
    try:
        mes_dict = eval(result)
    except NameError as e:
        raise QueryError("Can't find keyword") from e
    query_result = mes_dict['QuotationCodeTable']['Data']

    # Filter result
    stock_list = [Stock(code=x['Code'], name=x['Name'], market_id=x['MktNum'], type_id=x['SecurityType']) 
                  for x in query_result if x['MktNum'] in stock_market and \
                                           x['SecurityType'] in stock_type and\
                                           x["SecurityTypeName"] != "曾用"]
    if not stock_list:
        raise QueryError(f"Result not in A-SHARE\n{query_result}")
    if echo:
        print(stock_list)
    return stock_list

def data_collector(stock, time_begin='19900101', time_end='20991231'):
    stock_url = eastmoney_base.format(market=stock.market_id, 
                                      bench_code=stock.code, 
                                      time_begin=time_begin, 
                                      time_end=time_end)
    try:
        stock_data = pd.DataFrame(map(lambda x: x.split(','), 
                                      requests.get(stock_url).json()["data"]["klines"]))
    except TypeError as e:
        raise QueryError("Can't find kline data") from e
    stock_data.columns = ["date", "open", "close", "high", "low", "volume", "money", "change"]
    stock_data["date"] = pd.to_datetime(stock_data["date"])
    return stock_data

def plot_kline(stock_data, title='', output='./test.jpg'):
    #TODO analysis, e.g. MACD, RSI
    # issue#316 of mplfinance might be helpful
    stock_kline = stock_data.set_index("date")
    stock_kline.index = pd.to_datetime(stock_kline.index)
    stock_kline = stock_kline.astype(float)
    ma_value = (5, 10, 20)
    kwargs = dict(type='candle', mav=ma_value, volume=True, figratio=(11, 8), figscale=0.85)
    style = mpf.make_mpf_style(base_mpf_style='yahoo', rc={'font.size':8})
    fig, axes = mpf.plot(stock_kline, **kwargs, 
                         style=style, 
                         scale_padding={'left': 0.1, 'top': 1, 'right': 1, 'bottom': 1}, 
                         returnfig=True)
    mav_leg = axes[0].legend(['ma_{}'.format(i) for i in ma_value], loc=9, ncol=3, 
                              prop={'size': 7}, fancybox=True, borderaxespad=0.)
    mav_leg.get_frame().set_alpha(0.4)
    axes[0].set_title(title)
    fig.savefig(output, dpi=300)

if __name__ == '__main__':
    x = data_collector(stock_query('000300', echo=True)[0])
    plot_kline(x)
    # _query_test(_test_stock_code)