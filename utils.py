import re
import time

import requests
import mplfinance as mpf
import matplotlib.pyplot as plt
import pandas as pd

eastmoney_base = "http://push2his.eastmoney.com/api/qt/stock/kline/get?secid={market}.{bench_code}&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58&klt=101&fqt=0&beg={time_begin}&end={time_end}"
#TODO US market

class QueryError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message

class Stock():
    #TODO refined Stock class
    pass

#TODO minus plot
def stock_query(keyword):
    """
    borrowed from https://github.com/pengnanxiaomeimei/stock_data_analysis/
    """
    if keyword.isspace() or not keyword:
        raise QueryError("Empty query")

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
    # print(result)
    try:
        mes_dict = eval(result)
    except NameError as e:
        raise QueryError("Can't find keyword") from e
    query_result = mes_dict['QuotationCodeTable']['Data']

    # filter SH and SZ
    stock_list = [x for x in query_result if x['SecurityType'] == '1' or x['SecurityType'] == '2']

    if not stock_list:
        raise QueryError("Result not in A-SHARE") #TODO may consider broader area
    return stock_list

def data_collector(code='000300', time_begin='19900101', time_end='20991231'):
    stock_url = eastmoney_base.format(market=1, bench_code=code, time_begin=time_begin, time_end=time_end)
    # print(stock_url)
    stock_data = pd.DataFrame(map(lambda x: x.split(','), 
                                   requests.get(stock_url).json()["data"]["klines"]))
    stock_data.columns = ["date", "open", "close", "high", "low", "volume", "money", "change"]
    stock_data["date"] = pd.to_datetime(stock_data["date"])
    return stock_data

def plot_kline(stock_data, title='', output='./test.jpg'):
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
    x = data_collector()
    plot_kline(x)
