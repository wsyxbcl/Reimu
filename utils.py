import datetime
import os
import pickle
import re
import time

import mplfinance as mpf
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

matplotlib.rcParams['font.family'] = ['Source Han Sans']

data_path = './data'
_test_path = './demo'
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

class Stock_mix:
    """
    Mixed stocks, Inherence not considered currently 
    """
    def __init__(self, code, name, stock_list, holding_ratio, create_time=datetime.datetime.utcnow()):
        self.code = code
        self.name = name
        self.stock_list = stock_list
        self.holding_ratio = holding_ratio
        self.create_time = create_time #TODO just date is fine, consider using utf-8

    def draw(self, benefits= None, output=os.path.join(_test_path, 'stock_mix.jpg')):
        #TODO may combine with benefits
        labels = [stock.name for stock in self.stock_list]
        ratios = [ratio for ratio in self.holding_ratio]
        colors = plt.cm.get_cmap('tab20c').colors
        fig1, ax1 = plt.subplots()
        ax1.pie(ratios, colors=colors, labels=labels, autopct='%1.1f%%', startangle=90) #draw circle
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle) # Equal aspect ratio ensures that pie is drawn as a circle
        ax1.axis('equal')
        ax1.set_title(f"{self.name}({self.code}), created at {self.create_time}(UTC)")
        plt.tight_layout()
        fig.savefig(output, dpi=300)

    def save(self):
        output=os.path.join(data_path, self.code+'.pickle')
        with open(output, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        
    def __repr__(self):
        return "<Stock_mix code={0.code!r} name={0.name!r}>".format(self)
    def __str__(self):
        return "{0.name!s}({0.code!s}), created at {0.create_time!s}\n".format(self) +\
               "\n".join("{!s}\t{:.1%}".format(stock, ratio)\
               for stock, ratio in zip(self.stock_list, self.holding_ratio))


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
    if (local_stock := keyword+'.pickle') in os.listdir(data_path):
        #TODO do query instead of match
        with open(local_stock, 'rb') as f:
            local_stock = pickle.load(f)
        return [local_stock] # to make return value consistent

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

def mix_data_collector(stock_mix, price='norm', time_begin='20210101', time_end='20991231'):
    """
    Collecting and postprocessing for Stock_mix, where only close price are collected
    Noted that long time range can cause date inconsistency
    
    price: 'norm' or 'average'
    """
    stock_data = [data_collector(stock, time_begin='20210101') for stock in enl_stock_mix.stock_list]
    # Checking whether the dates are consistent
    try:
        matrix_date = np.array([np.array(stock['date']) for stock in stock_data])
    except ValueError:
        # operands could not be broadcast together
        raise
    if not np.equal(matrix_date[0], matrix_date).all():
        print("date inconsistent")

    matrix_close_price = np.array([np.array(stock['close']) for stock in stock_data]).astype(float)
    matrix_volume = np.array([np.array(stock['volume']) for stock in stock_data]).astype(float)
    # only need close price here
    close_price_mix = np.average(matrix_close_price, axis=0, weights=enl_stock_ratio)
    if price == 'norm':
        close_price_mix = close_price_mix / close_price_mix[0] # norm to time_begin
    volume_mix = np.sum(matrix_volume, axis=0)
    mix_data = stock_data[0].copy()
    mix_data = mix_data.drop(['money', 'change'], axis=1)
    mix_data['close'] = close_price_mix
    mix_data['volume'] = volume_mix
    # Data redundancy, rather inelegant here, might go PR on mplfinance (or simplily using plot instead)
    mix_data['low'] = mix_data['open'] = mix_data['high'] = np.zeros(len(stock_data[0]['date']))
    return mix_data

def plot_kline(stock_data, title='', plot_type='candle', output=os.path.join(_test_path, 'kline.jpg')):
    #TODO analysis, e.g. MACD, RSI
    # issue#316 of mplfinance might be helpful
    stock_kline = stock_data.set_index("date")
    stock_kline.index = pd.to_datetime(stock_kline.index)
    stock_kline = stock_kline.astype(float)
    if plot_type == 'line':
        ma_value = ()
    else:
        ma_value = (5, 10, 20)
    kwargs = dict(type=plot_type, mav=ma_value, volume=True, figratio=(11, 8), figscale=0.85)
    style = mpf.make_mpf_style(base_mpf_style='yahoo', rc={'font.size':8})
    fig, axes = mpf.plot(stock_kline, **kwargs, 
                         style=style, 
                         scale_padding={'left': 0.1, 'top': 1, 'right': 1, 'bottom': 1}, 
                         returnfig=True)
    if ma_value:
        mav_leg = axes[0].legend(['ma_{}'.format(i) for i in ma_value], loc=9, ncol=3, 
                                prop={'size': 7}, fancybox=True, borderaxespad=0.)
        mav_leg.get_frame().set_alpha(0.4)
    axes[0].set_title(title)
    fig.savefig(output, dpi=300)

def gen_stock_mix(mix_code, mix_name, stock_names, holding_ratios):
    stock_list = []
    for stock_name in stock_names:
        query_result = stock_query(stock_name, echo=True)
        if len(query_result) == 1:
            stock_list.append(query_result[0])
        else:
            print("multiple query results on "+stock_name)
    stock_mix = Stock_mix(code=mix_code, name=mix_name, stock_list=stock_list, 
                              holding_ratio=holding_ratios)
    stock_mix.save()
    print(stock_mix)
    return stock_mix

if __name__ == '__main__':
    # x = data_collector(stock_query('000300', echo=True)[0])
    # plot_kline(x, plot_type='line')
    # _query_test(_test_stock_code)
    enl_stock_name = ["隆基股份", "通威股份", "宁德时代", "亿纬锂能", "药明康德", 
                      "华大基因", "中船防务", "航发动力", "海康威视", "金山办公", 
                      "石头科技", "恒力石化", "三一重工", "恒立液压", "上机数控", 
                      "金域医学", "英科医疗", "安井食品", "高德红外", "比亚迪", 
                      "东方财富"]
    enl_stock_ratio = [1/len(enl_stock_name)] * len(enl_stock_name)
    enl_stock_mix = gen_stock_mix(mix_code='enl001', mix_name="enl stock mix", stock_names=enl_stock_name, holding_ratios=enl_stock_ratio)
    enl_stock_mix.draw()
    print(enl_stock_mix)
    mix_data = mix_data_collector(enl_stock_mix)
    plot_kline(mix_data, title=enl_stock_mix.name, plot_type='line')
