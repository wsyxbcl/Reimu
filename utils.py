import aiohttp
import asyncio
import datetime
from functools import wraps, reduce
import hashlib
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
# fqt is for split-adjusted price
# eastmoney_base = "http://push2his.eastmoney.com/api/qt/stock/kline/get?secid={market}.{bench_code}&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58&klt=101&fqt=0&beg={time_begin}&end={time_end}"
eastmoney_base = "http://push2his.eastmoney.com/api/qt/stock/kline/get?secid={market}.{bench_code}&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58&klt=101&fqt=1&beg={time_begin}&end={time_end}"


# "沪市（主板、科创板、基金）、深市（主板、中小板、创业板、基金）", guided by Maple
# http://www.szse.cn/
# http://www.sse.com.cn/
_test_stock_code_CN = ['600000', '688111', '510010', '000001', '002001', '300001', '159001']
_test_stock_HK = ["腾讯控股", "美团", "小米集团", "舜宇光学科技", "华虹半导体", "思摩尔国际", "海底捞"]
# For filtering eastmoney searchapi
# "TypeUS" seems to be a strong factor, but with uncertain meaning 
# MktNum: MarketName
stock_market =  {'0': "SZ", '1': "SH", '105': "US", '106': "US", '107': "US", '156': "US", '100': "US", '116': "HK"}

# SecurityType: SecurityTypeName
stock_type = {'1': "沪A", 
              '25': "科创板", 
              '2': "深A", 
              '8': "基金", 
              '5': "指数", 
              '19': "港股", 
              '6': "港股"}

# kline color style
mc = {'candle': {'up': '#fe3032', 'down': '#00b060'},
      'edge': {'up': '#fe3032', 'down': '#00b060'},
      'wick': {'up': '#fe3032', 'down': '#00b060'},
      'ohlc': {'up': '#fe3032', 'down': '#00b060'},
      'volume': {'up': '#fd6b6c', 'down': '#4dc790'},
      'vcedge': {'up': '#1f77b4', 'down': '#1f77b4'},
      'vcdopcod': False,
      'alpha': 0.7}

class QueryError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message

class Stock:
    def __init__(self, code, name, market_id, type_id):
        self.code = code
        self.name = name
        self.market_id = market_id
        self.type_id = type_id

    @property
    def market(self):
        return stock_market[self.market_id]
    @property
    def stock_type(self):
        return self.stock_type[type_id]
    @property
    def md5(self):
        m = hashlib.md5()
        s = ''.join([self.code, self.name, self.market_id, self.type_id])
        m.update(s.encode())
        return m.hexdigest()

    def collect_data(self, time_begin='19900101', time_end='20991231'):
        stock_url = eastmoney_base.format(market=self.market_id, 
                                          bench_code=self.code, 
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

    def __repr__(self):
        return "<Stock code={0.code!r} name={0.name!r} market_id={0.market_id!r} type_id={0.type_id!r}>".format(self)
    def __str__(self):
        return "{0.name!s}({0.code!s})".format(self)

class Stock_mix:
    """
    Mixed stocks, Inherence not considered currently 
    """
    def __init__(self, code, name, stock_list, holding_ratio, create_time):
        self.code = code
        self.name = name
        self.stock_list = stock_list
        self.holding_ratio = holding_ratio
        self.create_time = create_time 

    def draw(self, output=os.path.join(_test_path, 'stock_mix.jpg')):
        labels = [stock.name for stock in self.stock_list]
        ratios = [ratio for ratio in self.holding_ratio]
        colors = plt.cm.get_cmap('tab20c').colors
        matplotlib.rcParams['font.family'] = ['Source Han Sans']
        fig1, ax1 = plt.subplots()
        ax1.pie(ratios, colors=colors, labels=labels, autopct='%1.1f%%', startangle=90) #draw circle
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle) # Equal aspect ratio ensures that pie is drawn as a circle
        ax1.axis('equal')
        ax1.set_title(f"{self.name}({self.code}), created at {self.create_time}(UTC)")
        plt.tight_layout()
        fig.savefig(output, dpi=300)
        plt.close(fig)

    def save(self):
        output=os.path.join(data_path, self.code+'.pickle')
        with open(output, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        
    def get_profit_ratio(self, mix_data, matrix_close_price, date_ref=None):
        # mix_data can be calculated from matrix_close_price, 
        # just for saving time here

        # Match datetime with utc, NOTICE that timezone are assumed to be same
        # in one mixed stock 
        # date_ref == 'latest': return profit_ratio based on the latest two days (dirty fix for /now

        # Timezone convert
        #TODO a utc convert method for Stock_mix
        if date_ref == 'latest':
            date_ref_latest = True
        else:
            date_ref_latest = False
            if date_ref is None:
                date_ref_index_utc = pd.Timestamp(self.create_time).tz_localize('UTC')
            else:
                date_ref_index_utc = pd.Timestamp(date_ref).tz_localize('UTC')
            date_ref_index = (
                date_ref_index_utc.tz_convert('Asia/Shanghai').date()
                if stock_market[self.stock_list[0].market_id] == 'SZ' or 'SH' or 'HK'
                else date_ref_index_utc.tz_convert('US/Eastern').date())

        get_value = lambda x: (x.index.values[0], x.values[0])
        if date_ref_latest:
            mix_price_ref_idx, mix_price_ref = get_value(mix_data.loc[mix_data['date'] == list(mix_data['date'])[-2]]['close'])
        else:
            for i in range(9):
                try:
                    mix_price_ref_idx, mix_price_ref = get_value(mix_data.loc[mix_data['date'] == str(date_ref_index - datetime.timedelta(days=i))]['close'])
                except IndexError:
                    continue
                break
        try:
            mix_price_today_idx, _ = get_value(mix_data.loc[mix_data['date'] == list(mix_data['date'])[-1]]['close'])
        except UnboundLocalError:
            print("No ref data in mix_data")
            raise
        profit_ratio = (mix_data['close'].values - mix_price_ref) / mix_price_ref

        matrix_price_ref = matrix_close_price[:, mix_price_ref_idx]
        matrix_price_today = matrix_close_price[:, mix_price_today_idx]
        # matrix_profit_ratio = (matrix_close_price - matrix_price_ref.reshape(-1, 1)) / matrix_price_ref.reshape(-1, 1)
        stock_profit_ratio = (matrix_price_today - matrix_price_ref) / matrix_price_ref
        return profit_ratio, stock_profit_ratio

    def __repr__(self):
        return "<Stock_mix code={0.code!r} name={0.name!r}>".format(self)
    def __str__(self):
        holding_ratio_sum = sum([float(ratio) for ratio in self.holding_ratio])
        holding_ratio_norm = [float(ratio) / holding_ratio_sum for ratio in self.holding_ratio]
        return "{0.name!s}({0.code!s}), created at {0.create_time!s}\n".format(self) +\
               "\n".join("{!s}\t{:.1%}".format(stock, ratio)\
               for stock, ratio in zip(self.stock_list, holding_ratio_norm))

# Test utilities
def _query_test(stock_list):
    """
    Leave test 
    """
    for stock in stock_list:
        try:
            stock_query(stock, echo=True)
        except QueryError:
            print(f"QueryError on {stock}")

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

# Data collecting utilities
def stock_query(keyword, filter_md5=None, filter_code=None, echo=False):
    """
    borrowed from https://github.com/pengnanxiaomeimei/stock_data_analysis/
    Not ideal but works.
    """
    if (local_stock := (keyword+'.pickle')) in os.listdir(data_path):
        with open(os.path.join(data_path, local_stock), 'rb') as f:
            local_stock = pickle.load(f)
        return [local_stock] # to make return value consistent

    if keyword.isspace() or not keyword:
        raise QueryError("Empty query on <{}>".format(keyword))
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
        raise QueryError(f"Can't find keyword {keyword}") from e
    query_result = mes_dict['QuotationCodeTable']['Data']
    # if echo:
    #     print(query_result)
    stock_list = [Stock(code=x['Code'], name=x['Name'], market_id=x['MktNum'], type_id=x['SecurityType']) 
                  for x in query_result if x['MktNum'] in stock_market and \
                                           (x['SecurityType'] in stock_type or x['Classify'] == "UsStock" or x['Classify'] == "UniversalIndex") and \
                                           x["SecurityTypeName"] != "曾用"]
    if filter_md5:
        stock_list = [stock for stock in stock_list if stock.md5 == filter_md5]
    if filter_code:
        # to make sure that one can always define a Stock_mix by codes
        if keyword in [stock.code for stock in stock_list]:
            stock_list = [stock for stock in stock_list if stock.code == keyword]
    if echo:
        print(stock_list)
    if not stock_list:
        raise QueryError(f"Empty stock_list from \n{query_result}")
    return stock_list

# @timing
def mix_data_collector(stock_mix, time_begin='20210101', time_end='20991231', time_ref=None):
    """
    Collecting and postprocessing for Stock_mix, where only close price are collected
    Noted that long time range can cause date inconsistency
    """
    if time_ref is None:
        time_ref = time_begin
    stock_data = [stock.collect_data(time_begin=time_begin, time_end=time_end) for stock in stock_mix.stock_list]
    try:
        matrix_date = np.array([stock['date'].values for stock in stock_data], dtype=object)
        #TODO #17
            # for i, date in enumerate(matrix_date):
            #     if i == 0:
            #         date_ref = date
            #     else:
            #         date_ref = matrix_date[i-1]
            #     print(str(i)+' '+str(len(date)))
            #     if len(date_ref) != (date):
            #         print(set(date_ref) - set(date))
    except ValueError:
        # operands could not be broadcast together
        raise
    if not np.equal(matrix_date[0], matrix_date).all():
        print("date inconsistent")

    matrix_close_price = np.array([np.array(stock['close']) for stock in stock_data]).astype(float)
    # matrix_volume = np.array([np.array(stock['volume']) for stock in stock_data]).astype(float)
    # only need close price here
    close_price_ref = matrix_close_price[:, 0]
    stock_share_ratios = stock_mix.holding_ratio / close_price_ref
    value_mix = np.average(matrix_close_price, axis=0, weights=stock_share_ratios) 
    value_mix = value_mix / value_mix[0] # norm to 1
    mix_data = stock_data[0].copy()
    mix_data = mix_data.drop(['money', 'change', 'volume'], axis=1)
    mix_data['close'] = value_mix
    # mix_data['volume'] = volume_mix
    # Data redundancy, rather inelegant here, might go PR on mplfinance (or simplily using plot instead)
    mix_data['low'] = mix_data['open'] = mix_data['high'] = np.zeros(len(stock_data[0]['date']))
    return mix_data, matrix_close_price # to be used in profit analysis


# Plot utilities
def plot_kline(stock_data, title='', plot_type='candle', volume=True, macd=False, output=os.path.join(_test_path, 'kline.jpg')):
    # issue#316 of mplfinance might be helpful
    stock_kline = stock_data.set_index("date")
    stock_kline.index = pd.to_datetime(stock_kline.index)
    stock_kline = stock_kline.astype(float)
    if plot_type == 'line':
        ma_value = ()
    else:
        ma_value = (5, 10, 20)
    if macd:
        # Using MACD(12,26,9) here
        # https://en.wikipedia.org/wiki/MACD
        ema_12 = stock_data['close'].ewm(span=12, adjust=False).mean()
        ema_26 = stock_data['close'].ewm(span=26, adjust=False).mean()
        macd_line = ema_12 - ema_26
        macd_signal = macd_line.ewm(span=9, adjust=False).mean()
        macd_histogram = macd_line - macd_signal

        apds = [# mpf.make_addplot(ema_12, color='lime'),
                # mpf.make_addplot(ema_26, color='c'),
                mpf.make_addplot(macd_histogram, type='bar', width=0.7, panel=1,
                                 color='dimgray', alpha=1, secondary_y=False),
                mpf.make_addplot(macd_line, panel=1, color='b', width=0.5, secondary_y=True),
                mpf.make_addplot(macd_signal, panel=1, color='r', width=0.5, secondary_y=True)]
        if volume:
            kwargs = dict(type=plot_type, addplot=apds, mav=ma_value, volume=volume, 
                          figratio=(4, 3), figscale=0.85, volume_panel=2, panel_ratios=(6, 3, 2))
        else:
            kwargs = dict(type=plot_type, addplot=apds, mav=ma_value, volume=volume, 
                          figratio=(11, 8), figscale=0.85)
    else:
        kwargs = dict(type=plot_type, mav=ma_value, volume=volume, figratio=(11, 8), figscale=0.85)
    style = mpf.make_mpf_style(base_mpf_style='yahoo', rc={'font.size':8}, marketcolors=mc)
    fig, axes = mpf.plot(stock_kline, **kwargs, 
                         style=style, 
                         scale_padding={'left': 0.4, 'top': 1, 'right': 1, 'bottom': 1}, 
                         returnfig=True)
    if ma_value:
        mav_leg = axes[0].legend(['ma_{}'.format(i) for i in ma_value], loc=9, ncol=3, 
                                prop={'size': 7}, fancybox=True, borderaxespad=0.)
        mav_leg.get_frame().set_alpha(0.4)
    if macd:
        mav_leg = axes[3].legend(["MACD", "MACD Signal"], loc=9, ncol=3, 
                                prop={'size': 7}, fancybox=True, borderaxespad=0.)
        mav_leg.get_frame().set_alpha(0.4)
    axes[0].set_title(title)
    fig.savefig(output, dpi=300)
    plt.close(fig)

def plot_profitline(stock_data, profit_ratio, title='', output=os.path.join(_test_path, 'profitline.jpg')):
    stock_data['close'] = profit_ratio
    stock_profitline = stock_data.set_index("date")
    stock_profitline.index = pd.to_datetime(stock_profitline.index)
    stock_profitline = stock_profitline.astype(float)
    kwargs = dict(type='line', volume=False, ylabel='Return Rate', figratio=(11, 8), figscale=0.85)
    style = mpf.make_mpf_style(base_mpf_style='yahoo', rc={'font.size':8})
    fig, axes = mpf.plot(stock_profitline, **kwargs, 
                         style=style, 
                         scale_padding={'left': 0.1, 'top': 1, 'right': 1, 'bottom': 1}, 
                         returnfig=True)
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))
    axes[0].set_title(title)
    fig.savefig(output, dpi=300)
    plt.close(fig)
   
def plot_stock_profit(stock_mix, stock_profit_ratio, title='', output=os.path.join(_test_path, 'profitstocks.jpg')):
    stock_df = pd.DataFrame()
    stock_df['name'] = [stock.name for stock in stock_mix.stock_list]
    stock_df['profit'] = stock_profit_ratio
    stock_df['colors'] = ['red' if x >= 0 else 'green' for x in stock_df['profit']]
    stock_df.sort_values('profit', inplace=True)
    stock_df.reset_index(inplace=True)

    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    matplotlib.rcParams['font.family'] = ['Source Han Sans']
    plt.figure()
    plt.hlines(y=stock_df.index, xmin=0, xmax=stock_df.profit, color=stock_df.colors, alpha=0.4, linewidth=5)

    plt.gca().set(ylabel='Stock', xlabel='Return Rate')
    plt.gca().xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))
    plt.yticks(stock_df.index, stock_df.name, fontsize=8)
    plt.title(title)
    plt.grid(linestyle='--', alpha=0.5)
    plt.savefig(output, dpi=300)
    plt.close()

def plot_return_rate_anlys(collection, date_begin, ref=None, excess_return=False, output=os.path.join(_test_path, 'compare.jpg')):
    """
    Perform return rate anaylsis on collection of stock or stock_mix, by plotting return rates in same axis. 
    """
    if ref is None:
        ref_idx = 0
    collection_rr = []
    for stock in collection:
        stock_kline = stock.collect_data(time_begin=date_begin).set_index('date')
        stock_kline.index = pd.to_datetime(stock_kline.index)
        stock_kline = stock_kline.astype(float)
        stock_kline[stock.name] = (stock_kline['close'] - stock_kline['close'][ref_idx]) / stock_kline['close'][ref_idx]
        collection_rr.append(stock_kline[stock.name])
    collection_rr_df = reduce(lambda x, y: pd.merge(x, y, how='outer', on='date', sort=True), collection_rr)

    # create a 'base layer' placeholder for plot
    place_holder = np.empty(collection_rr_df.shape[0])
    place_holder[:] = np.nan
    collection_rr_df['close'] = collection_rr_df['low'] = collection_rr_df['open'] = collection_rr_df['high'] = place_holder

    apdict = [mpf.make_addplot(collection_rr_df[stock.name]) for stock in collection]
    kwargs = dict(type='candle', figratio=(11, 8), figscale=0.85)
    style = mpf.make_mpf_style(base_mpf_style='yahoo', rc={'font.size':8, 'font.family': 'Source Han Sans'}, marketcolors=mc)
    fig, axes = mpf.plot(collection_rr_df, **kwargs, 
                         style=style, 
                         scale_padding={'left': 0.4, 'top': 1, 'right': 1, 'bottom': 1}, 
                         returnfig=True, 
                         ylabel='Return rate', 
                         addplot=apdict)
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))
    legend = axes[0].legend([stock.name for stock in collection], prop={'size': 7}, fancybox=True, borderaxespad=0.)
    legend.get_frame().set_alpha(0.4)
    fig.savefig(output, dpi=300)
    plt.close(fig)

# async utilities
async def data_collector_async(stock, client, time_begin='19900101', time_end='20991231'):
    stock_url = eastmoney_base.format(market=stock.market_id, 
                                      bench_code=stock.code, 
                                      time_begin=time_begin, 
                                      time_end=time_end)
    try:
        response = await client.request(method='GET', url=stock_url)
        response_json = await response.json()
        stock_data = pd.DataFrame(map(lambda x: x.split(','), response_json["data"]["klines"]))
    except HTTPError as e:
        raise QueryError(f"HTTP error: {e}") from e
    except TypeError as e:
        raise QueryError("Can't find kline data") from e
    stock_data.columns = ["date", "open", "close", "high", "low", "volume", "money", "change"]
    stock_data["date"] = pd.to_datetime(stock_data["date"])
    return stock_data

# @timing_async
async def mix_data_collector_async(stock_mix, time_begin='20210101', time_end='20991231', time_ref='latest'):
    """
    Collecting and postprocessing for Stock_mix, where only close price are collected
    Noted that long time range can cause date inconsistency
    """
    # using the same client instead of creating everytime may improve the performance
    async with aiohttp.ClientSession() as client: 
        # stock_data: list of pd.df
        stock_data = await asyncio.gather(*[data_collector_async(stock, client, time_begin=time_begin, time_end=time_end) for stock in stock_mix.stock_list])
    # Checking whether the dates are consistent, trading suspention handled
    try:
        # matrix_date: (n_stock, n_days) np.array, type: numpy.datetime64
        matrix_date = np.array([stock['date'].values for stock in stock_data])
        matrix_close_price = np.array([np.array(stock['close']) for stock in stock_data]).astype(float)
    except ValueError:
        # operands could not be broadcast together
        # print("Mix data can't broadcast")
        # a more robust solution #TODO can replace current flow
        collection_close_price = []
        for i, stock in enumerate(stock_data):
            stock_kline = stock.set_index('date')
            stock_kline.index = pd.to_datetime(stock_kline.index)
            stock_kline = stock_kline.astype(float)
            stock_kline[[stock.code for stock in stock_mix.stock_list][i]] = stock_kline['close']
            collection_close_price.append(stock_kline[[stock.code for stock in stock_mix.stock_list][i]])
        # dealing with trade suspention
        collection_close_price_df = reduce(lambda x, y: pd.merge(x, y, how='outer', on='date', sort=True), collection_close_price)
        collection_close_price_df = collection_close_price_df.fillna(method='ffill')
        matrix_date = [collection_close_price_df.index.values]
        matrix_close_price = np.transpose(collection_close_price_df.to_numpy())

    # only need close price here
    if time_ref == 'oldest':
        date_ref_index = 0
    elif time_ref == 'latest':
        date_ref_index = -1
    elif time_ref == 'created':
        date_created_stamp = pd.to_datetime(stock_mix.create_time.date())
        date_ref_index = np.where(matrix_date[0] == date_created_stamp)[0][0]
    else:
        raise ValueError
    close_price_ref = matrix_close_price[:, date_ref_index]
    stock_share_ratios = stock_mix.holding_ratio / close_price_ref
    value_mix = np.average(matrix_close_price, axis=0, weights=stock_share_ratios) 
    value_mix = value_mix / value_mix[date_ref_index] # norm to 1
    mix_data = stock_data[0].copy()
    mix_data = mix_data.drop(['money', 'change', 'volume'], axis=1)
    mix_data['close'] = value_mix
    # mix_data['volume'] = volume_mix
    # Data redundancy, rather inelegant here, might go PR on mplfinance (or simplily using plot instead)
    mix_data['low'] = mix_data['open'] = mix_data['high'] = np.zeros(len(stock_data[0]['date']))
    return mix_data, matrix_close_price # to be used in profit analysis

# generate Stock_mix
def gen_stock_mix(mix_code, mix_name, stock_names, holding_ratios, create_time):
    stock_list = []
    candidate_list = {}
    for stock_name in stock_names:
        query_result = stock_query(stock_name, filter_code=True, echo=True)
        if len(query_result) == 1:
            stock_list.append(query_result[0])
        else:
            candidate_list[stock_name] = [str(stock) for stock in query_result]
            # print("multiple query results on "+stock_name)
    if candidate_list:
        return candidate_list
    stock_mix = Stock_mix(code=mix_code, name=mix_name, stock_list=stock_list, 
                          holding_ratio=holding_ratios, create_time=create_time)
    stock_mix.save()
    # print(stock_mix)
    return stock_mix

async def main():
    # kline plot test
    # x = stock_query('000300', echo=True)[0].collect_data(time_begin='20210101')
    # plot_kline(x, title='test_kline', plot_type='candle', volume=True, macd=True)

    # Query test
    # _query_test(_test_stock_code_CN)
    # _query_test(_test_stock_HK)
    # Stock_mix test
    # enl_stock_name = [] # a list of query keywords
    # enl_stock_ratio = [1 / len(enl_stock_name)] * len(enl_stock_name)
    # enl_stock_mix = gen_stock_mix(mix_code='enltest', mix_name="enl001", 
    #                               stock_names=enl_stock_name, holding_ratios=enl_stock_ratio)

    # Stock_mix object Loading & Pie plot, return rate analysis test

    enl_stock_mix = stock_query('enl001')[0]
    # enl_stock_mix.draw()
    mix_data, matrix_close_price = mix_data_collector(enl_stock_mix)
    # datetime_yesterday = datetime.datetime.utcnow() - datetime.timedelta(days=1)
    # profit_ratio, stock_profit_ratio = enl_stock_mix.get_profit_ratio(mix_data, matrix_close_price, date_ref=datetime_yesterday)
    # print(profit_ratio)
    # print(stock_profit_ratio)
    # plot_kline(mix_data, title=enl_stock_mix.name, plot_type='line', volume=False)
    # matplotlib.rcParams['font.family'] = ['Source Han Sans']
    # plot_profitline(mix_data, profit_ratio)
    # plot_stock_profit(enl_stock_mix, stock_profit_ratio)

    # Async test
    mix_data_async, matrix_close_price_async = await mix_data_collector_async(enl_stock_mix)
    return mix_data_async, matrix_close_price_async 


if __name__ == '__main__':
    # loop = asyncio.get_event_loop()
    # mix_data_async, matrix_close_price_async = loop.run_until_complete(main())

    stock_list = ['000300', '秋田微', '贵州茅台', '火星人', '西大门']
    stock_list = [stock_query(keyword, echo=True)[0] for keyword in stock_list]
    plot_return_rate_anlys(stock_list, '20201001', output=os.path.join(_test_path, 'compare_percentage.jpg'))
