import aiohttp
import asyncio
import datetime
from functools import reduce
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

from style import mc

# VARIABLES
matplotlib.rcParams['font.family'] = ['Source Han Sans']
data_path = './data'
_test_path = './demo'
# fqt=1: split-adjusted price
eastmoney_base = "http://push2his.eastmoney.com/api/qt/stock/kline/get?secid={market}.{bench_code}&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58&klt=101&fqt=1&beg={time_begin}&end={time_end}"
eastmoney_base_live = "http://push2.eastmoney.com/api/qt/stock/trends2/get?fields1=f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13&fields2=f51,f52,f53,f54,f55,f56,f57,f58&ut=fa5fd1943c7b386f172d6893dbfba10b&iscr=0&ndays=1&secid={market}.{bench_code}"
headers = {'user-agent': "Mozilla/5.0 (X11; Linux x86_64; rv:92.0) Gecko/20100101 Firefox/92.0"}

# FILTERS
# "TypeUS" seems to be a strong parameter, but with uncertainty 
stock_market =  {'0': "SZ", '1': "SH", 
                 '105': "NASDAQ", '106': "NYSE", '107': "AMEX", '156': "US", '100': "US", '116': "HK"} # MktNum: MarketName
stock_type = {'1': "沪A", '25': "科创板", '2': "深A", '8': "基金", '5': "指数", 
              '19': "港股", '6': "港股"} # SecurityType: SecurityTypeName

market_group = {'A': ['0', '1'], 
                'H': ['116'],
                'HK': ['116'], 
                'US': ['105', '106', '107', '156', '100'],
                'ALL': list(stock_market.keys())}

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
    @property
    def company_info(self):
        """
        retrieve company_info from eastmoney (currently only the url is returned)
        """
        if (market := stock_market[self.market_id]) in ('SZ', 'SH'):
            h5_fc = self.code+{'SH': '01', 'SZ': '02'}[market]
            return f"https://emh5.eastmoney.com/html/?fc={h5_fc}&color=w"
        elif market in ('AMEX', 'NYSE', 'NASDAQ', 'HK'):
            h5_fc = self.code+{'HK': '', 'AMEX': '.A', 'NYSE': '.N', 'NASDAQ': '.O'}[market]
            h5_mkt = ({'HK': 'HK'} | dict.fromkeys(['AMEX', 'NYSE', 'NASDAQ'], 'US'))[market]
            return f"https://emh5.eastmoney.com/{h5_mkt}/index.html?fc={h5_fc}&color=w"
        else:
            return ''
            
    def collect_data_daily(self, time_begin='19900101', time_end='20991231'):
        stock_url = eastmoney_base.format(market=self.market_id, 
                                          bench_code=self.code, 
                                          time_begin=time_begin, 
                                          time_end=time_end)
        try:
            stock_data = pd.DataFrame(map(lambda x: x.split(','), 
                                      requests.get(stock_url, timeout=10, headers=headers).json()["data"]["klines"]))
        except TypeError as e:
            raise QueryError("Can't find kline data") from e
        stock_data.columns = ["date", "open", "close", "high", "low", "volume", "money", "change"]
        stock_data["date"] = pd.to_datetime(stock_data["date"])
        return stock_data

    def collect_data_live(self):
        stock_url = eastmoney_base_live.format(market=self.market_id, 
                                               bench_code=self.code)
        try:
            data_live = requests.get(stock_url, timeout=3, headers=headers).json()["data"]
            stock_preclose = data_live['preClose']
            stock_data = pd.DataFrame(map(lambda x: x.split(','), 
                                      data_live["trends"]))
        except TypeError as e:
            raise QueryError("Can't find kline data") from e
        stock_data.columns = ["date", "open", "close", "high", "low", "volume", "money", "change"] # follow mpf convention
        stock_data["date"] = pd.to_datetime(stock_data["date"])
        return stock_data, stock_preclose

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
        """
        # mix_data is actually reduntand, included here just for convinence. 
        # date_ref == 'latest': return profit_ratio based on the latest two days (dirty fix for /now
        """
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
        #TODO for created time, temporary fix
        profit_ratio[:mix_price_ref_idx] = 0.0

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

# DATA COLLECTION
def stock_query(keyword, filter_md5=None, filter_code=None, echo=False):
    """
    borrowed from https://github.com/pengnanxiaomeimei/stock_data_analysis/
    Not ideal but works.
    """
    try:
        # Using '@' to restrick the stock market
        mkt_to_search = market_group[keyword[keyword.index('@')+1:].upper()]
    except (KeyError, ValueError):
        mkt_to_search = market_group['ALL']
    else:
        keyword = keyword[:keyword.index('@')]
        
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

    r = requests.get(query_url, headers=headers)
    p2 = re.compile(r'[(](.*)[)]', re.S)
    result = re.findall(p2, r.content.decode('utf-8'))[0]
    try:
        mes_dict = eval(result)
    except NameError as e:
        raise QueryError(f"Can't find keyword {keyword}") from e
    query_result = mes_dict['QuotationCodeTable']['Data']
    stock_list = [Stock(code=x['Code'], name=x['Name'], market_id=x['MktNum'], type_id=x['SecurityType']) 
                  for x in query_result if x['MktNum'] in mkt_to_search and \
                                           (x['SecurityType'] in stock_type or x['Classify'] == "UsStock" or x['Classify'] == "UniversalIndex") and \
                                           x["SecurityTypeName"] != "曾用"]
    if filter_md5:
        stock_list = [stock for stock in stock_list if stock.md5 == filter_md5]
    if filter_code:
        # use exact match to ensure that one can always define a Stock_mix by codes
        if keyword in [stock.code for stock in stock_list]:
            stock_list = [stock for stock in stock_list if stock.code == keyword]
    if echo:
        print(stock_list)
    if not stock_list:
        raise QueryError(f"Empty stock_list from \n{query_result}")
    return stock_list

async def data_collector_async(stock, client, time_begin='19900101', time_end='20991231'):
    stock_url = eastmoney_base.format(market=stock.market_id, 
                                      bench_code=stock.code, 
                                      time_begin=time_begin, 
                                      time_end=time_end)
    try:
        response = await client.request(method='GET', url=stock_url)
        response_json = await response.json()
        stock_data = pd.DataFrame(map(lambda x: x.split(','), response_json["data"]["klines"]))
    except TypeError as e:
        raise QueryError("Can't find kline data") from e
    stock_data.columns = ["date", "open", "close", "high", "low", "volume", "money", "change"]
    stock_data["date"] = pd.to_datetime(stock_data["date"])
    return stock_data

async def mix_data_collector_async(stock_mix, time_begin='20210101', time_end='20991231', time_ref='latest'):
    """
    Collecting and postprocessing for Stock_mix, where only close price are collected
    Noted that long time range can cause date inconsistency
    """
    # using the same client instead of creating everytime may improve the performance
    async with aiohttp.ClientSession() as client: 
        # stock_data: list of pd.df
        stock_data = await asyncio.gather(*[data_collector_async(stock, client, time_begin=time_begin, time_end=time_end) for stock in stock_mix.stock_list])
    # Assuming that dates are inconsistent, trading suspention handled
    # a more robust solution
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
    dates_array = collection_close_price_df.index.values
    matrix_close_price = np.transpose(collection_close_price_df.to_numpy())

    # only need close price here
    if time_ref == 'oldest':
        date_ref_index = 0
    elif time_ref == 'latest':
        date_ref_index = -1
    elif time_ref == 'created':
        date_created_stamp = pd.to_datetime(stock_mix.create_time.date())
        for i in range(10):
            # range(10) is to match the buffer time for non-trading days
            # For Stock_mix created on non-trading days, 
            # we slide the dates_array by step of 1-day to find the nearest previous trading day
            try:
                date_ref_index = np.where((dates_array - np.timedelta64(i, 'D')) == date_created_stamp)[0][0]
                break
            except IndexError:
                pass
    else:
        raise ValueError
    close_price_ref = matrix_close_price[:, date_ref_index]
    stock_share_ratios = stock_mix.holding_ratio / close_price_ref
    value_mix = np.average(matrix_close_price, axis=0, weights=stock_share_ratios) 
    value_mix = value_mix / value_mix[date_ref_index] # norm to 1
    mix_data = pd.DataFrame(dates_array, columns=['date'])
    mix_data['close'] = value_mix
    # mix_data['volume'] = volume_mix
    # Data redundancy, rather inelegant here, might go PR on mplfinance (or simplily using plot instead)
    place_holder = np.empty(len(mix_data))
    place_holder[:] = np.nan
    mix_data['low'] = mix_data['open'] = mix_data['high'] = place_holder
    return mix_data, matrix_close_price # to be used in profit analysis

# PLOT
def plot_kline(stock_data, title='', plot_type='candle', volume=True, macd=False, live=False, preclose=None, output=os.path.join(_test_path, 'kline.jpg')):
    stock_kline = stock_data.set_index("date")
    stock_kline.index = pd.to_datetime(stock_kline.index)
    stock_kline = stock_kline.astype(float)
    if plot_type == 'line':
        ma_value = ()
    else:
        ma_value = (5, 10, 20)
    if macd:
        # Using MACD(12,26,9) here, https://en.wikipedia.org/wiki/MACD
        ema_12 = stock_data['close'].ewm(span=12, adjust=False).mean()
        ema_26 = stock_data['close'].ewm(span=26, adjust=False).mean()
        macd_line = ema_12 - ema_26
        macd_signal = macd_line.ewm(span=9, adjust=False).mean()
        macd_histogram = macd_line - macd_signal

        apds = [mpf.make_addplot(macd_histogram, type='bar', width=0.7, panel=1,
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
    if live:
        price_0 = preclose
        pct_axe = axes[0].secondary_yaxis('left', 
            functions=(lambda x: (x - price_0) / price_0, lambda x: price_0 * (1 + x)))
        pct_axe.set_ylabel("Percentage")
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

async def plot_return_rate_anlys_async(collection, date_begin, ref=None, excess_return=False, output=os.path.join(_test_path, 'compare.jpg')):
    """
    Perform return rate anaylsis on collection of stock or stock_mix, by plotting return rates in same axis. 
    """
    if len(collection_type := (set(map(type, collection)))) != 1:
        # collection containing both stock and stock_mix is not supported 
        return 1
    else:
        collection_type = list(collection_type)[0]
        collection_rr = []
    if collection_type is Stock:
        if ref is None:
            ref_idx = 0
        async with aiohttp.ClientSession() as client: 
            # stock_data: list of pd.df
            collection_data = await asyncio.gather(*[data_collector_async(stock, client, time_begin=date_begin) for stock in collection])
        for i, stock_kline in enumerate(collection_data):
            stock = collection[i]
            stock_kline = stock_kline.set_index('date')
            stock_kline.index = pd.to_datetime(stock_kline.index)
            stock_kline = stock_kline.astype(float)
            identifier = stock.name + '  ' + stock.code
            collection[i] = identifier
            stock_kline[identifier] = (stock_kline['close'] - stock_kline['close'][ref_idx]) / stock_kline['close'][ref_idx]
            collection_rr.append(stock_kline[identifier])
    elif collection_type is Stock_mix:
        # collection_data: (mix_data, mix_close_price)
        collection_data = await asyncio.gather(*[mix_data_collector_async(stock_mix, time_begin=(stock_mix.create_time - datetime.timedelta(days=9)).strftime("%Y%m%d")) for stock_mix in collection])
        for i, collection_data in enumerate(collection_data):
            stock_mix = collection[i]
            # price reference is already handled in Stock_mix.get_profit
            stock_data = collection_data[0]
            stock_close_price = collection_data[1]
            profit_ratio, _ = stock_mix.get_profit_ratio(stock_data, stock_close_price, 
                                                         date_ref=stock_mix.create_time)
            stock_profitline = stock_data.set_index("date")
            stock_profitline.index = pd.to_datetime(stock_profitline.index)
            stock_profitline = stock_profitline.astype(float)
            identifier = stock_mix.code + '  ' + stock_mix.name
            collection[i] = identifier
            stock_profitline[identifier] = profit_ratio
            collection_rr.append(stock_profitline[identifier])


    collection_rr_df = reduce(lambda x, y: pd.merge(x, y, how='outer', on='date', sort=True), collection_rr) # merge into single dataframe
    collection_rr_df = collection_rr_df.fillna(method='ffill').fillna(0.0) # second fillna for return rates before created
    print(collection_rr_df)

    # create a 'base layer' placeholder for plot
    place_holder = np.empty(collection_rr_df.shape[0])
    place_holder[:] = np.nan
    collection_rr_df['close'] = collection_rr_df['low'] = collection_rr_df['open'] = collection_rr_df['high'] = place_holder

    apdict = [mpf.make_addplot(collection_rr_df[identifier]) for identifier in collection]

    kwargs = dict(type='candle', figratio=(11, 8), figscale=0.85)
    style = mpf.make_mpf_style(base_mpf_style='yahoo', rc={'font.size':8, 'font.family': 'Source Han Sans'}, marketcolors=mc)
    fig, axes = mpf.plot(collection_rr_df, **kwargs, 
                         style=style, 
                         scale_padding={'left': 0.4, 'top': 1, 'right': 1, 'bottom': 1}, 
                         returnfig=True, 
                         ylabel='Return rate', 
                         addplot=apdict)
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))
    legend = axes[0].legend([identifier for identifier in collection], prop={'size': 7}, fancybox=True, borderaxespad=0.)
    legend.get_frame().set_alpha(0.4)
    fig.savefig(output, dpi=300)
    plt.close(fig)

# HELPER FUNCTIONS
def gen_stock_mix(mix_code, mix_name, stock_names, holding_ratios, create_time):
    """
    generate Stock_mix
    """
    stock_list = []
    candidate_list = {}
    for stock_name in stock_names:
        # To ignore indicies when creating a new portfolio.
        query_result = [stock for stock in stock_query(stock_name, filter_code=True, echo=True) if stock.type_id != '5']
        if len(query_result) == 1:
            stock_list.append(query_result[0])
        else:
            candidate_list[stock_name] = [str(stock) for stock in query_result]
    if candidate_list:
        return candidate_list
    stock_mix = Stock_mix(code=mix_code, name=mix_name, stock_list=stock_list, 
                          holding_ratio=holding_ratios, create_time=create_time)
    stock_mix.save()
    return stock_mix

def get_time_range(day_interval=120):
    """
    return ({day_interval} days ago, today + 1)
    """
    time_end = datetime.datetime.utcnow() + datetime.timedelta(days=1)
    time_begin = time_end - datetime.timedelta(days=day_interval)
    return (time_begin.strftime("%Y%m%d"), time_end.strftime("%Y%m%d"))
    
def price_to_percentage(x, price_0):
    return (x - price_0) / price_0

def percentage_to_price(x, price_0):
    return price_0 * (1 + x)


async def main():
    # kline plot test
    x = stock_query('000300', echo=True)[0].collect_data_daily(time_begin='20210101')
    plot_kline(x, title='test_kline', plot_type='hollow_candle', volume=True, macd=True)

    # Stock_mix test
    # enl_stock_name = [] # a list of query keywords
    # enl_stock_ratio = [1 / len(enl_stock_name)] * len(enl_stock_name)
    # enl_stock_mix = gen_stock_mix(mix_code='enltest', mix_name="enl001", 
    #                               stock_names=enl_stock_name, holding_ratios=enl_stock_ratio)

    # Stock_mix object Loading & Pie plot, return rate analysis test

    enl_stock_mix = stock_query('enl001')[0]
    # enl_stock_mix.draw()
    mix_data_async, matrix_close_price_async = await mix_data_collector_async(enl_stock_mix)
    # datetime_yesterday = datetime.datetime.utcnow() - datetime.timedelta(days=1)
    # profit_ratio, stock_profit_ratio = enl_stock_mix.get_profit_ratio(mix_data, matrix_close_price, date_ref=datetime_yesterday)
    # print(profit_ratio)
    # print(stock_profit_ratio)
    # plot_kline(mix_data, title=enl_stock_mix.name, plot_type='line', volume=False)
    # matplotlib.rcParams['font.family'] = ['Source Han Sans']
    # plot_profitline(mix_data, profit_ratio)
    # plot_stock_profit(enl_stock_mix, stock_profit_ratio)
    return mix_data_async, matrix_close_price_async 


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    mix_data_async, matrix_close_price_async = loop.run_until_complete(main())
