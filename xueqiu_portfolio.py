'''
XPT - Xueqiu Portfolio tools

This provides some useful functions to retreive information of Xueqiu Portfolios(组合).

Some functions are still under development, and they are commented out to aviod unnecessary dependency.
'''
import datetime
from textwrap import dedent
# import toml

from bs4 import BeautifulSoup
import requests
# import numpy as np
import pandas as pd

# from chinese_calendar import is_holiday

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 11_2_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36'}

# Some parameters
ignore_cash = True

""" class Assets(object): # To be finalized with the Porfolio class
    def __init__(self, code, name, market_id):
        self.code = code
        self.name = name
        self.market_id = market_id
    

class Porfolio(object): # Pending further research and design
    pass
 """

class XQpf:
    def __init__(self, code, name):
        self.code = code
        self.name = name
        
    def update(self):
        update_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return_rates = get_return_rate(self.code)
        update = f'''
            {self.name}
            更新时间：{update_time}
            最新净值：{return_rates['nv']}
            最新涨幅：{return_rates['dr']}
            累计涨幅：{return_rates['r']}
            组合链接：https://xueqiu.com/p/{self.code}
            '''
        return dedent(update)
    
    def asset_list(self):
        print(get_asset_list(self.code))
        

def connectxq(xqcode):
    if not xqcode.startswith('ZH'):
        raise ValueError('Porfolio code should start with ZH.')
    
    else:
        webpage = requests.get(f'https://xueqiu.com/p/{xqcode}', headers=headers)
        if webpage.status_code!=200:
            raise ValueError('Cannot find the porfolio, please check your code and web connection.')
        else:
            xqsoup = BeautifulSoup(webpage.text, 'html.parser')
    return xqsoup

def get_return_rate(xqcode):
    xqsoup = connectxq(xqcode)
    return_rate_block = xqsoup.find(class_='cube-profit-year fn-clear').parent
    keys = ['r', 'dr', 'mr', 'nv']
    values = [rate.string for rate in return_rate_block(class_='per')][0:4]
    values[0] = values[0] + '%'
    return dict(zip(keys, values))

def get_asset_list(xqcode):
    xqsoup = connectxq(xqcode)
    asset_list_block = xqsoup.find(class_='weight-list')
    df = pd.DataFrame([asset(string=True) for asset in asset_list_block('a')], columns=['Stock', 'a', 'Weight'])
    df.insert(1, 'Code', df['a'].str[2:])
    del df['a']
    cash_position = asset_list_block.contents[-1](string=True)
    if cash_position[0] == '现金' and not ignore_cash:
        cash_position.insert(1, 'Cash')
        cashdf = pd.DataFrame(cash_position).T
        cashdf.columns = df.columns
        df = pd.concat([df, cashdf], ignore_index=True)
    else:
        pass

    df.index = range(1, len(df) + 1)
    return df

    
def get_pfdata(xqcode):
    xqsoup = connectxq(xqcode)
    
    name = xqsoup.head.title.string.split(' - ')[0]
    code = xqcode
    return {'code': code, 'name': name}

""" def is_trading_day(date):
    if is_holiday(date) or date.isoweekday() in [6, 7]:
        return False
    else:
        return True """
            
