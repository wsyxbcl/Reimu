# create argparse from telegram bot commands

import argparse


def argparse_kline(message):
    """
    Usage: /kline [OPTION] STOCK_KEYWORD
    Do query on STOCK_KEYWORD and plot kline based on result

    Options:
        -d <days>   set the time range for kline plot, unit: days
        -h          show help message
        -e          for exact match(md5), not for users, DO NOT USE IT
    Examples:
        /kline AMD
        /kline comb001 -d 365
    """
    command = message.split(' ')
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-h', dest='help', action='store_true')
    parser.add_argument('-d', dest='days')
    parser.add_argument('-e', dest='exact', action='store_true')
    parser.add_argument('keyword', nargs='?')
    return parser.parse_args(command[1:])

def argparse_define(message):
    """
    Usage: /define [OPTION] CODE NAME [STOCK_KEYWORD [STOCK_KEYWORD ...]]
    Define a new stock combination named NAME with CODE, by combining given STOCK_KEYWORD, the default ratio of holding values are equal. 

    Options:
        -w [weight [weight ...]]    customize weights of each stock, 
        -f                          force rewrite existed stock combination
        -h                          show help message

    Examples:
        /define comb001 AMD_INTC_NVDA(EQ) AMD intel NVIDIA
        /define comb002 AMD_INTC_NVDA(CM) AMD intel NVIDIA -w 1 2 3
    """
    command = message.split(' ')
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-f', dest='force', action='store_true')
    parser.add_argument('-h', dest='help', action='store_true')
    parser.add_argument('-l', dest='stock_list', nargs='*')
    parser.add_argument('-w', dest='weights', nargs='*')
    parser.add_argument('code_and_name', nargs=2)
    return parser.parse_args(command[1:])

def argparse_status(message):
    """
    Usage: /status [OPTION] COMBINATION_CODE
    Plot return rate of COMBINATION_CODE since created

    Options:
        -d    detailed mode, plot return rates of all stocks in combination
        -h    show help message
    """
    command = message.split(' ')
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-d', dest='detail', action='store_true')
    parser.add_argument('-h', dest='help', action='store_true')
    parser.add_argument('stock_mix_code', nargs='?')
    return parser.parse_args(command[1:])

def argparse_today(message):
    """
    Usage: /today [OPTION] STOCK_KEYWORD
       or: /today [OPTION] COMBINATION_CODE
    Get real-time chart for STOCK_KEYWORD or today's return rate for a combination

    Options:
        -h    display help message
    """
    command = message.split(' ')
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-h', dest='help', action='store_true')
    parser.add_argument('stock', nargs='?')
    return parser.parse_args(command[1:])

