# create argparse from telegram bot commands

import argparse

def argparse_kline(message):
    """
    Usage: /kline [OPTION] STOCK_KEYWORD
    Do a query on STOCK_KEYWORD and plot kline based on the result

    Options:
        -d <days>   set the time range for kline plotting, unit: days
        -h          show help message
        -e <md5>    md5 match(md5), NOT FOR USERS
    Examples:
        /kline AMD
        /kline comb001 -d 365
    """
    command = message.split(' ')
    parser = argparse.ArgumentParser(add_help=False, exit_on_error=False)
    parser.add_argument('-h', dest='help', action='store_true')
    parser.add_argument('-d', dest='days')
    parser.add_argument('-e', dest='md5')
    parser.add_argument('keyword', nargs='*')
    args, _ = parser.parse_known_args(command[1:])
    return args

def argparse_define(message):
    """
    Usage: /define [OPTION] CODE NAME -l [STOCK_KEYWORD [STOCK_KEYWORD ...]]
    Define a new portfolio named NAME with CODE, by combining the given STOCK_KEYWORD, the default ratios of holding values are equal. 

    Options:
        -w [weight [weight ...]]    customize the weights of each stock, 
        -f                          force rewrite the existing portfolio
        -h                          show help message

    Examples:
        /define p001 AIN -l AMD intel NVDA
        /define p002 AIN -l AMD intel NVDA -w 1 2 3
    """
    command = message.split(' ')
    parser = argparse.ArgumentParser(add_help=False, exit_on_error=False)
    parser.add_argument('-f', dest='force', action='store_true')
    parser.add_argument('-h', dest='help', action='store_true')
    parser.add_argument('-l', dest='stock_list', nargs='*')
    parser.add_argument('-w', dest='weights', nargs='*')
    parser.add_argument('code_and_name', nargs='*')
    args, _ = parser.parse_known_args(command[1:])
    return args

def argparse_status(message):
    """
    Usage: /status [OPTION] PORTFOLIO_CODE
    Plot the return rate of PORTFOLIO_CODE since created

    Options:
        -d    detailed mode; plot the return rates of all stocks in the portfolio
        -h    show help message
        -l    list stocks in the portfolio
    """
    command = message.split(' ')
    parser = argparse.ArgumentParser(add_help=False, exit_on_error=False)
    parser.add_argument('-d', dest='detail', action='store_true')
    parser.add_argument('-l', dest='ls', action='store_true')
    parser.add_argument('-h', dest='help', action='store_true')
    parser.add_argument('stock_mix_code', nargs='?')
    args, _ = parser.parse_known_args(command[1:])
    return args

def argparse_now(message):
    """
    Usage: /now [OPTION] PORTFOLIO_CODE
    Plot today's return rate for a portfolio (currently valid for trading days only)

    Options:
        -h    display help message
    """
    #TODO Get real-time chart for STOCK_KEYWORD, /now [OPTION] STOCK_KEYWORD
    command = message.split(' ')
    parser = argparse.ArgumentParser(add_help=False, exit_on_error=False)
    parser.add_argument('-h', dest='help', action='store_true')
    parser.add_argument('stock', nargs='?')
    args, _ = parser.parse_known_args(command[1:])
    return args

def argparse_compare(message):
    """
    Usage: /compare [STOCK_KEYWORD [STOCK_KEYWORD ...]]

    Options:
        -h    display help message
    """
    command = message.split(' ')
    parser = argparse.ArgumentParser(add_help=False, exit_on_error=False)
    parser.add_argument('-h', dest='help', action='store_true')
    parser.add_argument('stocks', nargs='*')
    parser.add_argument('-d', dest='days', const=120, type=int)
    args, _ = parser.parse_known_args(command[1:])
    return args
