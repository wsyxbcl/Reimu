import datetime
import hashlib
import io
import logging
import re
import toml

from aiogram import Bot, Dispatcher, executor, types
from aiogram.types import InlineQuery, \
    InputTextMessageContent, InlineQueryResultPhoto

from utils import *

config = toml.load("reimu.toml")

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize bot and dispatcher
bot = Bot(token=config["telegram"]["token"])
dp = Dispatcher(bot)

def get_time_range(days_interval=100):
    """
    return ({days_interval} days ago, today + 1)
    """
    time_end = datetime.datetime.utcnow() + datetime.timedelta(days=1)
    time_begin = time_end - datetime.timedelta(days=days_interval)
    return (time_begin.strftime("%Y%m%d"), time_end.strftime("%Y%m%d"))

@dp.message_handler(commands=['start', 'help'])
async def send_welcome(message):
    await message.reply("Busy in stock marketing, no time for talk")

@dp.message_handler()
async def kline(message):
    code = message.text.split()[0]
    try:
        time_range = get_time_range(int(message.text.split()[1]))
    except IndexError:
        time_range = get_time_range()
    buf = io.BytesIO()
    plot_kline(stock_data=data_collector(code, time_range[0], time_range[1]), output=buf)
    buf.seek(0)
    await message.reply_photo(buf, caption=code)

#TODO inline mode to be refined
@dp.inline_handler()
async def inline_echo(inline_query):
    pass

if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
