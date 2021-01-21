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

@dp.message_handler(commands=['kline'])
async def kline(message):
    stock_data = stock_query(message.text.split()[1])
    if len(stock_data) == 1:
        code = stock_data[0]["Code"]
        name = stock_data[0]["Name"]
    else:
        stock_list = '\n'.join([stock["Code"]+'\t'+stock["Name"] for stock in stock_data])
        await message.reply(f"Find {len(stock_data)} results:\n"+stock_list) 
    try:
        time_range = get_time_range(int(message.text.split()[2]))
    except IndexError:
        time_range = get_time_range()
    buf = io.BytesIO()
    plot_kline(stock_data=data_collector(code, time_range[0], time_range[1]), 
               title=f'kline of {code} {name}',
               output=buf)
    buf.seek(0)
    await message.reply_photo(buf, caption=code)

#TODO inline mode to be developed

# @dp.inline_handler()
# async def inline_echo(inline_query: types.InlineQuery):
#     text = inline_query.query or "echo"
#     results = []
# 
#     code = inline_query.query
# 
#     results.append(
#         types.InlineQueryResultPhoto(
#             id=1, photo_url=mainURL, title=text, thumb_url=mainURL
#         )
#     )
# 
#     await bot.answer_inline_query(inline_query.id, results=results, cache_time=1)

if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
