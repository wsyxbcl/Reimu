import asyncio
import unittest
from unittest.mock import AsyncMock, patch

import aiohttp
import requests

from utils import Stock, QueryError, data_collector_async


class FetchFailureTests(unittest.TestCase):
    def test_collect_data_daily_handles_requests_errors(self):
        stock = Stock(code="000300", name="CSI300", market_id="1", type_id="1")
        with patch("utils.requests.get") as mock_get:
            mock_get.side_effect = requests.exceptions.ConnectionError("boom")
            with self.assertRaises(QueryError):
                stock.collect_data_daily("20240101", "20241231")

    def test_data_collector_async_handles_aiohttp_errors(self):
        stock = Stock(code="000300", name="CSI300", market_id="1", type_id="1")
        mock_client = AsyncMock()
        mock_client.request.side_effect = aiohttp.ClientConnectionError("boom")

        async def run():
            with self.assertRaises(QueryError):
                await data_collector_async(stock, mock_client, "20240101", "20241231")

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
