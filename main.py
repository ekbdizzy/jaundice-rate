import asyncio
import time
from enum import Enum
import os
import logging
from contextlib import asynccontextmanager

import aiofiles
import aiohttp
import anyio
from async_timeout import timeout
import pymorphy2

from adapters.inosmi_ru import sanitize
from adapters.exceptions import ArticleNotFound
from text_tools import split_by_words, calculate_jaundice_rate

FETCH_TIMEOUT = 5
MORPH_TIMEOUT = 5

TEST_ARTICLES = [
    "https://inosmi.ru/not/exist.html",
    "https://lenta.ru/brief/2021/08/26/afg_terror/",
    "https://inosmi.ru/20231021/neandertaltsy-266231832.html",
    "https://inosmi.ru/20231014/solntse-266120679.html",
    "https://inosmi.ru/20231009/obezyana-266007731.html",
    "https://inosmi.ru/20231009/lunatizm-265977298.html",
    "https://inosmi.ru/20231024/seks-266274274.html",
]


@asynccontextmanager
async def log_timing(*args, **kwargs):
    now = time.monotonic()
    try:
        yield
    finally:
        logger.info(f"Анализ закончен за {time.monotonic() - now} сек")



class ProcessingStatus(Enum):
    OK = 'OK'
    FETCH_ERROR = 'FETCH_ERROR'
    PARSING_ERROR = 'PARSING_ERROR'
    TIMEOUT = "TIMEOUT"


async def parse_charged_dicts(dicts_dir: str) -> list[str]:
    """Read all files in dicts_dir and return common list of words."""
    total_words = []
    for folder, _, file_paths in os.walk(dicts_dir):
        for file_path in file_paths:
            async with aiofiles.open(os.path.join(folder, file_path)) as file:
                words = [w.strip() for w in await file.readlines()]
                total_words.extend(words)
            return total_words


async def process_article(session: aiohttp.ClientSession,
                          morph: pymorphy2.MorphAnalyzer,
                          charged_words: list[str],
                          url: str,
                          results: list[dict]):
    try:
        async with timeout(FETCH_TIMEOUT):
            html = await fetch(session, url)

        async with timeout(MORPH_TIMEOUT):
            async with log_timing():
                splitted_text = await split_by_words(morph, sanitize(html))

        score = calculate_jaundice_rate(splitted_text, charged_words)

        results.append({
            'URL': url,
            'Рейтинг': score,
            'Статус': ProcessingStatus.OK.value,
            'Слов в статье': len(splitted_text)
        })
    except aiohttp.ClientResponseError:
        results.append({
            'URL': url,
            'Рейтинг': None,
            'Статус': ProcessingStatus.FETCH_ERROR.value,
            'Слов в статье': None
        })
    except ArticleNotFound:
        results.append({
            'URL': url,
            'Рейтинг': None,
            'Статус': ProcessingStatus.PARSING_ERROR.value,
            'Слов в статье': None
        })
    except asyncio.TimeoutError:
        results.append({
            'URL': url,
            'Рейтинг': None,
            'Статус': ProcessingStatus.TIMEOUT.value,
            'Слов в статье': None
        })


async def fetch(session, url):
    async with session.get(url) as response:
        response.raise_for_status()
        return await response.text()


async def main():
    articles_scoring = []
    charged_words = await parse_charged_dicts('charged_dict')
    async with aiohttp.ClientSession() as session:
        async with anyio.create_task_group() as tg:
            for url in TEST_ARTICLES:
                tg.start_soon(process_article, session, morph, charged_words, url, articles_scoring)
    print(articles_scoring)


if __name__ == '__main__':

    logger = logging.getLogger("root")
    logging.basicConfig(level=logging.DEBUG)

    morph = pymorphy2.MorphAnalyzer()
    asyncio.run(main())
