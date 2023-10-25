import asyncio
import os

import aiofiles
import aiohttp
import anyio
import pymorphy2

from adapters.inosmi_ru import sanitize
from text_tools import split_by_words, calculate_jaundice_rate

TEST_ARTICLES = [
    "https://inosmi.ru/20231021/neandertaltsy-266231832.html",
    "https://inosmi.ru/20231014/solntse-266120679.html",
    "https://inosmi.ru/20231009/obezyana-266007731.html",
    "https://inosmi.ru/20231009/lunatizm-265977298.html",
    "https://inosmi.ru/20231024/seks-266274274.html",
]


async def parse_charged_dicts(dicts_dir: str) -> list[str]:
    """Read all files in dicts_dir and return common list of words."""
    total_words = []
    for folder, _, file_paths in os.walk(dicts_dir):
        for file_path in file_paths:
            async with aiofiles.open(os.path.join(folder, file_path)) as file:
                words = [w.strip() for w in await file.readlines()]
                total_words.extend(words)
            return total_words


async def process_article(session, morph, charged_words, url):
    html = await fetch(session, url)
    splitted_text = split_by_words(morph, sanitize(html))
    score = calculate_jaundice_rate(splitted_text, charged_words)
    print('URL:', url)
    print('Рейтинг:', score)
    print('Слов в статье:', len(splitted_text))


async def fetch(session, url):
    async with session.get(url) as response:
        response.raise_for_status()
        return await response.text()


async def main():
    charged_words = await parse_charged_dicts('charged_dict')
    async with aiohttp.ClientSession() as session:
        async with anyio.create_task_group() as tg:
            for url in TEST_ARTICLES:
                tg.start_soon(process_article, session, morph, charged_words, url)


if __name__ == '__main__':
    morph = pymorphy2.MorphAnalyzer()
    asyncio.run(main())
