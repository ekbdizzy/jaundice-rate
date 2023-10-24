import aiohttp
import asyncio
import pymorphy2
import os
from adapters.inosmi_ru import sanitize
from text_tools import split_by_words, calculate_jaundice_rate
import aiofiles


async def parse_charged_dicts(dicts_dir: str) -> list[str]:
    """Read all files in dicts_dir and return common list of words."""
    total_words = []
    for folder, _, file_paths in os.walk(dicts_dir):
        for file_path in file_paths:
            async with aiofiles.open(os.path.join(folder, file_path)) as file:
                words = [w.strip() for w in await file.readlines()]
                total_words.extend(words)
            return total_words


async def fetch(session, url):
    async with session.get(url) as response:
        response.raise_for_status()
        return await response.text()


async def main():
    async with aiohttp.ClientSession() as session:
        html = await fetch(session, 'https://inosmi.ru/economic/20190629/245384784.html')
        splitted_text = split_by_words(morph, sanitize(html))
        words = await parse_charged_dicts('charged_dict')
        rate = calculate_jaundice_rate(splitted_text, words)
        print(rate, len(splitted_text))


if __name__ == '__main__':
    morph = pymorphy2.MorphAnalyzer()
    asyncio.run(main())
