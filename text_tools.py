import asyncio
import logging
import os
import string
import time
from async_timeout import timeout
from contextlib import asynccontextmanager
from enum import Enum

import aiohttp
import pytest
import pymorphy2

from adapters import ArticleNotFound
from adapters.inosmi_ru import sanitize
from settings import FETCH_TIMEOUT, MORPH_TIMEOUT

pytest_plugins = ('pytest_asyncio',)

logger = logging.getLogger('root')


def _clean_word(word):
    word = word.replace('«', '').replace('»', '').replace('…', '')
    # FIXME какие еще знаки пунктуации часто встречаются ?
    word = word.strip(string.punctuation)
    return word


async def split_by_words(morph, text):
    """Учитывает знаки пунктуации, регистр и словоформы, выкидывает предлоги."""
    words = []
    for word in text.split():
        await asyncio.sleep(0)
        cleaned_word = _clean_word(word)
        normalized_word = morph.parse(cleaned_word)[0].normal_form
        if len(normalized_word) > 2 or normalized_word == 'не':
            words.append(normalized_word)
    return words


@pytest.mark.asyncio
async def test_split_by_words():
    # Экземпляры MorphAnalyzer занимают 10-15Мб RAM т.к. загружают в память много данных
    # Старайтесь организовать свой код так, чтоб создавать экземпляр MorphAnalyzer заранее и в единственном числе
    morph = pymorphy2.MorphAnalyzer()

    assert await split_by_words(morph, 'Во-первых, он хочет, чтобы') == ['во-первых', 'хотеть', 'чтобы']

    assert await split_by_words(morph, '«Удивительно, но это стало началом!»') == ['удивительно', 'это', 'стать',
                                                                                   'начало']


def calculate_jaundice_rate(article_words, charged_words):
    """Расчитывает желтушность текста, принимает список "заряженных" слов и ищет их внутри article_words."""

    if not article_words:
        return 0.0

    found_charged_words = [word for word in article_words if word in set(charged_words)]

    score = len(found_charged_words) / len(article_words) * 100

    return round(score, 2)


def test_calculate_jaundice_rate():
    assert -0.01 < calculate_jaundice_rate([], []) < 0.01
    assert 33.0 < calculate_jaundice_rate(['все', 'аутсайдер', 'побег'], ['аутсайдер', 'банкротство']) < 34.0


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


async def fetch(session, url):
    async with session.get(url) as response:
        response.raise_for_status()
        return await response.text()


def parse_charged_dicts(dicts_dir: str) -> list[str]:
    """Read all files in dicts_dir and return common list of words."""
    total_words = []
    for folder, _, file_paths in os.walk(dicts_dir):
        for file_path in file_paths:
            with open(os.path.join(folder, file_path)) as file:
                words = [w.strip() for w in file.readlines()]
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
