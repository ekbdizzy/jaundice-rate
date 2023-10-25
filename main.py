import json
import functools
import logging

import aiohttp
from aiohttp import web
import anyio
import pymorphy2

from settings import MAX_URLS_IN_QUERY
from text_tools import process_article, parse_charged_dicts


async def index(morph, charged_words, request):

    urls = request.query.get("urls", "").split(",")

    if len(urls) > MAX_URLS_IN_QUERY:
        return web.json_response(
            {"error": f"Too many urls in request, should be {MAX_URLS_IN_QUERY} or less"},
            status=400
        )

    articles_scoring = []
    async with aiohttp.ClientSession() as session:
        async with anyio.create_task_group() as tg:
            for url in urls:
                tg.start_soon(process_article, session, morph, charged_words, url, articles_scoring)

    return web.json_response({"urls": articles_scoring},
                             dumps=functools.partial(json.dumps, ensure_ascii=False))


if __name__ == '__main__':

    logger = logging.getLogger("root")
    logging.basicConfig(level=logging.DEBUG)

    morph = pymorphy2.MorphAnalyzer()
    charged_words = parse_charged_dicts('charged_dict')

    app = web.Application()
    app.add_routes([
        web.get('/', functools.partial(index, morph, charged_words)),
    ])

    web.run_app(app)
