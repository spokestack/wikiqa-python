"""
Simple QA dialogue manager
"""
import re

import wikipedia  # type: ignore


MATCH = r"\(.*\)[,\s]*"


def dialogue_manager(results) -> str:
    summary = "I do not know what that is."
    if results.slots:
        entity = results.slots.get("entity").get("raw_value")
        if entity:
            try:
                summary = wikipedia.summary(entity, 1)
                summary = re.sub(MATCH, "", summary)
            except:
                entity = wikipedia.search(entity)[0]
                summary = wikipedia.summary(entity, 1, auto_suggest=False)
                summary = re.sub(MATCH, "", summary)

    return summary
