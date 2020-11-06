"""
Simple QA dialogue manager
"""

import tensorflow as tf  # type: ignore
from mediawiki import MediaWiki  # type: ignore
from spokestack.nlu.result import Result
from spokestack.nlu.tflite import TFLiteNLU
from transformers import AutoTokenizer, TFAutoModelForQuestionAnswering  # type: ignore


class DialogueManager:
    """ Simple Question Answering Dialogue Manager """

    def __init__(self, log_path: str, base_model: str) -> None:
        self._wiki = MediaWiki()
        self._entity_recognizer = TFLiteNLU(log_path)
        self._tokenizer = AutoTokenizer.from_pretrained(base_model)
        self._answerer = TFAutoModelForQuestionAnswering.from_pretrained(base_model)

    def __call__(self, utterance: str) -> str:
        result = self._entity_recognizer(utterance)
        if result.intent == "ask.question":
            return self._answer(result)
        elif result.intent == "greet":
            return self.greet()
        elif result.intent == "command.exit":
            return self.exit()
        elif result.intent == "request.help":
            return self.help()
        else:
            return self.fallback()

    def _answer(self, result: Result) -> str:
        if result.slots:
            # get the tagged entity for page search
            entity = result.slots.get("entity").get("raw_value")
            # perform the search to find the wikipedia page
            entity = self._wiki.search(entity)[0]
            # get the page content to feed as context to the qa model
            passage = self._wiki.page(entity, auto_suggest=False).content
            # prepare qa model inputs
            inputs = self._tokenizer(
                result.utterance,
                passage,
                return_tensors="tf",
                padding=True,
                truncation=True,
            )
            # compute answer span
            start_scores, end_scores = self._answerer(inputs)
            start, end = tf.argmax(start_scores, -1)[0], tf.argmax(end_scores, -1)[0]
            # prepare the passage ids for slicing
            tokens = self._tokenizer.convert_ids_to_tokens(
                (inputs["input_ids"].numpy()[0])
            )
            # retrieve only the answer from the passage
            answer = self._tokenizer.convert_tokens_to_string(tokens[start : end + 1])
            return answer
        return "I don't have an answer for that"

    @staticmethod
    def greet() -> str:
        return "Hello, Ask me anything"

    @staticmethod
    def exit() -> str:
        return "Goodbye"

    @staticmethod
    def fallback() -> str:
        return (
            "I'm having trouble understanding your request, could you please "
            "repeat it"
        )

    @staticmethod
    def help() -> str:
        return "Ask a question like, how long is the amazon river?"
