"""
Main app
"""
from spokestack.activation_timeout import ActivationTimeout  # type: ignore
from spokestack.asr.google.speech_recognizer import (  # type: ignore
    GoogleSpeechRecognizer,
)
from spokestack.io.pyaudio import PyAudioInput, PyAudioOutput  # type: ignore
from spokestack.pipeline import SpeechPipeline  # type: ignore
from spokestack.tts.clients.spokestack import TextToSpeechClient  # type: ignore
from spokestack.tts.manager import TextToSpeechManager  # type: ignore
from spokestack.vad.webrtc import VoiceActivityDetector  # type: ignore
from spokestack.wakeword.tflite import WakewordTrigger  # type: ignore

from config import KEY_ID, KEY_SECRET
from dialogue_manager import DialogueManager


def main():
    pipeline = SpeechPipeline(
        PyAudioInput(frame_width=20, sample_rate=16000, exception_on_overflow=False),
        [
            VoiceActivityDetector(),
            WakewordTrigger(pre_emphasis=0.97, model_dir="tflite"),
            GoogleSpeechRecognizer("../spokestack-python/examples/google_asr.json"),
            ActivationTimeout(),
        ],
    )

    dialogue_manager = DialogueManager(
        "tflite", "distilbert-base-cased-distilled-squad"
    )
    manager = TextToSpeechManager(
        TextToSpeechClient(KEY_ID, KEY_SECRET),
        PyAudioOutput(),
    )

    @pipeline.event
    def on_activate(context):
        print(context.is_active)

    @pipeline.event
    def on_recognize(context):
        pipeline.pause()
        answer = dialogue_manager(context.transcript)
        manager.synthesize(answer, "text", "demo-male")
        pipeline.resume()

    @pipeline.event
    def on_deactivate(context):
        print(context.is_active)

    manager.synthesize(dialogue_manager.greet(), "text", "demo-male")
    pipeline.start()
    pipeline.run()


if __name__ == "__main__":
    main()
