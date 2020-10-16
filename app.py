import logging

from spokestack.activation_timeout import ActivationTimeout
from spokestack.asr.speech_recognizer import CloudSpeechRecognizer
from spokestack.io.pyaudio import PyAudioInput, PyAudioOutput
from spokestack.nlu.tflite import TFLiteNLU
from spokestack.pipeline import SpeechPipeline
from spokestack.tts.clients.spokestack import TextToSpeechClient
from spokestack.tts.manager import TextToSpeechManager
from spokestack.vad.webrtc import VoiceActivityDetector
from spokestack.wakeword.tflite import WakewordTrigger

from config import KEY_ID, KEY_SECRET
from dialogue_manager import dialogue_manager


def main():
    pipeline = SpeechPipeline(
        PyAudioInput(frame_width=20, sample_rate=16000, exception_on_overflow=False),
        [
            VoiceActivityDetector(vad_fall_delay=20),
            WakewordTrigger(pre_emphasis=0.97, model_dir="tflite"),
            CloudSpeechRecognizer(spokestack_id=KEY_ID, spokestack_secret=KEY_SECRET),
            ActivationTimeout(),
        ],
    )

    nlu = TFLiteNLU("tflite")
    manager = TextToSpeechManager(
        TextToSpeechClient(KEY_ID, KEY_SECRET), PyAudioOutput(),
    )

    @pipeline.event
    def on_activate(context):
        print("active")

    @pipeline.event
    def on_recognize(context):
        pipeline.pause()
        results = nlu(context.transcript)
        summary = dialogue_manager(results)
        manager.synthesize(summary, "text", "demo-male")
        pipeline.resume()

    manager.synthesize(
        "Welcome to wiki question answering, ask me about anything", "text", "demo-male"
    )
    pipeline.start()
    pipeline.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
