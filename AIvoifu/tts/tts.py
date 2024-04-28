import os
from typing import Dict, Literal
import soundfile as sf

# write your own tts class and place it in this folder
# model weight will be downloaded and save at tts_base_model
# class khanomtal11: deprecated (now was used as an example)
#     def __init__(self) -> None:
#         from TTS.api import TTS

#         root = os.path.dirname(os.path.abspath(__file__))
#         # change config.json to match your model
#         json_path = os.path.join(root, "config.json")
#         configs = json.load(open(json_path, "r"))
#         configs["speakers_file"] = os.path.join(root, "speakers.pth")
#         configs["language_ids_file"] = os.path.join(root, "language_ids.json")
#         configs["model_args"]["speakers_file"] = os.path.join(root, "speakers.pth")
#         configs["model_args"]["speaker_ids_file"] = os.path.join(
#             root, "speaker_ids.json"
#         )
#         configs["model_args"]["speaker_encoder_config_path"] = os.path.join(
#             root, "config_se.json"
#         )
#         configs["model_args"]["speaker_encoder_model_path"] = os.path.join(
#             root, "model_se.pth"
#         )
#         self.sr = configs["audio"]["sample_rate"]
#         # save the new config
#         json.dump(configs, open(json_path.replace("config", "nconfig"), "w"), indent=4)

#         self.model = TTS(
#             model_path=os.path.join(root, "best_model.pth"),
#             config_path=os.path.join(root, "nconfig.json"),
#             progress_bar=False,
#             gpu=False,
#         )
#         self.model.model_name = "khanomtal11"

#     def tts(self, text, out_path, speaker=0, language=3):
#         self.model.tts_to_file(
#             text=text,
#             file_path=out_path,
#             speaker=self.model.speakers[speaker],
#             language=self.model.languages[language],
#         )

#     def supported_languages(self) -> list:
#         return ['tha_Thai']


class BaseTTS:
    def __init__(self) -> None:
        self.model_name = "base_tts"
        self.sr = 22050
        self.model = None

    def tts(self, text, out_path):
        raise NotImplementedError(
            f"Please Implement all necessery method in '{self.model_name}' model"
        )

    def supported_languages(self) -> list:
        raise NotImplementedError(
            f"Please Implement all necessery method in '{self.model_name}' model"
        )

    def requested_additional_args(self, **kwargs) -> None:
        # this function will be called when user select this model
        # you can request additional args here by overriding this function, please handle it in the class atritube
        return None


class Gtts(BaseTTS):
    """[Google Text-to-Speech(gTTs)](https://github.com/pndurette/gTTS)\n
    is a Python library and CLI tool to interface with Google Translate's text-to-speech API.\n
    - Allow Language Selection
    """

    def __init__(self, language: str = None) -> None:
        from gtts import gTTS

        # disable gtts debug
        import logging

        logging.getLogger("gtts").setLevel(logging.CRITICAL)

        self.model_name = "gtts"
        self.sr = 16000
        self.model = gTTS
        self.language = None

    def tts(self, text, out_path, language="en"):
        language = self.language if self.language is not None else language
        tts = self.model(text=text, lang=language)
        tts.save(out_path)

    def supported_languages(self) -> list:
        return [
            "af",
            "ar",
            "bg",
            "bn",
            "bs",
            "ca",
            "cs",
            "da",
            "de",
            "el",
            "en",
            "es",
            "et",
            "fi",
            "fr",
            "gu",
            "hi",
            "hr",
            "hu",
            "id",
            "is",
            "it",
            "iw",
            "ja",
            "jw",
            "km",
            "kn",
            "ko",
            "la",
            "lv",
            "ml",
            "mr",
            "ms",
            "my",
            "ne",
            "nl",
            "no",
            "pl",
            "pt",
            "ro",
            "ru",
            "si",
            "sk",
            "sq",
            "sr",
            "su",
            "sv",
            "sw",
            "ta",
            "te",
            "th",
            "tl",
            "tr",
            "uk",
            "ur",
            "vi",
            "zh-CN",
            "zh-TW",
            "zh",
        ]

    def requested_additional_args(self, **kwargs) -> None:
        # this function will be called when user select this model
        # you can request additional args here, please handle it in the class atritube
        # example:
        self.language = kwargs.get("language", None)
        if self.language is None:
            raise ValueError("Language is not specified")
        if self.language not in self.supported_languages():
            raise ValueError(
                f"Invalid language selection: {self.language} please select from"
                + "\n".join(self.supported_languages())
            )
        return None


class EdgeTTS(BaseTTS):
    """[Edge TTS](https://github.com/rany2/edge-tts)\n
    is a TTS model that run on microsoft edge TTS services.\n
    utilize asyncio to run the tts function under the hood.\n
    - Allow voice (speaker) selection
    """

    def __init__(self) -> None:
        self.model_name = "edge_tts"

    async def __tts_async(self, text, out_path, voice="en-US-RogerNeural"):
        import edge_tts

        voice = self.speaker if self.speaker is not None else voice
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(out_path)

    def __run_in_thread(self, text, out_path, voice):
        import asyncio

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.__tts_async(text, out_path, voice))
        loop.close()

    def supported_languages(self) -> list:
        return ["en"]

    def tts(self, text, out_path, voice="en-US-RogerNeural"):
        import asyncio
        import threading

        voice = self.speaker if self.speaker is not None else voice
        # Check if there's an event loop running that can be used
        t = threading.Thread(target=self.__run_in_thread, args=(text, out_path, voice))
        t.start()
        t.join()  # Wait for the thread to complete

    def requested_additional_args(self, **kwargs) -> None:
        self.speaker = kwargs.get("speaker", None)
        return None


class Bark(BaseTTS):
    """[Bark](https://github.com/suno-ai/bark)
    is a TTS model that run on Bark TTS Model.\n
    - [Allow voice (speaker) selection](https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c)
    """

    def __init__(self) -> None:
        from transformers import AutoProcessor, BarkModel
        import torch

        super().__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )  # mps slow af on this, won't be used
        print(f"Using {self.device} as a running device")
        model = "suno/bark"
        self.processor = AutoProcessor.from_pretrained(model)
        self.model = (
            BarkModel.from_pretrained(model).to(self.device).to_bettertransformer()
        )

    def supported_languages(self) -> list:
        return ["en"]

    def tts(self, text, out_path, voice="v2/en_speaker_6"):
        print(text)
        print(out_path, voice)
        voice = self.speaker if self.speaker is not None else voice
        # use the model to generate audio using device
        print("Perform Bark TTS...")
        print("text to model input...")
        inputs = self.processor(text, voice_preset=voice).to(self.device)
        print("model input to audio...")
        audio_array = self.model.generate(**inputs).to(self.device)
        print("saving audio...")
        # convert to numpy array
        audio_array = audio_array.cpu().numpy().squeeze()

        # save to file using soundfile
        sf.write(out_path, audio_array, self.model.generation_config.sample_rate)

    def requested_additional_args(self, **kwargs) -> None:
        self.speaker = kwargs.get("speaker", None)
        return None


# add your tts model mapping 'key' here
# possible_model = Literal['khanomtal11', 'key2', 'key3', ...]
possible_model = Literal[
    "gtts",
    "edge_tts",
    "bark",
]


class auto_tts:
    possible_model = possible_model  # allow auto_tts.possible_model access

    def __init__(self, model_selection: possible_model = None, **kwargs) -> None:
        self.model_mapping: Dict[possible_model, BaseTTS] = {
            # "khanomtal11": khanomtal11,
            "gtts": Gtts,
            "edge_tts": EdgeTTS,
            "bark": Bark,
            # 'key' : your tts class
        }

        # manual model selection and validation if model existed
        if model_selection is None:
            print("---- Available TTS models: ---")
            for key in self.list_all_models():
                supported_languages = self.model_mapping[key]().supported_languages()
                if len("".join(supported_languages)) > 400:
                    print(
                        "-",
                        key,
                        "| Supported Language/Speakers",
                        len(supported_languages),
                    )
                    print(
                        "(There is too much) This is Some Of them.",
                        supported_languages[:10],
                    )
                else:
                    print(
                        "-",
                        key,
                        "| Supported Language/Speakers",
                        self.model_mapping[key]().supported_languages(),
                    )
            print("-------------------------------")
            model_selection = input("Select TTS model: ")

        if not self.__validate_model(model_selection):
            raise ValueError(f"Invalid model selection: {model_selection}")

        self.model: BaseTTS = self.model_mapping[model_selection]()
        # request additional args if needed, will do nothing if it was not implemented
        self.model.requested_additional_args(**kwargs)

    def tts(self, text, out_path, model_name=None, **kwargs):
        if model_name is not None:  # use this model_name instead
            self.model_mapping[model_name]().tts(text, out_path, **kwargs)
        # use default model, initialized in __init__
        self.model.tts(text, out_path, **kwargs)
        self.__validate_file_exist(out_path)

    def __validate_file_exist(self, file_path: str) -> bool:
        if not os.path.isfile(file_path):
            raise ValueError(f"File not found: {file_path}")
        return True

    def list_all_models(self) -> list:
        return list(self.model_mapping.keys())

    def __validate_model(self, model_name: str) -> bool:
        return model_name in self.model_mapping.keys()
