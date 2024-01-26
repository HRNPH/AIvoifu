# ----------- Waifu Vocal Pipeline -----------------------
from AIvoifu.tts import tts
from AIvoifu.voice_conversion import vc_inference as vc

class tts_pipeline:
    def __init__(self,
            tts_model_selection: tts.auto_tts.possible_model = None,
            vc_model_selection: str = None,
            hubert_model: str = None,
            force_load_model: bool = False,
            **kwargs
        ) -> None:
        print('Loading Waifu Vocal Pipeline...')
        self.cache_root = './audio_cache'
        self.model = tts.auto_tts(model_selection=tts_model_selection, **kwargs)
        self.vc_model = vc.vc_inference(selected_model_from_zoo=vc_model_selection, hubert_model=hubert_model, force_load_model=force_load_model)
        print('Loaded Waifu Vocal Pipeline')

    def tts(self, text, voice_conversion=True, save_path=None):
        # text to speech
        if not save_path:
            save_path = f'{self.cache_root}/dialog_cache.wav'
        self.model.tts(text, save_path)
        # voice conversion
        if voice_conversion:
            self.vc_model.convert(save_path, save_path=save_path, vc_transpose=3)
        return save_path
    

if __name__ == '__main__':
    model = tts_pipeline()
    model.tts('Hello This Is A Test Text Anyway', save_path='./test.wav')