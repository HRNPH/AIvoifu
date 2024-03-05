from AIvoifu import client_pipeline
import os


def test_tts_generate():
    model = client_pipeline.tts.auto_tts(model_selection="gtts", language="en")
    model.tts("Hello This Is A Test Text Anyway", out_path="./tts.wav")
    assert os.path.exists("./tts.wav")
    os.remove("./tts.wav")


def test_tts_vc_generate():
    model = client_pipeline.tts_pipeline(
        tts_model_selection="gtts",
        vc_model_selection="ayaka-jp",
        hubert_model="zomehwh-hubert-base",
        language="en",
    )
    model.tts("Hello This Is A Test Text Anyway", save_path="./test.wav")
    assert os.path.exists("./test.wav")
    os.remove("./test.wav")
