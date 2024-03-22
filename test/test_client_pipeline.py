from AIvoifu import client_pipeline
import os


def test_tts_generate():
    for tts_model in ["gtts", "edge_tts", "xtts"]:
        model = client_pipeline.tts.auto_tts(model_selection=tts_model, language="en")
        model.tts("Hello This Is A Test Text Anyway", out_path=f"./tts_{tts_model}.wav")
        assert os.path.exists(f"./tts_{tts_model}.wav")
        os.remove(f"./tts_{tts_model}.wav")
        print(f"Passed {tts_model} TTS Test")


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
