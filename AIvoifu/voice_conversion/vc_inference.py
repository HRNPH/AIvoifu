from typing import Dict, List
import os
import json
import math
from typing import Literal
import wget
from glob import glob
import soundfile as sf # used to save audio
import torch
import torchaudio
from .RVC.infer_pack.models import SynthesizerTrnMs256NSFsid, SynthesizerTrnMs256NSFsid_nono
from .RVC.python_inference import load_hubert, create_vc_fn, VC
from .RVC.config import (
    is_half,
    device
)
from scipy.io.wavfile import write
from torchaudio.functional import resample
torchaudio.set_audio_backend("soundfile") # use soundfile backend, due to error with sox backend

class vc_inference:
    def __init__(self, selected_model_from_zoo:str, hubert_model:str, force_load_model=False, F0=None) -> None: # load form local not yet supported
        file_root = os.path.dirname(os.path.abspath(__file__))
        self.config_path = f'{file_root}/config.json'
        self.zoo_path = f'{file_root}/zoo' # do not end with /
        self.model_root = f'{file_root}/models' # do not end with /
        
        print("Initializing Waifu Voice Conversion Pipeline...")
        # ask if download zoo model or select from local
        if selected_model_from_zoo:
            name, checkpoint_link, feature_retrieval_library_link, feature_file_link = self.__select_model_from_zoo(model_name=selected_model_from_zoo)
            self.pretrain_model_name = name
            self.checkpoint_link = checkpoint_link
            self.feature_retrieval_library_link = feature_retrieval_library_link
            self.feature_file_link = feature_file_link
            
            print('Downloading model from zoo...')
            self.__download_model_from_zoo(model_name=selected_model_from_zoo, force_reload=force_load_model)
            self.pretrain_model_path = f'{self.model_root}/{self.pretrain_model_name}'
        # local model not yet implemented
        
        pretrain_path = self.__retrieve_model_checkpoint_path()
        print(f'Using model {self.pretrain_model_path}')

        # load content encoder (Hubert)
        hubert_path = self.__download_hubert_model(hubert_model, force_reload=force_load_model)
        load_hubert(hubert_path)
        # load pretrain model
        print('Loading pretrain model...')
        cpt = torch.load(pretrain_path['model_pth'], map_location="cpu")
        tgt_sr = cpt["config"][-1] # target sample rate
        cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
        if_f0 = cpt.get("f0", 1)
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
        del net_g.enc_q
        print(net_g.load_state_dict(cpt["weight"], strict=False))  # 不加这一行清不干净, 真奇葩
        net_g.eval().to(device)
        if is_half:
            net_g = net_g.half()
        else:
            net_g = net_g.float()
        vc = VC(tgt_sr, device, is_half)
        # create voice conversion function
        self.vc_fn = create_vc_fn(tgt_sr, net_g, vc, if_f0, pretrain_path['feature_retrieval_library'], pretrain_path['feature_file'])

    # def __update_self(self, selected_model_from_zoo:str, force_load_model=False, F0=None) -> None:
    def convert(self, audio_path: str, save_path=None, vc_transpose: int = 0, vc_f0method: str = "harvest", vc_index_ratio: int = 0.6):
        if vc_f0method not in ["harvest", "pm"]: # Pitch extraction algorithm, PM is fast but Harvest is better for low frequencies",
            print("Pitch extraction algorithm, PM is fast but Harvest is better for low frequencies")
            raise ValueError("vc_f0method must be one of ['harvest', 'pm']")
        # info, audio = vc_fn(vc_input, vc_transpose, vc_f0method[1], vc_index_ratio)
        info, audio = self.vc_fn(audio_path, vc_transpose, vc_f0method, vc_index_ratio)
        sample_rate, audio_data = audio
        if (sample_rate is None) or (audio_data is None):
            raise ValueError("Audio data is None, There's probably some shit wrong with the Model you downloaded, Please Report it in the repo!")
        if not save_path:
            save_path = "output.wav"
        sf.write(save_path, audio[1], audio[0])
        return audio_data, sample_rate
    
    def __get_available_model_from_zoo(self) -> Dict[str, Dict[Literal['name', 'author', 'description', 'origin', 'checkpoint_link', 'feature_retrieval_library_link', 'feature_file_link'], str]]:
        models: Dict[str, Dict] = {}
        for model_path in glob(f'{self.zoo_path}/*/meta.json'):
            model_name = os.path.basename(os.path.dirname(model_path)) # folder name
            model_meta = json.load(open(model_path, 'r'))
            models[model_name] = {
                'name': model_name,
                'author': model_meta['AUTHOR'],
                'license': model_meta['LICENSE'], # 'MIT', 'CC-BY-4.0'
                'description': model_meta['DESCRIPTION'],
                'origin': model_meta['ORIGIN'],
                'checkpoint_link': model_meta['CHECKPOINT_LINK'],
                'feature_retrieval_library_link': model_meta['FEATURE_RETRIEVAL_LIBRARY_LINK'],
                'feature_file_link': model_meta['FEATURE_FILE_LINK']
            }
        return models 
    
    def __is_model_available_in_zoo(self, model_name: str) -> bool:
        return model_name in self.__get_available_model_from_zoo()
    
    def __select_model_from_zoo(self, model_name: str) -> "tuple[str, str, str, str]":
        """
        return: (model_name, checkpoint_link, feature_retrieval_library_link, feature_file_link)
        """
        if not self.__is_model_available_in_zoo(model_name):
            err_qry = ['please select a model from the list below:'] + list(self.__get_available_model_from_zoo().keys())
            print(err_qry)
            raise ValueError(f'Model |{model_name}| is not available in zoo ' + '\n - '.join(err_qry))
        else:
            model_meta = self.__get_available_model_from_zoo()[model_name]
            return model_meta['name'], model_meta['checkpoint_link'], model_meta['feature_retrieval_library_link'], model_meta['feature_file_link']
    
    def __download_model_from_zoo(self, model_name: str = None, force_reload: bool = False) -> "tuple[str, str, str, str]":
        """
        model_name: model name
        force_reload: force reload checkpoint
        return: (model_name, checkpoint_path, feature_retrieval_library_path, feature_file_path)
        """
        # validate model name
        name, checkpoint_link, feature_retrieval_library_link, feature_file_link = self.__select_model_from_zoo(model_name if model_name else self.pretrain_model_name if self.pretrain_model_name else None)
        # download model
        os.makedirs(f'{self.model_root}/{model_name}/', exist_ok=True)
        links = [checkpoint_link, feature_retrieval_library_link, feature_file_link]
        save_paths = [f'{self.model_root}/{name}/model.pth', f'{self.model_root}/{name}/feature_retrieval_library.index', f'{self.model_root}/{name}/feature.npy']
        labels = ['checkpoint', 'feature_retrieval_library', 'feature_file']
        self.__load_checkpoint_utils(links, save_paths, labels, force_reload=False)
        return name, save_paths[0], save_paths[1], save_paths[2]
    
    def __load_checkpoint_utils(self, checkpoint_links: List[str], save_paths: List[str], labels: List[str], force_reload: bool) -> None:
        """
        checkpoint_links: list of checkpoint links
        save_paths: list of save_path
        logging_msg: list of logging message for each checkpoint
        force_reload: force reload checkpoint
        """
        for i, checkpoint_link in enumerate(checkpoint_links):
            save_path = save_paths[i]
            label = labels[i]
            if not os.path.exists(save_path) or force_reload:
                print(f'Loading {label}...')
                print(f'Link: {checkpoint_link}')
                wget.download(checkpoint_link, save_path)
            else:
                print(f'{label} already exists. Skipping...')
        return
    
    def __retrieve_model_checkpoint_path(self) -> Dict[Literal['model_pth', 'feature_retrieval_library', 'feature_file'], str]:
        """
        return: (checkpoint_path, feature_retrieval_library_path, feature_file_path)
        """
        model_np, feature_retrieval_library_path, feature_file_path = f'{self.pretrain_model_path}/model.pth', f'{self.pretrain_model_path}/feature_retrieval_library.index', f'{self.pretrain_model_path}/feature.npy'
        return  {
            'model_pth': model_np,
            'feature_retrieval_library': feature_retrieval_library_path,
            'feature_file': feature_file_path
        }
    
    def __get_available_hubert_models(self) -> Dict[str, Dict[Literal['NAME', 'AUTHOR', 'CHECKPOINT_LINK'], str]]:
        """
        return: list of hubert models
        """
        models = {}
        with open(self.config_path, 'r') as f:
            config = json.load(f)
            for model in config['hubert_models']:
                models[model['NAME']] = ({
                    'NAME': model['NAME'],
                    'AUTHOR': model['AUTHOR'],
                    'CHECKPOINT_LINK': model['CHECKPOINT_LINK']
                })
        return models

    def __selet_hubert_model(self, hubert_model_name: str = None) -> "tuple[str, str]":
        """
        return: (model_name, checkpoint_link)
        """
        print('---- List available Hubert models... ----')
        model_map = self.__get_available_hubert_models()
        # validate model name
        hubert_model_name = hubert_model_name.strip()
        if hubert_model_name in model_map:
            print(f'Selected model: {hubert_model_name}')
            return hubert_model_name, model_map[hubert_model_name]["CHECKPOINT_LINK"]
        else:
            raise ValueError('Invalid model name. Please try again. from the list below:\n' + '\n'.join(model_map.keys()))
    
    def __download_hubert_model(self, hubert_model_name: str = None, force_reload: bool = False) -> str:
        """
        return: hubert model path
        """
        # validate model name
        hubert_model_name, checkpoint_link = self.__selet_hubert_model(hubert_model_name)
        hubert_path = f'{self.model_root}/hubert.pt'
        self.__load_checkpoint_utils([checkpoint_link], [hubert_path], ['hubert'], force_reload=force_reload)
        return hubert_path