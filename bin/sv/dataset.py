import os
import torch
import random
import torchaudio
from tqdm import tqdm
from pathlib import Path
from os.path import join, getsize
from joblib import Parallel, delayed
from torch.utils.data import Dataset
from librosa.util import find_files


# Voxceleb1 + 2
class Voxceleb_Dataset(Dataset):
    def __init__(self, file_path, max_timestep=500, meta_data=None, dev_spkr_ids=None, utterance_sample=5, is_dev=False, is_eval=False):

        # Read file
        self.root = file_path
        self.all_speakers  = [f.path for f in os.scandir(file_path) if f.is_dir()]
        # Crop seqs that are too long
        self.max_timestep = max_timestep
        self.utterance_sample = utterance_sample
        self.is_eval = is_eval
        self.meta_data =meta_data
        self.pair_table = None
        self.is_dev = is_dev

        if not self.is_dev and not self.is_eval:
            self.speakers  = [path for path in self.all_speakers if path.split("/")[-1] not in dev_spkr_ids]
            self.speaker_num = len(self.speakers)
            self.number_range = list(range(self.speaker_num))
        
        if self.is_eval:
            self.pair_table = []
            usage_list = open(self.meta_data, "r").readlines()
            for pair in usage_list:
                list_pair = pair.split()
                pair_1= os.path.join(self.root, list_pair[1])
                pair_2= os.path.join(self.root, list_pair[2])
                one_pair = [list_pair[0],pair_1,pair_2 ]
                self.pair_table.append(one_pair)

        if self.is_dev:
            self.pair_table = []
            usage_list = open(self.meta_data, "r").readlines()
            for pair in usage_list:
                list_pair = pair.split()
                pair_1= os.path.join(self.root, list_pair[1])
                pair_2= os.path.join(self.root, list_pair[2])
                one_pair = [list_pair[0],pair_1,pair_2 ]
                self.pair_table.append(one_pair)

            

    def __len__(self):
        if self.is_eval or self.is_dev:
            return len(self.pair_table)
        else:
            return len(self.speakers)
    
    def __getitem__(self, idx):
        if self.is_eval or self.is_dev:
            y_label, x1_path, x2_path = self.pair_table[idx]
            wav1, sr = torchaudio.load(x1_path)
            wav2, _ = torchaudio.load(x2_path)

            wav1 = wav1.squeeze(0)
            wav2 = wav2.squeeze(0)

            length1 = wav1.shape[0]
            length2 = wav2.shape[0]

            return wav1, wav2, torch.tensor(int(y_label[0])), torch.tensor(length1), torch.tensor(length2)
        else:
            path = random.sample(find_files(self.speakers[idx],ext=["wav"]), self.utterance_sample)
            x_list = []
            length_list = []
            index = []

            for i in range(len(path)):
                wav, sr=torchaudio.load(path[i])
                wav = wav.squeeze(0)
                length = wav.shape[0]
                if length > self.max_timestep:
                    x = wav[:self.max_timestep]
                    length = self.max_timestep
                else:
                    x = wav
                x_list.append(x)
                index.append(torch.tensor(idx).long())
                length_list.append(torch.tensor(length).long())
            
            return x_list, torch.stack(length_list), torch.stack(index)
        
