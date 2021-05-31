import torch
from torch.utils.data import Dataset, DataLoader


import math
import os
import random
import numpy as np

from bin.tts.utils import pad_1D, pad_2D
from bin.tts.text import text_to_sequence, sequence_to_text


class LibriTTSDataset(Dataset):
    def __init__(self, data_config, filename="train.txt", speaker_filename="train.txt", sort=True):
        ## configs
        self.use_spk_embed = data_config['use_spk_embed']
        self.preprocessed_path = data_config['preprocessed_path']
        self.dataset = data_config['dataset']
        ## 

        self.basename, self.text, _ = self.process_meta(os.path.join(self.preprocessed_path, filename))
        self.spk_basename, self.spk_text, self.spk_to_file_ids = self.process_meta(os.path.join(self.preprocessed_path, speaker_filename))
        self.sort = sort
        self.spk_table, self.inv_spk_table = self.get_spk_table()

        if filename == "train.txt":
            self.mode = "train"
        else:
            self.mode = "val"
            
            
    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        phone = np.array(text_to_sequence(self.text[idx], []))
        
        mel_path = os.path.join(self.preprocessed_path, "mel", "{}-mel-{}.npy".format(self.dataset, basename))
        mel_target = np.load(mel_path)
        
        D_path = os.path.join(self.preprocessed_path, "alignment", "{}-ali-{}.npy".format(self.dataset, basename))
        D = np.load(D_path)
        
        f0_path = os.path.join(self.preprocessed_path, "f0", "{}-f0-{}.npy".format(self.dataset, basename))
        f0 = np.load(f0_path)
        
        energy_path = os.path.join(self.preprocessed_path, "energy", "{}-energy-{}.npy".format(self.dataset, basename))
        energy = np.load(energy_path)

        # for spk examples
        if self.mode == "val":
            file_id = self.spk_to_file_ids[basename.split('_')[0]][0]

            example_basename = self.spk_basename[file_id]
            example_phone = np.array(text_to_sequence(self.spk_text[file_id], []))
            
            example_mel_path = os.path.join(self.preprocessed_path, "mel", "{}-mel-{}.npy".format(self.dataset, example_basename))
            example_mel_target = np.load(example_mel_path)
            
            example_D_path = os.path.join(self.preprocessed_path, "alignment", "{}-ali-{}.npy".format(self.dataset, example_basename))
            example_D = np.load(example_D_path)
            
            example_f0_path = os.path.join(self.preprocessed_path, "f0", "{}-f0-{}.npy".format(self.dataset, example_basename))
            example_f0 = np.load(example_f0_path)
            
            example_energy_path = os.path.join(self.preprocessed_path, "energy", "{}-energy-{}.npy".format(self.dataset, example_basename))
            example_energy = np.load(example_energy_path)
        
        sample = {"id"        : basename,
                  "text"      : phone,
                  "mel_target": mel_target,
                  "D"         : D,
                  "f0"        : f0,
                  "energy"    : energy}
        if self.mode == "val":
            sample.update(
                 {"example_id"        : example_basename,
                  "example_text"      : example_phone,
                  "example_mel_target": example_mel_target,
                  "example_D"         : example_D,
                  "example_f0"        : example_f0,
                  "example_energy"    : example_energy}
            )

        return sample
    # batch : 16 * 16
    # real_batchsize : 16
    # batch of sample, so batch is list of dicts
    def collate_fn(self, batch):
        output = self.reprocess(batch)
        return output

    def reprocess(self, batch):
        ids         = [d["id"] for d in batch]
        texts       = [d["text"] for d in batch]
        mel_targets = [d["mel_target"] for d in batch]
        Ds          = [d["D"] for d in batch]
        f0s         = [d["f0"] for d in batch]
        energies    = [d["energy"] for d in batch]

        spk_ids = [self.spk_table[_id.split("_")[0]] for _id in ids]

        if self.mode == "val":
            example_ids         = [d["example_id"] for d in batch]
            example_texts       = [d["example_text"] for d in batch]
            example_mel_targets = [d["example_mel_target"] for d in batch]
            example_Ds          = [d["example_D"] for d in batch]
            example_f0s         = [d["example_f0"] for d in batch]
            example_energies    = [d["example_energy"] for d in batch]

            ids = ids + example_ids
            texts = texts + example_texts
            mel_targets = mel_targets + example_mel_targets
            Ds = Ds + example_Ds
            f0s = f0s + example_f0s
            energies = energies + example_energies


        for text, D, id_ in zip(texts, Ds, ids):
            if len(text) != len(D):
                print(text, text.shape, D, D.shape, id_)
                
        length_text = np.array(list())
        for text in texts:
            length_text = np.append(length_text, text.shape[0])

        length_mel = np.array(list())
        for mel in mel_targets:
            length_mel = np.append(length_mel, mel.shape[0])
            
        texts = pad_1D(texts)
        Ds = pad_1D(Ds)
        mel_targets = pad_2D(mel_targets)
        f0s = pad_1D(f0s)
        energies = pad_1D(energies)
        log_Ds = np.log(Ds + 1.)

        out = {"id": ids,
               "text": texts,
               "mel_target": mel_targets,
               "D": Ds,
               "log_D": log_Ds,
               "f0": f0s,
               "energy": energies,
               "src_len": length_text,
               "mel_len": length_mel,
               "spk_ids": spk_ids}

        return out
    
    def process_meta(self, meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            text = []
            name = []
            spk_to_file_ids = {}
            for i, line in enumerate(f.readlines()):
                #100_122655_000027_000001|{HH IY1 R IH0 Z UW1 M D}
                n, t = line.strip('\n').split('|')
                name.append(n)
                text.append(t)
                spk = n.split('_')[0]
                if spk in spk_to_file_ids:
                    spk_to_file_ids[spk].append(i)
                else:
                    spk_to_file_ids[spk] = [i]
            return name, text, spk_to_file_ids

    def get_spk_table(self):
        '''
        spk_table     : {'14' :0, '16': 1, ...}
        inv_spk_table : { 0:'14', 1: '16', ...}
        '''
        spk_table = {}
        spk_id = 0
        
        ## temp, use TextGrid folder to get spk table
        spks_dir = os.path.join(self.preprocessed_path, "TextGrid")
        spks = os.listdir(spks_dir)
        for i in spks: 
            if i.endswith(".txt") : spks.remove(i)

        spks.sort()
        for spk in spks:
            spk_table[spk] = spk_id
            spk_id += 1
        inv_spk_table = {v:k for k, v in spk_table.items()}
        return spk_table, inv_spk_table
