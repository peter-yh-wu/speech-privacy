"""
Data loading utilities
"""

import glob
import math
import numpy as np
import os
import random
import soundfile as sf
import torch

from torch.utils.data import Dataset


def collate_1d_float(batch):
    text_xs = []
    text_ys = []
    max_len_text = 0
    for (text_x, _) in batch:
        x_len = len(text_x)
        if x_len > max_len_text:
            max_len_text = x_len
    for (text_x, text_y) in batch:
        text_x = np.pad(text_x, (0, max_len_text-len(text_x)), 'constant', constant_values=0)
        text_xs.append(torch.tensor(text_x))
        text_ys.append(text_y)
    text_xs = torch.stack(text_xs)
    return text_xs.float(), torch.tensor(text_ys)


class librispeech_dataset(Dataset):
    def __init__(self, phase, mode='default', num_spkr=1, spkr_seed=-1):
        self.mode = mode
        if phase == 'train':
            data_dir = 'LibriSpeech/train-clean-100'
        else:
            data_dir = 'LibriSpeech/test-clean'
        if not os.access(data_dir, os.R_OK):
            raise FileNotFoundError
        
        metadata_path = 'LibriSpeech/SPEAKERS.TXT'
        if not os.access(metadata_path, os.R_OK):
            raise FileNotFoundError

        with open(metadata_path, 'r') as inf:
            metadata = inf.readlines()
        # 30   | F | train-clean-360  | 25.19 | Annie Coleman Rothenberg
        metadata = [l.strip() for l in metadata]
        metadata = [l for l in metadata if len(l) > 0 and l[0] != ';']
        metadata = [l.split('|') for l in metadata]
        metadata = [[l.strip() for l in ll] for ll in metadata]
        id_to_gender = {l[0]:l[1] for l in metadata}

        ext = 'flac'
        search_path = os.path.join(data_dir, '**/*.' + ext)

        self.audio_paths = []
        self.labels = []
        self.speaker_ids = []
        file_paths = []
        for fname in glob.iglob(search_path, recursive=True):
            if '-avc' not in fname and '-gvc' not in fname and '-rb' not in fname and '-pitch' not in fname:
                file_path = os.path.realpath(fname)
                slash_idx = file_path.rfind('/')
                end_idx = file_path[:slash_idx].rfind('/')
                start_idx = file_path[:end_idx].rfind('/')+1
                speaker_id = file_path[start_idx:end_idx]
                self.speaker_ids.append(speaker_id)
                self.audio_paths.append(file_path)
                gender = id_to_gender[speaker_id]
                self.labels.append(int(gender == 'F'))
                file_paths.append(file_path)
        
        if phase == 'train':
            speaker_list = ['311', '2843', '3664', '3168', '2518', '7190', '78', '831', '8630', '3830', '322', '2391', '7517', '8324', '19', '1898', '7078', '5339', '4051', '4640']
            m_list = [s for s in speaker_list if id_to_gender[s] == 'M']
            f_list = [s for s in speaker_list if id_to_gender[s] == 'F']    
            if spkr_seed != -1:
                random.Random(spkr_seed).shuffle(m_list)
                random.Random(spkr_seed).shuffle(f_list)
                speaker_list = m_list[:num_spkr] + f_list[:num_spkr]
                print(speaker_list)
            else:
                speaker_list = m_list[:num_spkr] + f_list[:num_spkr]
            speaker_set = set(speaker_list)
            self.audio_paths = [p for i, p in enumerate(self.audio_paths) if self.speaker_ids[i] in speaker_set]
            self.labels = [p for i, p in enumerate(self.labels) if self.speaker_ids[i] in speaker_set]
            self.speaker_ids = [p for i, p in enumerate(self.speaker_ids) if self.speaker_ids[i] in speaker_set]

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, i):
        audio_path = self.audio_paths[i]

        if self.mode == 'avc':
            audio_path = audio_path.replace('.flac', '-avc.flac')
        elif self.mode == 'gvc':
            audio_path = audio_path.replace('.flac', '-gvc.flac')
        elif self.mode == 'pitch':
            audio_path = audio_path.replace('.flac', '-rb.flac')
        elif self.mode != 'default':
            audio_path = audio_path.replace('.flac', '-%s.flac' % self.mode)

        audio, sr = sf.read(audio_path)

        return audio.astype(float), self.labels[i]
