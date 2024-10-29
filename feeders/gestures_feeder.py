import numpy as np
import pandas as pd
import os
import pickle
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import sys
import random
sys.path.extend(['../'])
from feeders import tools
import torchaudio
from torchaudio import functional as F
from torch.functional import F as F2
from torchaudio.utils import download_asset
import librosa
from einops import rearrange
from data.read_process_poses import load_keypoints_dict

from tqdm import tqdm
from utils.augmentation_utils import compose_random_augmentations
mediapipe_flip_index = np.concatenate(([0,2,1,4,3,6,5], [28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48], [7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26, 27], ), axis=0) 

mmpose_flip_index = np.concatenate(([0,2,1,4,3,6,5],[17,18,19,20,21,22,23,24,25,26],[7,8,9,10,11,12,13,14,15,16]), axis=0) 

effect = ",".join(
    [
        "lowpass=frequency=300:poles=1",  # apply single-pole lowpass filter
        "atempo=0.8",  # reduce the speed
        "aecho=in_gain=0.8:out_gain=0.9:delays=200:decays=0.3|delays=400:decays=0.3"
        # Applying echo gives some dramatic feeling
    ],
)

# effect = [
#     ["lowpass", "-f", "300", "-p", "1"],  # apply single-pole lowpass filter
#     ["atempo", "0.8"],                   # reduce the speed
#     ["aecho", "0.8", "0.9", "200", "0.3", "400", "0.3"] # echo effect
# ]

# Define effects
effects = [
    ["lowpass", "-1", "300"],  # apply single-pole lowpass filter
   #  ["speed", "0.8"],  # reduce the speed
    # This only changes sample rate, so it is necessary to
    # add `rate` effect with original sample rate after this.
   #  ["rate", f"{sample_rate1}"],
    ["reverb", "-w"],  # Reverbration gives some dramatic feeling
]

configs = [
    {"format": "wav", "encoding": "ULAW", "bits_per_sample": 8},
    {"format": "gsm"},
   #  {"format": "vorbis", "compression": -1},
]
class Feeder(Dataset):
    def __init__(
            self,
            random_choose=True,
            random_shift=True,
            random_move=True,
            window_size=25,
            normalization=True,
            debug=False,
            use_mmap=True,
            random_mirror=False,
            random_mirror_p=0.5,
            is_vector=False,
            fold=0,
            sample_rate1 = 16000,
            debug_audio=False,
            data_path='./data',
            poses_path='./data',
            modalities=["skeleton", "speech"],
            audio_path = './data',
            n_views=2,
            apply_skeleton_augmentations=True,
            skeleton_augmentations=None,
            random_scale=True,
            fps = 29.97,
            speech_buffer = 0.5,
            gesture_buffer = 0.25,
            apply_speech_augmentations=False,
            global_normalization=False,
    ):
        """
         
        :param poses_path: path to poses 
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move: 
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        :param data_path: path to data
        :param modalities: list of modalities
        :param n_views: number of views (1 or 2)
        :param apply_augmentations: whether to apply augmentations to view1, view2 is augmented by default
        """
        self.data_path = data_path
        self.poses_path = poses_path
        self.audio_path = audio_path
        self.modalities = modalities
        self.apply_speech_augmentations = apply_speech_augmentations
        self.global_normalization = global_normalization
        self.apply_skeleton_augmentations=apply_skeleton_augmentations
        self.skeleton_augmentations=skeleton_augmentations
        self.random_scale = random_scale
        if apply_skeleton_augmentations:
            assert self.skeleton_augmentations is not None, "Augmentations are not provided"
        self.n_views=n_views

        self.debug_audio = debug_audio
        self.debug = debug
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.random_mirror = random_mirror
        self.random_mirror_p = random_mirror_p
        self.audio_sample_rate = sample_rate1
        self.fps = fps
        self.speech_buffer = speech_buffer
        self.gesture_buffer = gesture_buffer

        self.load_data()
        self.is_vector = is_vector
        
        if self.global_normalization:
            self.get_mean_map()
        self.audio_sample_per_frame = int(self.audio_sample_rate/self.fps)
        SAMPLE_RIR = download_asset("tutorial-assets/Lab41-SRI-VOiCES-rm1-impulse-mc01-stu-clo-8000hz.wav")
        SAMPLE_NOISE = download_asset("tutorial-assets/Lab41-SRI-VOiCES-rm1-babb-mc01-stu-clo-8000hz.wav")
        rir_raw, rir_sample_rate = torchaudio.load(SAMPLE_RIR)
        self.rir_sample_rate = rir_sample_rate
        self.noise, noise_sample_rate = torchaudio.load(SAMPLE_NOISE)
        # target sample rate is 16000
        self.noise = F.resample(self.noise, orig_freq=noise_sample_rate, new_freq=sample_rate1)
        rir = rir_raw[:, int(rir_sample_rate * 1.01) : int(rir_sample_rate * 1.3)]
        self.rir = rir / torch.linalg.vector_norm(rir, ord=2)

    def load_skeletal_data(self):
        self.data = pd.read_pickle(self.data_path)
        self.data = self.data.reset_index(drop=True)
        if self.debug:
            # select rows of pair04
            self.data = self.data[self.data['pair_speaker'] == 'pair04_A']
        self.pairs_speakers = self.data['pair_speaker'].unique()
        self.poses = {}
        self.poses, self.mirrored_poses = load_keypoints_dict()

    def load_speech_data(self):
        if not self.debug_audio:
            self.audio_dict = {}
            print('Loading audio files...')
            for speaker in tqdm(self.pairs_speakers):
                pair = speaker.split('_')[0]
                speaker = speaker.split('_')[1]
                pair_speaker = f"{pair}_{speaker}"
                try:
                    audio_path = self.audio_path.format(pair, speaker)
                    input_audio, sample_rate = librosa.load(audio_path, sr=self.audio_sample_rate)
                except FileNotFoundError:
                    audio_path = self.audio_path.format(pair, pair, speaker)
                    input_audio, sample_rate = librosa.load(audio_path, sr=self.audio_sample_rate)
                self.audio_dict[pair_speaker] = {'audio': input_audio, 'sample_rate': sample_rate}

    def load_data(self):
        if "skeleton" in self.modalities:
            self.load_skeletal_data()
        if "speech" in self.modalities:
            self.load_speech_data()
        
    # TODO: define augmentations from HAR project. Should the strength of augmentations be the same?
    def augment_skeleton(self, data_numpy, config_dict):
        augmentations = transforms.Compose(compose_random_augmentations("skeleton", config_dict))
        return augmentations(data_numpy)
       

    def apply_codec(self, waveform, orig_sample_rate, **kwargs):
        if orig_sample_rate != 8000:
            waveform = F.resample(waveform, orig_sample_rate, 8000)
            sample_rate = 8000
        augmented = F.apply_codec(waveform, sample_rate, **kwargs)
        # effector = torchaudio.io.AudioEffector(effect=effect)
        # augmented_effector = effector.apply(waveform, sample_rate)
        # resample to original sample rate
        augmented = F.resample(augmented, sample_rate, orig_sample_rate)
        return augmented
       
    def augment_audio(self, audio, augemntation_apply=True):
        if not augemntation_apply:
            return audio
        # apply effects
        lengths = audio.shape[0]
        audio = torch.from_numpy(audio).float().unsqueeze(0)
        # apply effects with 50% probability
        coin_toss = random.random()
        if coin_toss < 0.33:
            audio, _ = torchaudio.sox_effects.apply_effects_tensor(audio, self.audio_sample_rate, effects)
            # choose randomly one augmented speech from the two augmented speech
            idx = random.randint(0, audio.shape[0] - 1)
            audio = audio[idx].unsqueeze(0)
            augemntation_apply = False
        elif coin_toss < 0.66:
            if self.noise.shape[1] < audio.shape[1]:
                noise = self.noise.repeat(1, 2)[:,:audio.shape[1]]
            else:
                noise = self.noise[:, : audio.shape[1]]
            snr_dbs = torch.tensor([20, 10, 3])
            audio = F.add_noise(audio, noise, snr_dbs)
            # choose randomly one noisy speech
            idx = random.randint(0, audio.shape[0] - 1)
            audio = audio[idx].unsqueeze(0)
            augemntation_apply = False
        else:
            waveforms = []
            for param in configs:
                augmented = self.apply_codec(audio, self.audio_sample_rate, **param)
                waveforms.append(augmented)
            # choose randomly one codec
            idx = random.randint(0, len(waveforms) - 1)
            audio = waveforms[idx]
            augemntation_apply = False
            if audio.shape[1] > lengths: # TODO: check the validity of this operation: if the augmented speech is longer than the original speech, truncate it
                audio = audio[:, :lengths]
        audio = audio.squeeze(0).numpy()
        return audio
  
    def get_mean_map(self):
        data = self.get_all_poses()
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))
    def get_all_poses(self):
        # reset index of the data
        all_poses = np.zeros((len(self.data), 3, 30, 27, 1))
        for index, row in tqdm(self.data.iterrows(), total=self.data.shape[0], desc='Loading poses...'):
            start_frame = int(row['start_frames']) #- round(self.gesture_buffer * self.fps)
            end_frame = int(row['end_frames']) #+ round(self.gesture_buffer * self.fps)
            middle_frame = (start_frame + end_frame) // 2
            # take 15 frames before and after the middle frame
            start_frame = middle_frame - 15
            end_frame = start_frame + 30
            pair_speaker = row['pair_speaker']
            all_poses[index] = self.poses[pair_speaker][:, start_frame:end_frame, :, :]
        return all_poses
                
    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        item = {
            "label": 0
        }
        row = self.data.iloc[index]
        # check if embeddings are present
        if 'embeddings' in self.data.columns:
            item['embeddings'] = row['embeddings']
        else:
            item['embeddings'] = 0
        start_frame = int(row['start_frames']) #- round(self.gesture_buffer * self.fps)
        end_frame = int(row['end_frames']) #+ round(self.gesture_buffer * self.fps)
        middle_frame = (start_frame + end_frame) // 2
        # take 15 frames before and after the middle frame
        start_frame = middle_frame - 15
        end_frame = start_frame + 30
        speech_start_frame = middle_frame - 30
        speech_end_frame = speech_start_frame + 60
        pair_speaker = row['pair_speaker']
        number_speech_frames = int(2 * self.audio_sample_rate)
        if "skeleton" in self.modalities:
            # select either the original or the mirrored poses
            if self.random_mirror and random.random() > self.random_mirror_p:
                skeleton_data = self.mirrored_poses[pair_speaker][:, start_frame:end_frame, :, :]  
            else:
                skeleton_data = self.poses[pair_speaker][:, start_frame:end_frame, :, :]          
            item["skeleton"] = {}
            # skeleton_data = self.data[index]
            skeleton_data_numpy = np.array(skeleton_data)
            if self.normalization:
                assert skeleton_data_numpy.shape[0] == 3
                if self.global_normalization:
                    skeleton_data_numpy = (skeleton_data_numpy - self.mean_map) / self.std_map
                elif self.is_vector:
                    skeleton_data_numpy[0,:,0,:] = skeleton_data_numpy[0,:,0,:] - skeleton_data_numpy[0,:,0,0].mean(axis=0)
                    skeleton_data_numpy[1,:,0,:] = skeleton_data_numpy[1,:,0,:] - skeleton_data_numpy[1,:,0,0].mean(axis=0)
                else:
                    skeleton_data_numpy[0,:,:,:] = skeleton_data_numpy[0,:,:,:] - skeleton_data_numpy[0,:,0,0].mean(axis=0)
                    skeleton_data_numpy[1,:,:,:] = skeleton_data_numpy[1,:,:,:] - skeleton_data_numpy[1,:,0,0].mean(axis=0)
            item["skeleton"]["orig"] = skeleton_data_numpy

            # skeleton data augmentation and view2 generation
            if self.apply_skeleton_augmentations:
                skeleton_data_numpy_1 = np.array(self.augment_skeleton(
                    torch.tensor(skeleton_data_numpy).float(), 
                    self.skeleton_augmentations))
                item["skeleton"]["view1"] = skeleton_data_numpy_1
                if self.n_views == 2:
                    # skeleton_data_numpy_2 = self.augment_skeleton_simple(skeleton_data_numpy, augemntation_apply=self.apply_augmentations)
                    skeleton_data_numpy_2 = np.array(self.augment_skeleton(
                        torch.tensor(skeleton_data_numpy).float(),
                        self.skeleton_augmentations))
                    item["skeleton"]["view2"] = skeleton_data_numpy_2
        if "speech" in self.modalities:
            item["speech"] = {}
            start_frame = speech_start_frame * self.audio_sample_per_frame
            end_frame = start_frame + 2 * self.audio_sample_rate
            if self.debug_audio:
                audio_data = np.zeros(2 * self.audio_sample_rate)
            else: 
                if start_frame < 0:
                    padding = np.zeros(abs(start_frame))
                    start_frame = 0
                    audio_data = np.concatenate((padding, self.audio_dict[pair_speaker]['audio'][start_frame:end_frame]))
                elif end_frame > len(self.audio_dict[pair_speaker]['audio']):
                    padding = np.zeros(end_frame - len(self.audio_dict[pair_speaker]['audio']))
                    end_frame = len(self.audio_dict[pair_speaker]['audio'])
                    audio_data = np.concatenate((self.audio_dict[pair_speaker]['audio'][start_frame:end_frame], padding))
                else:
                    audio_data = self.audio_dict[pair_speaker]['audio'][start_frame:end_frame]
                if len(audio_data) > number_speech_frames:
                    audio_data = audio_data[:number_speech_frames]
                elif len(audio_data) < number_speech_frames:
                    padding = np.zeros(number_speech_frames - len(audio_data))
                    audio_data = np.concatenate((audio_data, padding))
             
            item["speech"]["orig"] = audio_data  
            # speech data augmentation and view2 generation  
            if self.apply_speech_augmentations:
                speech_wave_form_1 = self.augment_audio(audio_data)
                item["speech"]["view1"] = speech_wave_form_1

                if self.n_views == 2:
                    speech_wave_form_2 = self.augment_audio(audio_data)                
                    item["speech"]["view2"] = speech_wave_form_2

        return item
