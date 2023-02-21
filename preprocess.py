import os
import numpy as np
import librosa
import torch
import pyworld as pw
import parselmouth
import argparse
import shutil
from logger import utils
from tqdm import tqdm
from ddsp.vocoder import Audio2Mel
from librosa.filters import mel as librosa_mel_fn
from logger.utils import traverse_dir
import concurrent.futures

def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="path to the config file")
    return parser.parse_args(args=args, namespace=namespace)
    
def preprocess(
        path_srcdir, 
        path_meldir,
        path_f0dir,
        path_uvdir,
        path_skipdir,
        device,
        f0_extractor,
        f0_min,
        f0_max,
        sampling_rate,
        hop_length,
        win_length,
        n_mel_channels,
        mel_fmin,
        mel_fmax):
        
    # list files
    filelist =  traverse_dir(
        path_srcdir,
        extension='wav',
        is_pure=True,
        is_sort=True,
        is_ext=True)

    # initilize extractor
    mel_extractor = Audio2Mel(
        hop_length=hop_length,
        sampling_rate=sampling_rate,
        n_mel_channels=n_mel_channels,
        win_length=win_length,
        mel_fmin=mel_fmin,
        mel_fmax=mel_fmax,
        clamp=1e-6).to(device)

    # run
    
    def process(file):
        ext = file.split('.')[-1]
        binfile = file[:-(len(ext)+1)]+'.npy'
        path_srcfile = os.path.join(path_srcdir, file)
        path_melfile = os.path.join(path_meldir, binfile)
        path_f0file = os.path.join(path_f0dir, binfile)
        path_uvfile = os.path.join(path_uvdir, binfile)
        
        # load audio
        x, _ = librosa.load(path_srcfile, sr=sampling_rate)
        x_t = torch.from_numpy(x).float().to(device)
        x_t = x_t.unsqueeze(0).unsqueeze(0) # (T,) --> (1, 1, T)

        # extract mel
        m_t = mel_extractor(x_t)
        mel = m_t.squeeze().to('cpu').numpy()

        # extract f0 using parselmouth
        if f0_extractor == 'parselmouth':
            f0 = parselmouth.Sound(x, sampling_rate).to_pitch_ac(
                time_step=hop_length / sampling_rate, 
                voicing_threshold=0.6,
                pitch_floor=f0_min, 
                pitch_ceiling=f0_max).selected_array['frequency']
            pad_size=(int(len(x) // hop_length) - len(f0) + 1) // 2
            f0 = np.pad(f0,[[pad_size,len(mel) - len(f0) - pad_size]], mode='constant')
            
        # extract f0 using dio
        elif f0_extractor == 'dio':
            _f0, t = pw.dio(
                x.astype('double'), 
                sampling_rate, 
                f0_floor=f0_min, 
                f0_ceil=f0_max, 
                channels_in_octave=2, 
                frame_period=(1000*hop_length / sampling_rate))
            f0 = pw.stonemask(x.astype('double'), _f0, t, sampling_rate)
            f0 = f0.astype('float')[:len(mel)]
        
        # extract f0 using harvest
        elif f0_extractor == 'harvest':
            f0, _ = pw.harvest(
                x.astype('double'), 
                sampling_rate, 
                f0_floor=f0_min, 
                f0_ceil=f0_max, 
                frame_period=(1000*hop_length / sampling_rate))
            f0 = f0.astype('float')[:len(mel)]
            
        else:
            raise ValueError(f" [x] Unknown f0 extractor: {f0_extractor}")
               
        uv = f0 == 0
        if len(f0[~uv]) > 0:
            # interpolate the unvoiced f0
            f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
            uv = uv.astype('float')
            uv = np.min(np.array([uv[:-2],uv[1:-1],uv[2:]]),axis=0)
            uv = np.pad(uv, (1, 1))
            # save npy
            os.makedirs(path_meldir, exist_ok=True)
            np.save(path_melfile, mel)
            os.makedirs(path_f0dir, exist_ok=True)
            np.save(path_f0file, f0)
            os.makedirs(path_uvdir, exist_ok=True)
            np.save(path_uvfile, uv)
        else:
            print('\n[Error] F0 extraction failed: ' + path_srcfile)
            os.makedirs(path_skipdir, exist_ok=True)
            shutil.move(path_srcfile, path_skipdir)
            print('This file has been moved to ' + os.path.join(path_skipdir, file))
    print('Preprocess the audio clips in :', path_srcdir)
    
    # single process
    for file in tqdm(filelist, total=len(filelist)):
        process(file)
    
    # multi-process (have bugs)
    '''
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        list(tqdm(executor.map(process, filelist), total=len(filelist)))
    '''
                
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # parse commands
    cmd = parse_args()
    
    # load config
    args = utils.load_config(cmd.config)
    f0_extractor = args.data.f0_extractor
    f0_min = args.data.f0_min
    f0_max = args.data.f0_max
    sampling_rate  = args.data.sampling_rate
    hop_length     = args.data.block_size
    win_length     = args.data.win_length
    n_mel_channels = args.data.n_mels
    mel_fmin = args.data.mel_fmin
    mel_fmax = args.data.mel_fmax
    train_path = args.data.train_path
    valid_path = args.data.valid_path
    
    # run
    for path in [train_path, valid_path]:
        path_srcdir  = os.path.join(path, 'audio')
        path_meldir  = os.path.join(path, 'mel')
        path_f0dir  = os.path.join(path, 'f0')
        path_uvdir  = os.path.join(path, 'uv')
        path_skipdir = os.path.join(path, 'skip')
        preprocess(
            path_srcdir, 
            path_meldir, 
            path_f0dir,
            path_uvdir,
            path_skipdir,
            device,
            f0_extractor,
            f0_min,
            f0_max,
            sampling_rate,
            hop_length,
            win_length,
            n_mel_channels,
            mel_fmin,
            mel_fmax)
    
