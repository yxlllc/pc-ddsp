import os
import numpy as np
import librosa
import torch
import pyworld as pw
import parselmouth
import argparse
import shutil
import torch.multiprocessing as mp
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
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default=None,
        required=False,
        help="cpu or cuda, auto if not set")
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=2,
        required=False,
        help="number of worker processes (default: 2)")
    return parser.parse_args(args=args, namespace=namespace)
    
def load_extractors(parameters, device='cuda'):
    # initilize mel extractor
    mel_extractor = Audio2Mel(
        hop_length=parameters['hop_length'],
        sampling_rate=parameters['sampling_rate'],
        n_mel_channels=parameters['n_mel_channels'],
        win_length=parameters['win_length'],
        n_fft=parameters['n_fft'],
        mel_fmin=parameters['mel_fmin'],
        mel_fmax=parameters['mel_fmax'],
        clamp=1e-6).to(device)
    return mel_extractor


# Global variables for worker processes (each process has its own copy)
_worker_mel_extractor = None
_worker_device = None
_worker_parameters = None


def _worker_init(parameters, device):
    """Initialize extractors for each worker process."""
    global _worker_mel_extractor, _worker_device, _worker_parameters
    _worker_device = device
    _worker_parameters = parameters
    _worker_mel_extractor = load_extractors(parameters, device=device)


def _process_file(file, path_srcdir, path_meldir, path_f0dir, path_uvdir, path_skipdir):
    """
    Process a single audio file using the worker-global extractors.

    Args:
        file: Audio filename
        path_srcdir, path_meldir, path_f0dir, path_uvdir, path_skipdir: Output directories

    Returns:
        True if successful, False otherwise
    """
    global _worker_mel_extractor, _worker_device, _worker_parameters

    mel_extractor = _worker_mel_extractor
    device = _worker_device
    f0_extractor = _worker_parameters['f0_extractor']
    f0_min = _worker_parameters['f0_min']
    f0_max = _worker_parameters['f0_max']
    sampling_rate = _worker_parameters['sampling_rate']
    hop_length = _worker_parameters['hop_length']

    ext = file.split('.')[-1]
    binfile = file[:-(len(ext)+1)]+'.npy'
    path_srcfile = os.path.join(path_srcdir, file)
    path_melfile = os.path.join(path_meldir, binfile)
    path_f0file = os.path.join(path_f0dir, binfile)
    path_uvfile = os.path.join(path_uvdir, binfile)
    path_skipfile = os.path.join(path_skipdir, file)
    
    # load audio
    x, _ = librosa.load(path_srcfile, sr=sampling_rate)
    x_t = torch.from_numpy(x).float().to(device)
    x_t = x_t.unsqueeze(0).unsqueeze(0) # (T,) --> (1, 1, T)

    # extract mel
    m_t = mel_extractor(x_t)
    mel = m_t.squeeze().to('cpu').numpy()

    # extract f0 using parselmouth
    if f0_extractor == 'parselmouth':            
        l_pad = int(np.ceil(1.5 / f0_min * sampling_rate))
        r_pad = hop_length * ((len(x) - 1) // hop_length + 1) - len(x) + l_pad + 1
        s = parselmouth.Sound(np.pad(x, (l_pad, r_pad)), sampling_rate).to_pitch_ac(
                time_step=hop_length / sampling_rate, voicing_threshold=0.6,
                pitch_floor=f0_min, pitch_ceiling=f0_max)
        assert np.abs(s.t1 - 1.5 / f0_min) < 0.001
        f0 = s.selected_array['frequency']
        if len(f0) < len(mel):
            f0 = np.pad(f0, (0, len(mel) - len(f0)))
        f0 = f0[: len(mel)]
        
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
        uv = np.pad(uv, (1, 1), constant_values=(uv[0], uv[-1]))
        # save npy
        os.makedirs(os.path.dirname(path_melfile), exist_ok=True)
        np.save(path_melfile, np.ascontiguousarray(mel, dtype=np.float32))
        os.makedirs(os.path.dirname(path_f0file), exist_ok=True)
        np.save(path_f0file, np.asarray(f0, dtype=np.float32))
        os.makedirs(os.path.dirname(path_uvfile), exist_ok=True)
        np.save(path_uvfile, np.asarray(uv, dtype=np.float32))
        return True
    else:
        print('\n[Error] F0 extraction failed: ' + path_srcfile)
        os.makedirs(os.path.dirname(path_skipfile), exist_ok=True)
        shutil.move(path_srcfile, os.path.dirname(path_skipfile))
        print('This file has been moved to ' + path_skipfile)
    return False
    

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
        n_fft,
        n_mel_channels,
        mel_fmin,
        mel_fmax,
        workers=1):
        
    # list files
    filelist =  traverse_dir(
        path_srcdir,
        extension='wav',
        is_pure=True,
        is_sort=True,
        is_ext=True)
    
    # parameters of the extractors, passed to the worker processes
    parameters = {
        'f0_extractor': f0_extractor,
        'f0_min': f0_min,
        'f0_max': f0_max,
        'sampling_rate': sampling_rate,
        'hop_length': hop_length,
        'win_length': win_length,
        'n_fft': n_fft,
        'n_mel_channels': n_mel_channels,
        'mel_fmin': mel_fmin,
        'mel_fmax': mel_fmax}

    print('Preprocess the audio clips in :', path_srcdir)
    
    # multiprocessing with ProcessPoolExecutor
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_init,
        initargs=(parameters, device)
    ) as executor:
        # submit tasks
        future_to_file = {
            executor.submit(
                _process_file,
                file,
                path_srcdir,
                path_meldir,
                path_f0dir,
                path_uvdir,
                path_skipdir): file
            for file in filelist
        }

        # collect results with progress bar
        for future in tqdm(concurrent.futures.as_completed(future_to_file), total=len(filelist)):
            file = future_to_file[future]
            try:
                future.result()
            except Exception as e:
                print(f'\n[Error] Task for {file} generated an exception: {e}')
                
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    
    # parse commands
    cmd = parse_args()
    
    device = cmd.device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    # get number of workers
    workers = cmd.workers
    print(f'Using {workers} worker processes')
    print(f'Using device: {device}')
    
    # load config
    args = utils.load_config(cmd.config)
    f0_extractor = args.data.f0_extractor
    f0_min = args.data.f0_min
    f0_max = args.data.f0_max
    sampling_rate  = args.data.sampling_rate
    hop_length = args.data.block_size
    win_length = args.data.win_length
    n_fft = args.data.n_fft
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
            n_fft,
            n_mel_channels,
            mel_fmin,
            mel_fmax,
            workers=workers)
    
