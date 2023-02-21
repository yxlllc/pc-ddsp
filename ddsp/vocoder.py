import os
import numpy as np
import yaml
import torch
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel_fn
from .mel2control import Mel2Control
from .core import frequency_filter, upsample, remove_above_fmax

class DotDict(dict):
    def __getattr__(*args):         
        val = dict.get(*args)         
        return DotDict(val) if type(val) is dict else val   

    __setattr__ = dict.__setitem__    
    __delattr__ = dict.__delitem__
    
def load_model(
        model_path,
        device='cpu'):
    config_file = os.path.join(os.path.split(model_path)[0], 'config.yaml')
    with open(config_file, "r") as config:
        args = yaml.safe_load(config)
    args = DotDict(args)
    
    # load model
    model = None

    if args.model.type == 'Sins':
        model = Sins(
            sampling_rate=args.data.sampling_rate,
            block_size=args.data.block_size,
            n_harmonics=args.model.n_harmonics,
            n_mag_allpass=args.model.n_mag_allpass,
            n_mag_noise=args.model.n_mag_noise,
            n_mels=args.data.n_mels)
    
    elif args.model.type == 'CombSub':
        model = CombSub(
            sampling_rate=args.data.sampling_rate,
            block_size=args.data.block_size,
            n_mag_allpass=args.model.n_mag_allpass,
            n_mag_harmonic=args.model.n_mag_harmonic,
            n_mag_noise=args.model.n_mag_noise,
            n_mels=args.data.n_mels)
            
    else:
        raise ValueError(f" [x] Unknown Model: {args.model.type}")
    
    print(' [Loading] ' + model_path)
    ckpt = torch.load(model_path, map_location=torch.device(device))
    model.to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, args
    
class Audio2Mel(torch.nn.Module):
    def __init__(
        self,
        hop_length,
        sampling_rate,
        n_mel_channels,
        win_length,
        n_fft=None,
        mel_fmin=0,
        mel_fmax=None,
        clamp = 1e-5
    ):
        super().__init__()
        n_fft = win_length if n_fft is None else n_fft
        window = torch.hann_window(win_length).float()
        mel_basis = librosa_mel_fn(
            sr=sampling_rate,
            n_fft=n_fft, 
            n_mels=n_mel_channels, 
            fmin=mel_fmin, 
            fmax=mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer("window", window)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels
        self.clamp = clamp

    def forward(self, audio):
        '''
              audio: B x C x T
        og_mel_spec: B x T_ x C x n_mel 
        '''
        B, C, T = audio.shape
        audio = audio.reshape(B * C, T)
        fft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            return_complex=False)
        real_part, imag_part = fft.unbind(-1)
        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        mel_output = torch.matmul(self.mel_basis, magnitude)
        log_mel_spec = torch.log10(torch.clamp(mel_output, min=self.clamp))

        # log_mel_spec: B x C, M, T
        T_ = log_mel_spec.shape[-1]
        log_mel_spec = log_mel_spec.reshape(B, C, self.n_mel_channels ,T_)
        log_mel_spec = log_mel_spec.permute(0, 3, 1, 2)

        # print('og_mel_spec:', log_mel_spec.shape)
        log_mel_spec = log_mel_spec.squeeze(2) # mono
        return log_mel_spec
        
class Sins(torch.nn.Module):
    def __init__(self, 
            sampling_rate,
            block_size,
            n_harmonics,
            n_mag_allpass,
            n_mag_noise,
            n_mels=80):
        super().__init__()

        print(' [DDSP Model] Sinusoids Additive Synthesiser')

        # params
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))
        # Mel2Control
        split_map = {
            'amplitudes': n_harmonics,
            'group_delay': n_mag_allpass,
            'noise_magnitude': n_mag_noise,
        }
        self.mel2ctrl = Mel2Control(n_mels, split_map)

    def forward(self, mel_frames, f0_frames, initial_phase=None, max_upsample_dim=32):
        '''
            mel_frames: B x n_frames x n_mels
            f0_frames: B x n_frames x 1
        '''
        # exciter phase
        f0 = upsample(f0_frames, self.block_size)
        if initial_phase is None:
            initial_phase = torch.zeros(f0.shape[0], 1, 1).to(f0)
            
        x = torch.cumsum(f0.double() / self.sampling_rate, axis=1) + initial_phase.double() / 2 / np.pi
        x = x - torch.round(x)
        x = x.float()
        
        phase = 2 * np.pi * x
        phase_frames = phase[:, ::self.block_size, :]
        
        # parameter prediction
        ctrls = self.mel2ctrl(mel_frames, phase_frames)
        
        amplitudes_frames = torch.exp(ctrls['amplitudes'])/ 128
        group_delay = np.pi * torch.tanh(ctrls['group_delay'])
        noise_param = torch.exp(ctrls['noise_magnitude']) / 128
        
        # sinusoids exciter signal 
        amplitudes_frames = remove_above_fmax(amplitudes_frames, f0_frames, self.sampling_rate / 2, level_start = 1)
        n_harmonic = amplitudes_frames.shape[-1]
        level_harmonic = torch.arange(1, n_harmonic + 1).to(phase)
        sinusoids = 0.
        for n in range(( n_harmonic - 1) // max_upsample_dim + 1):
            start = n * max_upsample_dim
            end = (n + 1) * max_upsample_dim
            phases = phase * level_harmonic[start:end]
            amplitudes = upsample(amplitudes_frames[:,:,start:end], self.block_size)
            sinusoids += (torch.sin(phases) * amplitudes).sum(-1)
        
        # harmonic part filter (apply group-delay)
        harmonic = frequency_filter(
                        sinusoids,
                        torch.exp(1.j * torch.cumsum(group_delay, axis = -1)),
                        hann_window = False)
                        
        # noise part filter 
        noise = torch.rand_like(harmonic).to(noise_param) * 2 - 1
        noise = frequency_filter(
                        noise,
                        torch.complex(noise_param, torch.zeros_like(noise_param)),
                        hann_window = True)
                        
        signal = harmonic + noise

        return signal, phase, (harmonic, noise) #, (noise_param, noise_param)

class CombSub(torch.nn.Module):
    def __init__(self, 
            sampling_rate,
            block_size,
            n_mag_allpass,
            n_mag_harmonic,
            n_mag_noise,
            n_mels=80):
        super().__init__()

        print(' [DDSP Model] Combtooth Subtractive Synthesiser')
        # params
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))
        # Mel2Control
        split_map = {
            'group_delay': n_mag_allpass,
            'harmonic_magnitude': n_mag_harmonic, 
            'noise_magnitude': n_mag_noise
        }
        self.mel2ctrl = Mel2Control(n_mels, split_map)

    def forward(self, mel_frames, f0_frames, initial_phase=None, **kwargs):
        '''
            mel_frames: B x n_frames x n_mels
            f0_frames: B x n_frames x 1
        '''
        # exciter phase
        f0 = upsample(f0_frames, self.block_size)
        if initial_phase is None:
            initial_phase = torch.zeros(f0.shape[0], 1, 1).to(f0)
            
        x = torch.cumsum(f0.double() / self.sampling_rate, axis=1) + initial_phase.double() / 2 / np.pi
        x = x - torch.round(x)
        x = x.float()
        
        phase_frames = 2 * np.pi * x[:, ::self.block_size, :]
        
        # parameter prediction
        ctrls = self.mel2ctrl(mel_frames, phase_frames)
        
        group_delay = np.pi * torch.tanh(ctrls['group_delay'])
        src_param = torch.exp(ctrls['harmonic_magnitude'])
        noise_param = torch.exp(ctrls['noise_magnitude']) / 128
        
        # combtooth exciter signal 
        combtooth = torch.sinc(self.sampling_rate * x / (f0 + 1e-3))
        combtooth = combtooth.squeeze(-1)
        
        # harmonic part filter (using dynamic-windowed LTV-FIR, with group-delay prediction)
        harmonic = frequency_filter(
                        combtooth,
                        torch.exp(1.j * torch.cumsum(group_delay, axis = -1)),
                        hann_window = False)
        harmonic = frequency_filter(
                        harmonic,
                        torch.complex(src_param, torch.zeros_like(src_param)),
                        hann_window = True,
                        half_width_frames = 1.5 * self.sampling_rate / (f0_frames + 1e-3))
                  
        # noise part filter (using constant-windowed LTV-FIR, without group-delay)
        noise = torch.rand_like(harmonic).to(noise_param) * 2 - 1
        noise = frequency_filter(
                        noise,
                        torch.complex(noise_param, torch.zeros_like(noise_param)),
                        hann_window = True)
                        
        signal = harmonic + noise

        return signal, phase_frames, (harmonic, noise)