import os
import numpy as np
import yaml
import torch
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel_fn
from .mel2control import Mel2Control
from .core import frequency_filter, mean_filter, upsample

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
    print(' [Loading] ' + model_path)
    if model_path.split('.')[-1] == 'jit':
        model = torch.jit.load(model_path, map_location=torch.device(device))
    else:
        if args.model.type == 'Sins':
            model = Sins(
                sampling_rate=args.data.sampling_rate,
                block_size=args.data.block_size,
                win_length=args.model.win_length,
                use_mean_filter=args.model.use_mean_filter,              
                n_harmonics=args.model.n_harmonics,
                n_mag_noise=args.model.n_mag_noise,
                n_mels=args.data.n_mels)
    
        elif args.model.type == 'CombSub':
            model = CombSub(
                sampling_rate=args.data.sampling_rate,
                block_size=args.data.block_size,
                win_length=args.model.win_length,
                use_mean_filter=args.model.use_mean_filter,               
                n_mag_harmonic=args.model.n_mag_harmonic,
                n_mag_noise=args.model.n_mag_noise,
                n_mels=args.data.n_mels)
                        
        else:
            raise ValueError(f" [x] Unknown Model: {args.model.type}")
        model.to(device)
        ckpt = torch.load(model_path, map_location=torch.device(device))
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
        self.hann_window = {}
        mel_basis = librosa_mel_fn(
            sr=sampling_rate,
            n_fft=n_fft, 
            n_mels=n_mel_channels, 
            fmin=mel_fmin, 
            fmax=mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels
        self.clamp = clamp

    def forward(self, audio, keyshift=0, speed=1):
        '''
              audio: B x C x T
        log_mel_spec: B x T_ x C x n_mel 
        '''
        factor = 2 ** (keyshift / 12)       
        n_fft_new = int(np.round(self.n_fft * factor))
        win_length_new = int(np.round(self.win_length * factor))
        hop_length_new = int(np.round(self.hop_length * speed))
        
        keyshift_key = str(keyshift)+'_'+str(audio.device)
        if keyshift_key not in self.hann_window:
            self.hann_window[keyshift_key] = torch.hann_window(win_length_new).to(audio.device)
            
        B, C, T = audio.shape
        audio = audio.reshape(B * C, T)
        fft = torch.stft(
            audio,
            n_fft=n_fft_new,
            hop_length=hop_length_new,
            win_length=win_length_new,
            window=self.hann_window[keyshift_key],
            center=True,
            return_complex=True)
        magnitude = torch.sqrt(fft.real.pow(2) + fft.imag.pow(2))
        
        if keyshift != 0:
            size = self.n_fft // 2 + 1
            resize = magnitude.size(1)
            if resize < size:
                magnitude = F.pad(magnitude, (0, 0, 0, size-resize))
            magnitude = magnitude[:, :size, :] * self.win_length / win_length_new
            
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
            win_length,
            use_mean_filter,
            n_harmonics,
            n_mag_noise,
            n_mels=80):
        super().__init__()

        print(' [DDSP Model] Sinusoids Additive Synthesiser')

        # params
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))
        self.register_buffer("win_length", torch.tensor(win_length))
        self.register_buffer("window", torch.hann_window(win_length))
        # Mel2Control
        split_map = {
            'amplitudes': n_harmonics,
            'harmonic_phase': win_length // 2 + 1,
            'noise_magnitude': n_mag_noise,
            'noise_phase': n_mag_noise,
        }
        self.mel2ctrl = Mel2Control(n_mels, block_size, split_map)
        # mean filter kernel size
        if use_mean_filter:
            self.mean_kernel_size = win_length // block_size
        else:
            self.mean_kernel_size = 1
    
    def fast_phase_gen(self, f0_frames):
        n = torch.arange(self.block_size, device=f0_frames.device)
        s0 = f0_frames / self.sampling_rate
        ds0 = F.pad(s0[:, 1:, :] - s0[:, :-1, :], (0, 0, 0, 1))
        rad = s0 * (n + 1) + 0.5 * ds0 * n * (n + 1) / self.block_size
        rad2 = torch.fmod(rad[..., -1:].float() + 0.5, 1.0) - 0.5
        rad_acc = rad2.cumsum(dim=1).fmod(1.0).to(f0_frames)
        rad += F.pad(rad_acc[:, :-1, :], (0, 0, 1, 0))
        phase = 2 * np.pi * rad.reshape(f0_frames.shape[0], -1, 1)
        return phase
    
    def forward(self, 
            mel_frames, 
            f0_frames, 
            output_f0_frames=None,
            infer=True, 
            max_upsample_dim=32):
        '''
            mel_frames: B x n_frames x n_mels
            f0_frames: B x n_frames x 1
        '''
        # exciter phase
        phase = self.fast_phase_gen(f0_frames)
        
        # sinusoid exciter signal
        sinusoid = torch.sin(phase).squeeze(-1)
        sinusoid_frames = sinusoid.unfold(1, self.block_size, self.block_size)
        
        # noise exciter signal
        noise = torch.randn_like(sinusoid)
        noise_frames = noise.unfold(1, self.block_size, self.block_size)
        
        # parameter prediction
        ctrls = self.mel2ctrl(mel_frames, sinusoid_frames, noise_frames)
        if self.mean_kernel_size > 1:
            ctrls['amplitudes'] = mean_filter(ctrls['amplitudes'], self.mean_kernel_size)
            ctrls['harmonic_phase'] = mean_filter(ctrls['harmonic_phase'], self.mean_kernel_size)
            
        src_allpass = torch.exp(1.j * np.pi * ctrls['harmonic_phase'])
        src_allpass = torch.cat((src_allpass, src_allpass[:,-1:,:]), 1)
        amplitudes_frames = torch.exp(ctrls['amplitudes'])/ 128
        noise_param = torch.exp(ctrls['noise_magnitude'] + 1.j * np.pi * ctrls['noise_phase']) / 128
        
        # harmonic additive synthesis
        if infer and output_f0_frames is not None:
            f0_frames = output_f0_frames
            phase = self.fast_phase_gen(output_f0_frames)  
        n_harmonic = amplitudes_frames.shape[-1]
        level_harmonic = torch.arange(1, n_harmonic + 1, device=phase.device)
        mask = (f0_frames * level_harmonic < self.sampling_rate / 2).float() + 1e-7
        amplitudes_frames *= mask
        sinusoids = 0.
        for n in range(( n_harmonic - 1) // max_upsample_dim + 1):
            start = n * max_upsample_dim
            end = (n + 1) * max_upsample_dim
            phases = phase * level_harmonic[start:end]
            amplitudes = upsample(amplitudes_frames[:,:,start:end], self.block_size)
            sinusoids += (torch.sin(phases) * amplitudes).sum(-1)
        
        # harmonic part filter (all pass)
        harmonic_spec = torch.stft(
                            sinusoids,
                            n_fft = self.win_length,
                            win_length = self.win_length,
                            hop_length = self.block_size,
                            window = self.window,
                            center = True,
                            return_complex = True)
        harmonic_spec = harmonic_spec * src_allpass.permute(0, 2, 1)
        harmonic = torch.istft(
                        harmonic_spec,
                        n_fft = self.win_length,
                        win_length = self.win_length,
                        hop_length = self.block_size,
                        window = self.window,
                        center = True)
                        
        # noise part filter (using constant-windowed LTV-FIR) 
        noise = frequency_filter(
                        noise,
                        noise_param)
                        
        signal = harmonic + noise

        return signal, sinusoids, (harmonic, noise)
        
        
class CombSub(torch.nn.Module):
    def __init__(self, 
            sampling_rate,
            block_size,
            win_length,
            use_mean_filter,
            n_mag_harmonic,
            n_mag_noise,
            n_mels=80):
        super().__init__()

        print(' [DDSP Model] Combtooth Subtractive Synthesiser')
        # params
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))
        self.register_buffer("win_length", torch.tensor(win_length))
        self.register_buffer("window", torch.hann_window(win_length))
        # Mel2Control
        split_map = {            
            'harmonic_magnitude': n_mag_harmonic,
            'harmonic_phase': win_length // 2 + 1,
            'noise_magnitude': n_mag_noise,
            'noise_phase': n_mag_noise,
        }
        self.mel2ctrl = Mel2Control(n_mels, block_size, split_map)
        # mean filter kernel size
        if use_mean_filter:
            self.mean_kernel_size = win_length // block_size
        else:
            self.mean_kernel_size = 1
    
    def fast_source_gen(self, f0_frames):
        n = torch.arange(self.block_size, device=f0_frames.device)
        s0 = f0_frames / self.sampling_rate
        ds0 = F.pad(s0[:, 1:, :] - s0[:, :-1, :], (0, 0, 0, 1))
        rad = s0 * (n + 1) + 0.5 * ds0 * n * (n + 1) / self.block_size
        s0 = s0 + ds0 * n / self.block_size
        rad2 = torch.fmod(rad[..., -1:].float() + 0.5, 1.0) - 0.5
        rad_acc = rad2.cumsum(dim=1).fmod(1.0).to(f0_frames)
        rad += F.pad(rad_acc[:, :-1, :], (0, 0, 1, 0))
        rad -= torch.round(rad)
        combtooth = torch.sinc(rad / (s0 + 1e-5)).reshape(f0_frames.shape[0], -1)
        return combtooth
        
    def forward(self, 
            mel_frames, 
            f0_frames,
            output_f0_frames=None,
            infer=True, 
            **kwargs):
        '''
            mel_frames: B x n_frames x n_mels
            f0_frames: B x n_frames x 1
        '''
                
        # combtooth exciter signal
        combtooth = self.fast_source_gen(f0_frames)
        combtooth_frames = combtooth.unfold(1, self.block_size, self.block_size)
        
        # noise exciter signal
        noise = torch.randn_like(combtooth)
        noise_frames = noise.unfold(1, self.block_size, self.block_size)
        
        # parameter prediction
        ctrls = self.mel2ctrl(mel_frames, combtooth_frames, noise_frames)
        if self.mean_kernel_size > 1:
            ctrls['harmonic_magnitude'] = mean_filter(ctrls['harmonic_magnitude'], self.mean_kernel_size)
            ctrls['harmonic_phase'] = mean_filter(ctrls['harmonic_phase'], self.mean_kernel_size)
        
        src_allpass = torch.exp(1.j * np.pi * ctrls['harmonic_phase'])
        src_allpass = torch.cat((src_allpass, src_allpass[:,-1:,:]), 1)
        src_param = torch.exp(ctrls['harmonic_magnitude'])
        noise_param = torch.exp(ctrls['noise_magnitude'] + 1.j * np.pi * ctrls['noise_phase']) / 128
        
        # harmonic part filter (using dynamic-windowed LTV-FIR)
        if infer and output_f0_frames is not None:
            f0_frames = output_f0_frames
            combtooth = self.fast_source_gen(output_f0_frames)
        harmonic = frequency_filter(
                        combtooth,
                        torch.complex(src_param, torch.zeros_like(src_param)),
                        hann_window = True,
                        half_width_frames = 1.5 * self.sampling_rate / (f0_frames + 1e-3))
               
        # harmonic part filter (all pass)
        harmonic_spec = torch.stft(
                            harmonic,
                            n_fft = self.win_length,
                            win_length = self.win_length,
                            hop_length = self.block_size,
                            window = self.window,
                            center = True,
                            return_complex = True)
        harmonic_spec = harmonic_spec * src_allpass.permute(0, 2, 1)
        harmonic = torch.istft(
                        harmonic_spec,
                        n_fft = self.win_length,
                        win_length = self.win_length,
                        hop_length = self.block_size,
                        window = self.window,
                        center = True)
                        
        # noise part filter (using constant-windowed LTV-FIR) 
        noise = frequency_filter(
                        noise,
                        noise_param)
                       
        signal = harmonic + noise

        return signal, combtooth, (harmonic, noise)