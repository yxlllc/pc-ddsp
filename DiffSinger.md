# Use DDSP Vocoders in DiffSinger (OpenVPI version)
Suppose you have already trained a model called `exp/combsub-test/model_100000.pt` using the code in this repository, run
```bash
python export.py -m exp/combsub-test/model_100000.pt --traced
```
This will create a `.jit`  format model file in the same directory.

Then, move this `.jit` model file and the `config.yaml` together to the `checkpoints/ddsp`  directory of the [**DiffSinger**](https://github.com/openvpi/DiffSinger) repository.

Finally, edit the  [**`configs/acoustic/nomidi.yaml`**](https://github.com/openvpi/DiffSinger/blob/refactor/configs/acoustic/nomidi.yaml) file in the [**DiffSinger**](https://github.com/openvpi/DiffSinger)  repository to enable the DDSP vocoder. the details are:
1. Set the `vocoder` option to `DDSP`.
2. Set the `vocoder_ckpt` option to the path of the `.jit` model.  An example may be `checkpoints/ddsp/model_100000-traced-torch1.9.1.jit`
3. Check whether other mel related parameters match the parameters in the `checkpoints/ddsp/config.yaml` file.  For the details, the `audio_sample_rate`,`audio_num_mel_bins`,`hop_size`,`fft_size`,`win_size`,`fmin` and  `fmax` in  the[**`configs/acoustic/nomidi.yaml`**](https://github.com/openvpi/DiffSinger/blob/refactor/configs/acoustic/nomidi.yaml) need to match `sampling_rate`, `n_mels`, `block_size`, `n_fft`, `win_length`,`mel_fmin` and `mel_fmax` in the `checkpoints/ddsp/config.yaml`, respectively.

After doing all this, [**DiffSinger**](https://github.com/openvpi/DiffSinger)'s default NSF-HiFiGAN vocoder has been replaced by your own trained DDSP vocoder, and you can perform preprocessing, training or inference normally.
