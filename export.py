import argparse
import os.path

import torch

from ddsp.vocoder import load_model


class DDSPWrapper(torch.nn.Module):
    def __init__(self, module, device):
        super().__init__()
        self.model = module
        self.to(device)

    def forward(self, mel, f0):
        f0 = f0[..., None]
        signal, _, (s_h, s_n) = self.model(mel, f0)
        return signal, s_h, s_n


def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser(
        description='Export model to standalone PyTorch traced module or ONNX format'
    )
    parser.add_argument(
        '-m',
        '--model_path',
        type=str,
        required=True,
        help='path to model file'
    )
    parser.add_argument(
        '--traced',
        required=False,
        action='store_true',
        help='export to traced module format'
    )
    parser.add_argument(
        '--onnx',
        required=False,
        action='store_true',
        help='export to ONNX format'
    )
    cmd = parser.parse_args(args=args, namespace=namespace)
    if not cmd.traced and not cmd.onnx:
        parser.error('either --traced or --onnx should be specified.')
    return cmd


def main():
    device = 'cpu'
    # parse commands
    cmd = parse_args()

    # load model
    model, args = load_model(cmd.model_path, device=device)
    #model = DDSPWrapper(model, device)

    # extract model dirname and filename
    directory = os.path.dirname(os.path.abspath(cmd.model_path))
    name = os.path.basename(cmd.model_path).rsplit('.', maxsplit=1)[0]

    # load input
    n_mel_channels = args.data.n_mels
    n_frames = 10
    mel = torch.randn((1, n_frames, n_mel_channels), dtype=torch.float32, device=device)
    f0 = torch.FloatTensor([[440.] * n_frames]).to(device)
    f0 = f0[..., None]
    
    # export model
    with torch.no_grad():
        if cmd.traced:
            torch_version = torch.version.__version__.rsplit('+', maxsplit=1)[0]
            export_path = os.path.join(directory, f'{name}-traced-torch{torch_version}.jit')
            print(f' [Tracing] {cmd.model_path} => {export_path}')
            model = torch.jit.trace(
                model,
                (
                    mel,
                    f0
                ),
                check_trace=False
            )
            torch.jit.save(model, export_path)

        if cmd.onnx:
            raise NotImplementedError('Exporting to ONNX format is not supported yet.')


if __name__ == '__main__':
    main()
