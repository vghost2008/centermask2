import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import os.path as osp
import mmcv
import numpy as np
import torch
from mmcv.runner import load_checkpoint
from onnxsim import simplify
import sys
sys.path.append("..")
from boedemo.predictor import InferenceModel


try:
    import onnx
    import onnxruntime as rt
except ImportError as e:
    raise ImportError(f'Please install onnx and onnxruntime first. {e}')

try:
    from mmcv.onnx.symbolic import register_extra_symbolics
except ModuleNotFoundError:
    raise NotImplementedError('please update mmcv to version>=1.0.4')


def _convert_batchnorm(module):
    """Convert the syncBNs into normal BN3ds."""
    module_output = module
    if isinstance(module, torch.nn.SyncBatchNorm):
        module_output = torch.nn.BatchNorm3d(module.num_features, module.eps,
                                             module.momentum, module.affine,
                                             module.track_running_stats)
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
            # keep requires_grad unchanged
            module_output.weight.requires_grad = module.weight.requires_grad
            module_output.bias.requires_grad = module.bias.requires_grad
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
    for name, child in module.named_children():
        module_output.add_module(name, _convert_batchnorm(child))
    del module
    return module_output


def pytorch2traced(model,
                 input_shape,
                 opset_version=11,
                 show=False,
                 output_file='tmp.onnx',
                 verify=False):
    """Convert pytorch model to onnx model.

    Args:
        model (:obj:`nn.Module`): The pytorch model to be exported.
        input_shape (tuple[int]): The input tensor shape of the model.
        opset_version (int): Opset version of onnx used. Default: 11.
        show (bool): Determines whether to print the onnx model architecture.
            Default: False.
        output_file (str): Output onnx model name. Default: 'tmp.onnx'.
        verify (bool): Determines whether to verify the onnx model.
            Default: False.
    """
    device = torch.device("cuda:0")
    model.cpu().eval()

    input_tensor = torch.randn(input_shape)
    input_tensor = input_tensor.to(device)

    register_extra_symbolics(opset_version)
    model.to(device)

    model = torch.jit.trace(
        model,
        input_tensor)
    
    model.save(output_file)
    print(f'Successfully exported traced model: {output_file}')


def parse_args():
    pdir_path = osp.dirname(osp.dirname(osp.dirname(__file__)))

    parser = argparse.ArgumentParser(
        description='Convert MMAction2 models to ONNX')
    parser.add_argument('--show', action='store_true', help='show onnx graph')
    parser.add_argument('--output-file', default="model.torch",type=str)
    parser.add_argument('--opset-version', type=int, default=11)
    parser.add_argument(
        '--verify',
        action='store_true',
        help='verify the onnx model output against pytorch output')
    parser.add_argument(
        '--is-localizer',
        action='store_true',
        help='whether it is a localizer')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[640,640,3],
        help='input video size')
    parser.add_argument(
        '--softmax',
        action='store_true',
        default=True,
        help='wheter to add softmax layer at the end of recognizers')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    assert args.opset_version == 11, 'only supports opset 11 now'
    model = InferenceModel()


    # conver model to onnx file
    pytorch2traced(
        model,
        args.shape,
        opset_version=args.opset_version,
        show=args.show,
        output_file=args.output_file,
        verify=args.verify)
