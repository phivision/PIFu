import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from lib.options import BaseOptions
from lib.train_util import *
from apps.eval import Evaluator

from collections import OrderedDict
import glob
import tqdm
import onnx
import coremltools as ct

# get options
opt = BaseOptions().parse()
# default size
DEFAULT_SIZE = 512, 512


def load_model(model_path, map_location):
    state_dict = torch.load(model_path, map_location=map_location)
    new_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove 'module' for parallel data models
        new_dict[name] = v
    return new_dict


class Converter(Evaluator):
    def __init__(self, option, projection_mode='orthogonal'):
        super().__init__(option, projection_mode=projection_mode)

    def export_to_onnx(self, export_onnx_path, sample_input, show_graph=False):
        net_list = []
        if self.netG:
            shape_model_file = "netG.onnx"
            export_g_path = os.path.join(export_onnx_path, shape_model_file)
            torch.onnx.export(self.netG, sample_input, export_g_path, export_params=True, opset_version=11)
            net_list.append(export_g_path)
        if self.netC:
            color_model_file = "netC.onnx"
            export_c_path = os.path.join(export_onnx_path, color_model_file)
            torch.onnx.export(self.netG, sample_input, export_c_path, export_params=True, opset_version=11)
            net_list.append(export_c_path)
        if show_graph and net_list:
            for net_path in net_list:
                # Load back in ONNX model
                onnx_model = onnx.load(net_path)
                # Check that the IR is well formed
                onnx.checker.check_model(onnx_model)
                # Print a human readable representation of the graph
                graph = onnx.helper.printable_graph(onnx_model.graph)
                print(graph)

    def export_to_coreml(self, export_coreml_path, sample_input):
        if self.netG:
            shape_model_file = "netG.mlmodel"
            export_g_path = os.path.join(export_coreml_path, shape_model_file)
            traced_model = torch.jit.trace(self.netG, example_inputs=sample_input)
            core_model = ct.convert(traced_model, inputs=[ct.TensorType(name="images", shape=sample_input[0].shape),
                                                          ct.TensorType(name="points", shape=sample_input[1].shape),
                                                          ct.TensorType(name="calibs", shape=sample_input[2].shape),
                                                          ct.TensorType(name="labels", shape=sample_input[3].shape)])
            core_model.save(export_g_path)
        if self.netC:
            color_model_file = "netC.mlmodel"
            export_c_path = os.path.join(export_coreml_path, color_model_file)
            traced_model = torch.jit.trace(self.netC, example_inputs=sample_input)
            core_model = ct.convert(traced_model)
            core_model.save(export_c_path)


if __name__ == '__main__':
    converter = Converter(opt)
    sample_image = [f for f in glob.glob(os.path.join(opt.test_folder_path, '*'))
                    if ('png' in f or 'jpg' in f) and (not 'mask' in f) and (not 'resized' in f)][0]
    sample_points = np.random.uniform(-1.0, 1.0, 3*opt.num_sample_inout).reshape((1, 3, opt.num_sample_inout))
    sample_labels = np.random.uniform(0.0, 1.0, opt.num_sample_inout).reshape((1, 1, opt.num_sample_inout))
    sample_points = torch.from_numpy(sample_points).float()
    sample_labels = torch.from_numpy(sample_labels).float()
    print(sample_image)
    data = converter.load_image(sample_image)
    sample_inputs = (data['img'].to(device=converter.cuda),
                     sample_points.to(device=converter.cuda),
                     data['calib'].to(device=converter.cuda),
                     sample_labels.to(device=converter.cuda))
    # converter.export_to_coreml(opt.export_coreml_path, sample_inputs)
    converter.export_to_onnx(opt.export_onnx_path, sample_inputs)

