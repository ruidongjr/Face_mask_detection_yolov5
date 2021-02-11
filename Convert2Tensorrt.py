# import onnx
#
# ONNX_FILE_PATH = 'runs/train/exp2/weights/best.onnx'
# onnx_model = onnx.load(ONNX_FILE_PATH)
# onnx.checker.check_model(onnx_model)

import os, sys
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
from pycuda.tools import make_default_context
from tensorflow.keras.applications.inception_v3 import preprocess_input


class JetsonInferenceEngine:
    """
    Copy the usual 'model' structure from tf.session to keep the use of Tiliter's code structure.
    """

    def __init__(self, jet_file):
        print("Creating structure Jetson Inference", file=sys.stderr)
        self.TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        cuda.init()
        self.ctx = make_default_context()
        with trt.Runtime(self.TRT_LOGGER) as runtime:
            with open(jet_file, "rb") as f:
                self.engine = runtime.deserialize_cuda_engine(f.read())
        self.classes = []
        self.predict = None
        print("Jetson Inference structure created", file=sys.stderr)

        self.h_input = cuda.pagelocked_empty(trt.volume(self.engine.get_binding_shape(0)), dtype=np.float32)
        self.h_output = cuda.pagelocked_empty(trt.volume(self.engine.get_binding_shape(1)), dtype=np.float32)
        # Allocate device memory for inputs and outputs.
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)
        self.num_output = 1
        self.output_names = ['output']

        # Create a stream in which to copy inputs/outputs and run inference.
        self.cuda_stream = cuda.Stream()
        self.context = self.engine.create_execution_context()

    def classify_with_image(self, image):
        try:
            self.ctx.push()
            image = preprocess_input(image)
            img = image.reshape(self.h_input.shape).astype(np.float32)
            # Transfer input data to the GPU.
            cuda.memcpy_htod_async(self.d_input, img, self.cuda_stream)
            # Run inference.
            self.context.execute_async(bindings=[int(self.d_input), int(self.d_output)],
                                       stream_handle=self.cuda_stream.handle)
            # Transfer predictions back from the GPU.
            cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.cuda_stream)
            output_jetson = [self.h_output, None, None]

            # Synchronize the stream
            self.cuda_stream.synchronize()
            self.ctx.pop()

        except:
            print("Oops!", sys.exc_info()[0], "occured.", file=sys.stderr)

        return output_jetson

    def __del__(self):
        del self.engine
        self.ctx.detach()
        print("Engine deleted..", file=sys.stderr)


def fromOnnx2TensorRtEngine(model_name=".onnx"):
    # Create the cuda engine, directly linked to the device's architecture
    print("This part should be run only on the Jetson Nano and will probably take a while", file=sys.stderr)
    onnx_file = os.path.join(model_name + '.onnx')
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    b = False
    with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_network() as network, \
            trt.OnnxParser(network, TRT_LOGGER) as parser:

        builder.max_workspace_size = 1 << 30  # 1GB
        builder.max_batch_size = 1
        builder.fp16_mode = True
        if not os.path.exists(onnx_file):
            print('ONNX file {} not found!'.format(onnx_file), file=sys.stderr)
            exit(0)
        print('Loading ONNX file from path {}'.format(onnx_file), file=sys.stderr)
        with open(onnx_file, 'rb') as model:
            print('Beginning ONNX file parsing', file=sys.stderr)
            b = parser.parse(model.read())  # if the parsing was completed b is true
        if b:
            print('Completed parsing of ONNX file', file=sys.stderr)
            print('Building an engine from file {}; this may take a while'.format(onnx_file), file=sys.stderr)
            engine = builder.build_cuda_engine(network)
            del parser
            if engine:
                print('Completed creating Engine', file=sys.stderr)
                with open(os.path.join(model_name + '.jet'), "wb") as f:
                    f.write(engine.serialize())
                    print("Engine saved as: {}".format(os.path.join(model_name + '.jet')), file=sys.stderr)
            else:
                print('Error building engine', file=sys.stderr)
                exit(1)
        else:
            print('Number of errors: {}'.format(parser.num_errors), file=sys.stderr)
            error = parser.get_error(0)  # if it gets mnore than one error this have to be changed
            del parser
            desc = error.desc()
            line = error.line()
            code = error.code()
            print('Description of the error: {}'.format(desc), file=sys.stderr)
            print('Line where the error occurred: {}'.format(line), file=sys.stderr)
            print('Error code: {}'.format(code), file=sys.stderr)
            print("Model was not parsed successfully", file=sys.stderr)
            exit(0)


