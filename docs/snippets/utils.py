# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import tempfile
import numpy as np
from sys import platform

import openvino as ov
import openvino.runtime.opset12 as ops

import ngraph as ng
from ngraph.impl import Function
import openvino.inference_engine as ie


def get_dynamic_model():
    param = ops.parameter(ov.PartialShape([-1, -1]), name="input")
    return ov.Model(ops.relu(param), [param])


def get_model(input_shape = None, input_dtype=np.float32) -> ov.Model:
    if input_shape is None:
        input_shape = [1, 3, 32, 32]
    param = ops.parameter(input_shape, input_dtype, name="input")
    relu = ops.relu(param, name="relu")
    relu.output(0).get_tensor().set_names({"result"})
    model = ov.Model([relu], [param], "test_model")

    assert model is not None
    return model


def get_ngraph_model(input_shape = None, input_dtype=np.float32):
    if input_shape is None:
        input_shape = [1, 3, 32, 32]
    param = ng.opset11.parameter(input_shape, input_dtype, name="data")
    relu = ng.opset11.relu(param, name="relu")
    func = Function([relu], [param], "test_model")
    caps = ng.Function.to_capsule(func)
    net = ie.IENetwork(caps)

    assert net is not None
    return net


def get_image(shape = (1, 3, 32, 32), dtype = "float32"):
    np.random.seed(42)
    return np.random.rand(*shape).astype(dtype)


def get_path_to_image():
    path_to_img = tempfile.NamedTemporaryFile(suffix="_image.bmp").name
    import cv2 as cv
    cv.imread(path_to_img, get_image())
    return path_to_img


def get_path_to_extension_library():
    library_path=""
    if platform == "win32":
        library_path="openvino_template_extension.dll"
    else:
        library_path="libopenvino_template_extension.so"
    return library_path


def get_path_to_model(input_shape = None, is_old_api=False):
    if input_shape is None:
        input_shape = [1, 3, 32, 32]
    path_to_xml = tempfile.NamedTemporaryFile(suffix="_model.xml").name
    if is_old_api:
        net = get_ngraph_model(input_shape, input_dtype=np.int64)
        net.serialize(path_to_xml)
    else:
        model = get_model(input_shape)
        ov.serialize(model, path_to_xml)
    return path_to_xml


def get_temp_dir():
    temp_dir = tempfile.TemporaryDirectory()
    return temp_dir

