# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import openvino as ov
import openvino.runtime.properties as props
import openvino.runtime.properties.hint as hints

from utils import get_model

# [get_available_devices]
core = ov.Core()
available_devices = core.available_devices
# [get_available_devices]

# [hetero_priorities]
device_priorites = core.get_property("HETERO", props.device.priorities())
# [hetero_priorities]

# [cpu_device_name]
cpu_device_name = core.get_property("CPU", props.device.full_name())
# [cpu_device_name]

model = get_model()
# [compile_model_with_property]
config = {hints.performance_mode(): hints.PerformanceMode.THROUGHPUT,
          hints.inference_precision: ov.Type.f32}
compiled_model = core.compile_model(model, "CPU", config)
# [compile_model_with_property]

# [optimal_number_of_infer_requests]
compiled_model = core.compile_model(model, "CPU")
nireq = compiled_model.get_property(props.optimal_number_of_infer_requests())
# [optimal_number_of_infer_requests]


# [core_set_property_then_compile]
# latency hint is a default for CPU
core.set_property("CPU", {hints.performance_mode(): hints.PerformanceMode.LATENCY})
# compiled with latency configuration hint
compiled_model_latency = core.compile_model(model, "CPU")
# compiled with overriden performance hint value
config = {hints.performance_mode(): hints.PerformanceMode.THROUGHPUT}
compiled_model_thrp = core.compile_model(model, "CPU", config)
# [core_set_property_then_compile]


# [inference_num_threads]
compiled_model = core.compile_model(model, "CPU")
nthreads = compiled_model.get_property(props.inference_num_threads)
# [inference_num_threads]

# [multi_device]
config = {props.device.priorities(): "CPU,GPU"}
compiled_model = core.compile_model(model, "MULTI", config)
# change the order of priorities
compiled_model.set_property({props.device.priorities(): "GPU,CPU"})
# [multi_device]
