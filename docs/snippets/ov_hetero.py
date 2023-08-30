import openvino as ov
from snippets import get_model

model = get_model()
core = ov.Core()

#! [set_manual_affinities]
for op in model.get_ops():
    rt_info = op.get_rt_info()
    rt_info["affinity"] = "CPU"
#! [set_manual_affinities]

#! [fix_automatic_affinities]
# This example demonstrates how to perform default affinity initialization and then
# correct affinity manually for some layers
device = "HETERO:GPU,CPU"

# query_model result contains mapping of supported operations to devices
supported_ops = core.query_model(model, device)

# update default affinities manually for specific operations
supported_ops["operation_name"] = "CPU"

# set affinities to a model
for node in model.get_ops():
    affinity = supported_ops[node.get_friendly_name()]
    node.get_rt_info()["affinity"] = "CPU"

# load model with manually set affinities
compiled_model = core.compile_model(model, device)
#! [fix_automatic_affinities]

#! [compile_model]
compiled_model = core.compile_model(model, device_name="HETERO:GPU,CPU")
# device priorities via configuration property
compiled_model = core.compile_model(
    model, device_name="HETERO", config={ov.properties.device.priorities(): "GPU,CPU"}
)
#! [compile_model]

#! [configure_fallback_devices]
core.set_property("HETERO", {ov.properties.device.priorities(): "GPU,CPU"})
core.set_property("GPU", {ov.properties.enable_profiling(): True})
core.set_property("CPU", {ov.properties.hint.inference_precision(): ov.Type.f32})
compiled_model = core.compile_model(model=model, device_name="HETERO")
#! [configure_fallback_devices]
