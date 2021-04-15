import tensorflow as tf
import tensorflow.keras.backend as K
from efficientnet import model
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph

def get_flops(model, batch_size=None):
    if batch_size is None:
        batch_size = 1

    real_model = tf.function(model).get_concrete_function(tf.TensorSpec([batch_size] + model.inputs[0].shape[1:], model.inputs[0].dtype))
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(real_model)

    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.compat.v1.profiler.profile(graph=frozen_func.graph,
                                            run_meta=run_meta, cmd='op', options=opts)
    return flops.total_float_ops

if __name__ == '__main__':
    netRes = model.EfficientNetExperimentalResolution(load_weights=False, num_classes=23)
    netWidth = model.EfficientNetExperimentalWidth(load_weights=False, num_classes=23)
    netDepth = model.EfficientNetExperimentalDepth(load_weights=False, num_classes=23)
    netBaseLine = model.EfficientNetB1(load_weights=False, num_classes=23)
    flopsBaseLine = get_flops(netBaseLine, batch_size=1)
    flopsRes = get_flops(netRes, batch_size=1)
    flopsWidth = get_flops(netWidth, batch_size=1)
    flopsDepth = get_flops(netDepth, batch_size=1)
    
    print(f'FLOPS Width: {flopsWidth}')
    print(f'FLOPS Resolution: {flopsRes}')
    print(f'FLOPS Depth: {flopsDepth}')
    print(f'Baseline FLOPS: {flopsBaseLine}')
