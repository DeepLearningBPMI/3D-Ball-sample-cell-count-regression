import pathlib

import tensorflow as tf
from tf_explain.core.grad_cam import GradCAM
from tf_explain.core.integrated_gradients import IntegratedGradients

tf.config.set_visible_devices(devices=[], device_type='GPU')

_DATASET_OPTIONS = tf.data.Options()
_DATASET_OPTIONS.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

_VAL_DIR = '/home/max/Documents/Bronchoalveolar lavage/Data/Validation/'
_TEST_DIR = '/home/max/Documents/Bronchoalveolar lavage/Data/Testing/'
_DATA_SELECTOR = '*.png'

_MODEL_DIR = '/home/max/Documents/Bronchoalveolar lavage/Code/Experiments/1/4_efficientnetv2b0-0.001/'
_MODEL_SELECTOR = '*.hdf5'

def process_case(mosaic_path):
    mosaic = tf.io.read_file(filename=mosaic_path)
    mosaic = tf.io.decode_png(contents=mosaic, channels=3)
    mosaic = tf.cast(x=mosaic, dtype=tf.dtypes.float32)

    labels_path = tf.strings.regex_replace(input=mosaic_path, pattern='[a-zA-Z0-9-_]*.png', rewrite='labels.txt')
    labels = tf.io.read_file(filename=labels_path)
    labels = tf.strings.to_number(input=tf.strings.split(input=labels, sep=','))

    return mosaic, labels

def data_fn(data_dir):
    subset_cases = list(pathlib.Path(data_dir).rglob(_DATA_SELECTOR))
    cases = [tf.compat.path_to_str(path=case) for case in subset_cases]

    dataset = (
        tf.data.Dataset
            .from_tensor_slices(tensors=cases)
            .map(process_case, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .batch(batch_size=1, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    )

    dataset = dataset.with_options(_DATASET_OPTIONS)

    return dataset

def model_fn():
    model_dir = pathlib.Path(_MODEL_DIR)
    saved_models = list(model_dir.glob(pattern=_MODEL_SELECTOR))
    saved_models = list(map(str, saved_models))
    saved_models.sort(key=lambda path: (path.split('-')[-1]), reverse=True)

    best_model = saved_models[-1]
    model = tf.keras.models.load_model(filepath=best_model, compile=False)

    return model

def run():
    model = model_fn()
    model = model.get_layer('efficientnetv2-b0')

    mosaic = tf.keras.utils.load_img(path=_VAL_DIR + 'LBAL005-I_0/LBAL005_File155-254_MOSAIC_SP2_THG100003PF100002PF10000_Gamma07_10x10_0.png')
    mosaic = tf.keras.utils.img_to_array(img=mosaic)

    explainer = GradCAM()
    for cell_type in range(4):
        grid = explainer.explain(validation_data=([mosaic], None), model=model, class_index=cell_type)
        explainer.save(grid=grid, output_dir=_VAL_DIR, output_name='LBAL005-I_0_grad-cam_' + str(cell_type) + '.png')

    mosaic = tf.keras.utils.load_img(path=_TEST_DIR + 'LBAL003-I_0/LBAL003_File156-255_MOSAIC_SP3_THG100003PF100002PF10000_Gamma07_10x10_0.png')
    mosaic = tf.keras.utils.img_to_array(img=mosaic)

    explainer = GradCAM()
    for cell_type in range(4):
        grid = explainer.explain(validation_data=([mosaic], None), model=model, class_index=cell_type)
        explainer.save(grid=grid, output_dir=_TEST_DIR, output_name='LBAL003-I_0_grad-cam_' + str(cell_type) + '.png')
 
if __name__ == '__main__':
    run()
