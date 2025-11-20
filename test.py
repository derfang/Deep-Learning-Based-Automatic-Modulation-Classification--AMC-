import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs Detected: {len(gpus)}")

if len(gpus) > 0:
    print(f"GPU Name: {tf.config.experimental.get_device_details(gpus[0])['device_name']}")
else:
    print("TensorFlow is running on CPU.")