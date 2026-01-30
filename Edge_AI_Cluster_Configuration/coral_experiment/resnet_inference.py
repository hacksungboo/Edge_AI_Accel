# code for inferencing with resnet using coral edge tpu
# model : tfhub_tf2_resnet_50_imagenet_ptq_edgetpu.tflite
# data : generate random images

import os
import time
import numpy as np
from PIL import Image
from pycoral.adapters import classify
from pycoral.adapters import common
from pycoral.utils.edgetpu import make_interpreter

# Specify the TensorFlow model, labels, and image
MODEL_PATH = "./model/tfhub_tf2_resnet_50_imagenet_ptq_edgetpu.tflite"

# 100 random images (224 x 224 RGB)
random_images = np.random.randint(0, 256, (100, 224, 224, 3), dtype=np.uint8)


# Initialize the TF interpreter
interpreter = make_interpreter(MODEL_PATH)
interpreter.allocate_tensors()


# inference
inference_times = []
for i, img in enumerate(random_images):
    
    # Run an inference
    start_time = time.time()
    common.set_input(interpreter, img)
    interpreter.invoke()
    end_time = time.time()
    classes = classify.get_classes(interpreter, top_k=1)

    pred_label = classes[0].id
    confidence = classes[0].score

    # inference time
    inference_time = end_time - start_time
    inference_times.append(inference_time)

    # result
    print(f"[{i+1}/100] Inference Time: {inference_time:.4f}s")


total_time = sum(inference_times)
avg_time = total_time / len(inference_times)

print("\n===== Inference Summary =====")
print(f"Total inference time for 100 images: {total_time:.4f} seconds")
print(f"Average inference time per image: {avg_time:.4f} seconds")

