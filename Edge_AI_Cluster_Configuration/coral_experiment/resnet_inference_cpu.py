# code for inferencing with mobilenet using cpu
# model : mobilenet_v2_1.0_224_quant_edgetpu.tflite
# data : generate random images

import time
import numpy as np
from tflite_runtime.interpreter import Interpreter

# Specify the TensorFlow model, labels, and image
MODEL_PATH = "./model/tfhub_tf2_resnet_50_imagenet_ptq.tflite"

# 100 random images (224 x 224 RGB)
random_images = np.random.randint(0, 256, (100, 224, 224, 3), dtype=np.uint8)

# Initialize the TF interpreter
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# 입력 / 출력 정보 가져오기
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# inference
inference_times = []
for i, img in enumerate(random_images):

    # image dim, dtype
    input_data = np.expand_dims(img, axis=0).astype(input_details[0]['dtype'])

    start_time = time.time()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    end_time = time.time()

    pred_label = np.argmax(output_data)
    confidence = np.max(output_data)

    inference_time = end_time - start_time
    inference_times.append(inference_time)

    print(f"[{i+1}/100] Inference Time: {inference_time:.4f}s")

# result
total_time = sum(inference_times)
avg_time = total_time / len(inference_times)

print("\n===== Inference Summary (CPU) =====")
print(f"Total inference time for 100 images: {total_time:.4f} seconds")
print(f"Average inference time per image: {avg_time:.4f} seconds")
