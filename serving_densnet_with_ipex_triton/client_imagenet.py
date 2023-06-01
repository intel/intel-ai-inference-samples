import numpy as np
from torchvision import transforms
from PIL import Image
import tritonclient.http as httpclient
from tritonclient.utils import triton_to_np_dtype
import os
import time
import argparse

argParser = argparse.ArgumentParser()
argParser.add_argument("--dataset", required = True, help="Path to IMAGENET dataset")

directory = argParser.parse_args()
directory = directory.dataset
print("Dataset directory = %s" % directory)

# preprocessing function
def image_preprocess(img_path="img2.jpg"):
    img = Image.open(img_path)
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(img).numpy()

filelist = []

for root, dirs, files in os.walk(directory):
	for file in files:
        #append the file name to the list
		filelist.append(os.path.join(root,file))

iteration = 0
elapsed_time = 0
for name in filelist:
    try:
        elapsed = 0
        start_epoch = time.time()
        print(name)
        transformed_img = image_preprocess(name)

        # Setting up client
        client = httpclient.InferenceServerClient(url="localhost:8000")

        # specify the names of the input and output layer(s) of our model
        inputs = httpclient.InferInput("input__0", transformed_img.shape, datatype="FP32")
        inputs.set_data_from_numpy(transformed_img, binary_data=True)

        outputs = httpclient.InferRequestedOutput("OUTPUT__0", binary_data=True, class_count=1000)

        # Querying the server
        results = client.infer(model_name="densenet", inputs=[inputs], outputs=[outputs])
        predictions = results.as_numpy('OUTPUT__0')
        print(predictions[:5])                                                        
        end_epoch = time.time()            
        
        iteration = iteration + 1
        
        elapsed = end_epoch - start_epoch
        elapsed_time = elapsed_time + elapsed
        print("Per Sample Inference Latency in sec", elapsed)
    except Exception:
        pass

print("Total Iteration", iteration)
print("Total elapsed time", elapsed_time)
print("Avg elapsed time per sample in sec", elapsed_time/iteration)
