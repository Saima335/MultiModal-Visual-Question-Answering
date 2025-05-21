from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch
import requests
import re

# Load processor and model
processor = AutoProcessor.from_pretrained("deepseek-ai/deepseek-vl2")
model = AutoModelForVision2Seq.from_pretrained(
    "deepseek-ai/deepseek-vl2",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
).to("cuda:0")

# Load image from path and convert to base64
image_path = "test5.png"  # Replace with your local image path
with open(image_path, "rb") as image_file:
    image_bytes = image_file.read()
    base64_str = base64.b64encode(image_bytes).decode('utf-8')

# Decode base64 back to PIL Image
image = Image.open(BytesIO(base64.b64decode(base64_str))).convert("RGB")

# Define instruction (can be localized)
instruction = """Analyze the image of a multiple-choice question. Identify the question, all answer options (even if there are more than four), and any relevant visuals like graphs or tables. Choose the correct answer based only on the image. Reply with just the letter of the correct option, explanation. give too"""

# Format as LLaVA chat template
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": instruction},
            {"type": "image"},
        ],
    },
]

# Prepare prompt and inputs
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda:0")

# Generate prediction
output = model.generate(**inputs, max_new_tokens=200)
response = processor.decode(output[0], skip_special_tokens=True)

# response = processor.batch_decode(output, skip_special_tokens=True)[0]
# response = response.split("[/INST]")[-1].strip()  # optional, to strip prompt
print("Predicted Answer:", response)