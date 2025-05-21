from datasets import load_dataset

# Load the dataset
dataset = load_dataset("MBZUAI/EXAMS-V")

# Let's look at the dataset structure
print(dataset)

# For this example, we'll use the validation split
ds = dataset['test']

from tqdm import tqdm
len_ds = len(ds)
# First get all indices where language is English
english_indices = [
    i for i in tqdm(range(len_ds))
    if ds[i]['language'] == 'English'
]

# Then create the filtered dataset using select
english_ds = ds.select(english_indices)

print(f"Number of English samples: {len(english_ds)}")

print(english_ds)


#  Import required libraries
import google.generativeai as genai
from PIL import Image
# from google.colab import files
import json
import io

# Set your Gemini API key
API_KEY = ""

# ⬅️ STEP 4: Define the Gemini prompt
PROMPT = (


     "You are given an exam-style image that contains a multiple-choice question in Arabic or English. "
    "The image may include any of the following: diagrams, graphs, labeled arrows, tables, circuit labels, or scientific notations.\n\n"

    "Your task is to extract and structure all visible information from the image *without solving the question*.\n\n"

    "Please extract the following:\n\n"
    "1. *question_text*: The full question as shown (in Arabic or English)\n"
    "2. *options*: List of choices (A, B, C, D) with exact phrasing\n"
    "3. *diagram_caption*: A detailed but concise description of any diagrams or illustrations. "
    "Mention visual elements like arrows, flows, labels (e.g., 'X', 'Y'), node names, or organism types\n"
    "For graphs, include the title (if present), axis labels, units, any trend lines, or highlighted points (e.g., A, B)\n"
    "4. *labels*: List any textual or visual labels seen in the diagram (e.g., circuit labels, anatomical terms, arrows, part names)\n"
    "5. *table*: If a table is present, extract all rows and columns into a JSON table structure. If no table, return null\n"
    "6. *relationships* (optional): If the diagram contains arrows showing flow, direction, or feeding paths (like food webs or circuits), extract them in this format:\n"
    "   [ {\"from\": \"rabbits\", \"to\": \"X\"}, {\"from\": \"X\", \"to\": \"hawks\"} ]\n"
    "7. *language*: Identify whether the question is in Arabic or English\n\n"

    "Output your response as a single JSON object with the keys:\n"
    "question_text, options, diagram_caption, labels, table, relationships, language\n\n"

    "Do not solve the question or infer any answers.\n"
    "Just describe and extract what is visually present — symbols, labels, directions, captions, and question content."
)

from PIL import Image

def extract_exam_info(image):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content([PROMPT, image])
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

def load_func(image):
    # print("Loading local image from:", image_path)
    # image = Image.open(image_path)

    # print("\n Extracting visible content with Gemini...\n")
    caption = extract_exam_info(image)

    # print("\n Extracted Question Text:\n")
    # print(caption)

    return caption

from langchain_groq import ChatGroq

import os
os.environ["GEMINI_API_KEY"] = API_KEY

import google.generativeai as genai

# Replace "YOUR_API_KEY" with your actual Gemini API key
genai.configure(api_key=API_KEY)

GROK_API_KEY = ""


def reason_with_deepseek(extracted_text):
    llm = ChatGroq(
        model="deepseek-r1-distill-llama-70b",
        temperature=0,
        groq_api_key=GROK_API_KEY
    )

    reasoning_prompt = (
        "You are answering a multiple-choice question.\n"
        "Return **only** the correct uppercase letter (A, B, C, D, etc).\n"
        "Do not explain.\n"
        "Do not write any reasoning.\n"
        "Do not add punctuation or extra text.\n"
        "Respond with only one letter.\n\n"

       "Example 1:\n"
        "Question: Which number is even?\n"
        "Options: A. 3  B. 5  C. 8  D. 7\n"
        "Correct Answer: C\n\n"

        "Example 2:\n"
        "Question: What is the capital of Japan?\n"
        "Options: A. Seoul  B. Beijing  C. Tokyo  D. Bangkok\n"
        "Correct Answer: C\n\n"

        "Now answer this:\n"
        f"{extracted_text}\n"
        "Correct Answer:"
    )
    # reasoning_prompt = (
    # "The carboxyl functional group (-COOH) is present in\n\n"
    # "A: picric acid\n"
    # "B: barbituric acid\n"
    # "C: ascorbic acid\n"
    # "D: aspirin\n"
    # "Choose the correct answer. Respond with only the single uppercase letter of the correct option (A, B, C, D, etc.) on its own line. Do not write anything else."
    # )

    response = llm.invoke(reasoning_prompt)
    print("\n✅ Predicted Answer:")
    # print(response.content)
    # s = "abc123xyz!@#"
    # last_alpha = None

    # for char in reversed(response.content):  # Check from the end
    #   if char.isalpha():    # True if alphabetic (A-Z, a-z)
    #       last_alpha = char
    #       break

# print(last_alpha)  # Output: 'z'
    print(response.content[-1])
    return response.content[-1]

# english_ds=english_ds[:]
# dict={}
# for i in range(20):
#   # i=int()
#   print(i)
#   image_url = english_ds['image'][i]
#   extracted_text = load_func(image_url)
#   answer=reason_with_deepseek(extracted_text)
#   dict[i]={}
#   dict[i]['id']=english_ds['sample_id'][i]

#   dict[i]['answer_key']=answer
#   dict[i]['language']='English'
#   # if i==5:
#   #   break
#   # break

# print(dict)
# # Run both steps
# image_url = english_ds['image'][0]
# extracted_text = load_func(image_url)
# reason_with_deepseek(extracted_text)

# import json
# # Save to a JSON file
# with open("pred.json", "w") as f:
#     json.dump(dict, f, indent=4)
# # json.dumps(dict)

# gold={}

# print(dict)

# import json
# # Save to a JSON file
# with open("gold.json", "w") as f:
#     json.dump(gold, f, indent=4)
# # json.dumps(dict)

# for i in range(20):
#   exists = any(inner_dict["id"] == english_ds['sample_id'][i] for inner_dict in dict.values())
#   if exists:
#     print("exists")
#     gold[i]={}
#     gold[i]['id']=english_ds['sample_id'][i]
#     gold[i]['answer_key']=english_ds['answer_key'][i]
#     gold[i]['language']='English'
#   else:
#     continue


# import json
# # Save to a JSON file
# with open("gold.json", "w") as f:
#     json.dump(gold, f, indent=4)
# # json.dumps(dict)

import json
import os

# Load existing pred.json if it exists
pred_file = "pred.json"
result_dict = {}

batch_size = 20
total_samples = len(english_ds)

for start in range(0, total_samples, batch_size):
    end = min(start + batch_size, total_samples)
    print(f"Processing samples {start} to {end - 1}")

    for i in range(start, end):
        image_url = english_ds['image'][i]
        extracted_text = load_func(image_url)
        answer = reason_with_deepseek(extracted_text)

        result_dict[i] = {
            'id': english_ds['sample_id'][i],
            'answer_key': answer,
            'language': 'English'
        }

    # Append/save results after each batch
    with open(pred_file, "w") as f:
        json.dump(result_dict, f, indent=4)
