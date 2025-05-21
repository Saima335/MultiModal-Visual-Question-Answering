# Multimodal Reasoning Visual Question Answering (VQA)

This project focuses on multimodal reasoning for Visual Question Answering (VQA), leveraging image captioning and large language models for reasoning.

## 🧠 Overview

The system performs the following:
1. Takes an image and a question.
2. Generates a caption using an image captioning model (Gemini API).
3. Performs reasoning using the caption and question via DeepSeek-R1-distill-llama-70b (Grok API).
4. Outputs the answer in a `pred.json` file.

## 📁 Project Structure

```

project/
│
├── testing/
│ ├── mistral.py # llava-v1.6-mistral-7b-hf experiments (not used)
│ ├── deepseek.py # other deepseek experiments
│ ├── ocr.py # OCR-based models
│ └── image_captioning.py # Other image captioning models (blip)
|
├── test_images/ # contain test images
│
├── evaluate.py # Generates pred.json with final answers
├── evaluate_results.py # Compares pred.json and gold.json, computes accuracy
├── pred.json # Predictions generated
├── gold.json # Ground truth labels
├── requirements.txt
└── README.md

````

## 🧪 Evaluation

Run the following commands to generate predictions and evaluate:

```bash
python evaluate.py
python evaluate_results.py
````

Output:

* `pred.json`: Contains list of dicts with `id`, `language`, `answer_key`
* Accuracy is printed after comparing with `gold.json`.

## 🔧 APIs Used

* **Google Gemini**: For image captioning
* **DeepSeek LLaMA-70B (Distilled)**: For reasoning

  > API keys are managed through Google Console (Gemini) and Grok (DeepSeek)

## ⚙️ Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

## 📌 Notes

* Initial experiments with `llava-v1.6-mistral`, `deepseek`, OCR, and local captioning gave poor results.
* Final pipeline uses Gemini + DeepSeek APIs.

````
