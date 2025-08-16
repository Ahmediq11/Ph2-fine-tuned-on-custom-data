link model: https://huggingface.co/Thegame1161/Phi2-fine-tuned-on-custom-data
# Fine-Tuning Microsoft's Phi-2 with QLoRA for Custom Task Generation 🚀

This repository contains a complete, end-to-end project demonstrating how to fine-tune the powerful `microsoft/phi-2` language model on a custom dataset using modern, memory-efficient techniques. The project focuses on teaching the model to perform two specific tasks: generating product names and product descriptions based on a given category.

The core of this project is the implementation of **QLoRA (Quantized Low-Rank Adaptation)**, which allows for the fine-tuning of large language models (LLMs) like Phi-2 (2.8 billion parameters) on a single, consumer-grade GPU.

## ✨ Key Features

  * **State-of-the-Art Model**: Fine-tunes `microsoft/phi-2`, a high-performing small language model.
  * **Memory-Efficient Fine-Tuning**: Utilizes **QLoRA** to train the model with maximum memory savings. This is achieved by:
      * **8-bit Quantization**: Loading the base model in 8-bit precision using `bitsandbytes`.
      * **Parameter-Efficient Fine-Tuning (PEFT)**: Freezing the base model's weights and only training a small set of adapter layers (\~1% of total parameters).
  * **Instruction-Based Tuning**: Structures the custom data into a clear instruction-prompt format to effectively teach the model new tasks.
  * **Multi-Task Learning**: Trains the model on two related tasks (product name and description generation) simultaneously within a single unified dataset.
  * **End-to-End Workflow**: Covers the entire process from data loading and preprocessing to model configuration, training, and inference, showing a clear "before and after" comparison.
  * **Hugging Face Ecosystem**: Built entirely on the Hugging Face stack, including `transformers`, `datasets`, `peft`, and `trl`.

-----

## ⚙️ Project Workflow

The project follows a structured pipeline for efficient LLM fine-tuning:

1.  **Environment Setup**: Installs all necessary libraries, including `bitsandbytes` for quantization and `peft` for LoRA.
2.  **Data Loading and Preprocessing**:
      * Loads an Amazon product dataset using `pandas`.
      * Cleans and restructures the data to create a multi-task dataset for generating both product names and descriptions.
3.  **Instruction Prompting**: A formatting function is created to convert each data entry into an instruction-based prompt, clearly telling the model what task to perform.
4.  **Model Loading**: The `microsoft/phi-2` model is loaded in 8-bit precision (`load_in_8bit=True`) to drastically reduce its memory footprint.
5.  **Baseline Inference**: Before training, a baseline test is performed to see how the pre-trained model responds to the prompt. This highlights the improvements gained from fine-tuning.
6.  **QLoRA Configuration**:
      * A `LoraConfig` is defined to specify which parts of the model will have trainable adapter layers attached.
      * The base model is wrapped with `get_peft_model`, making only the small adapter layers trainable. This reduces the number of trainable parameters by over 99%.
7.  **Training**:
      * The `Trainer` from the `transformers` library is configured with `TrainingArguments`.
      * A memory-efficient `paged_adamw_8bit` optimizer is used.
      * The model is fine-tuned, updating only the LoRA adapter weights.
8.  **Inference After Fine-Tuning**: The trained LoRA adapter weights are loaded and merged with the base model to perform inference, showing that the model has successfully learned the new tasks.
9.  **Saving the Adapter**: The final trained adapter is saved as a portable `.zip` file.

-----

## 🚀 Getting Started

### Prerequisites

  * Python 3.8+
  * A CUDA-enabled GPU is highly recommended to run this project.

### Installation

1.  Clone the repository to your local machine:

    ```bash
    git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
    cd YOUR_REPOSITORY_NAME
    ```

2.  Install the required Python libraries:

    ```bash
    pip install -q accelerate bitsandbytes trl peft transformers datasets -U
    ```

-----

## ▶️ How to Use the Fine-Tuned Adapter

The output of the training process is a set of LoRA adapter weights (saved in the final checkpoint, e.g., `checkpoint-500`). To use them for inference, you must load the original base model and then apply the adapter to it.

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 1. Load the base model in 8-bit
base_model_id = "microsoft/phi-2"
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    trust_remote_code=True,
    load_in_8bit=True,
    torch_dtype=torch.float16
)

# 2. Load the tokenizer
eval_tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    add_bos_token=True,
    trust_remote_code=True,
    use_fast=False
)
eval_tokenizer.pad_token = eval_tokenizer.eos_token

# 3. Load the LoRA adapter
#    Replace 'path/to/your/checkpoint' with the actual path
ft_model = PeftModel.from_pretrained(base_model, 'path/to/your/checkpoint')


# 4. Create a prompt and generate text
eval_prompt = """
Given the product category, you need to generate a 'Product Description'.
### Category: BatteryChargers
### Product Description:
"""

model_input = eval_tokenizer(eval_prompt, return_tensors="pt").to("cuda")

ft_model.eval()
with torch.no_grad():
  output = ft_model.generate(**model_input, max_new_tokens=256, repetition_penalty=1.15)
  result = eval_tokenizer.decode(output[0], skip_special_tokens=True)

print(result)
```

-----

## 📚 Dataset

This project uses the **Amazon Product Details** dataset. It was sourced from the following public GitHub repository:

  * **Data Source**: [https://github.com/laxmimerit/All-CSV-ML-Data-Files-Download/raw/master/amazon\_product\_details.csv](https://github.com/laxmimerit/All-CSV-ML-Data-Files-Download/raw/master/amazon_product_details.csv)

-----

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
