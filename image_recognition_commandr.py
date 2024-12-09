import requests
import base64
import json

# Your OpenRouter API Key
OPENROUTER_API_KEY = "Your-API"

def describe_image_with_model(image_path, model_name, instruction):
    """
    Sends an image to OpenRouter using the specified model and instruction.

    Args:
    - image_path (str): Path to the image file.
    - model_name (str): The name of the OpenRouter model to use.
    - instruction (str): Instruction to pass to the model.

    Returns:
    - str: Response from the model (LaTeX output or error message).
    """
    # Read the image file and encode it in base64
    with open(image_path, 'rb') as image_file:
        image_data = base64.b64encode(image_file.read()).decode('utf-8')
        image_content = f"data:image/jpeg;base64,{image_data}"

    # Define the request payload
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image_url", "image_url": {"url": image_content}}
                ]
            }
        ]
    }

    # Set the request headers
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            data=json.dumps(payload),
            timeout=60
        )

        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            return f"Request failed with status code {response.status_code}: {response.text}"
    except requests.exceptions.RequestException as e:
        return f"Request failed due to an exception: {e}"


def select_best_output_with_command_r(outputs):
    """
    Uses Command-R from OpenRouter to select the best LaTeX table from multiple options,
    taking into account both the accuracy of the extracted information and the LaTeX structure.

    Args:
    - outputs (list of str): LaTeX outputs from PixTral, Qwen, and LLaMA.

    Returns:
    - str: The LaTeX code of the best table as selected by Command-R.
    """
    formatted_outputs = "\n\n".join([f"Option {i+1}:\n{output}" for i, output in enumerate(outputs)])
    
    instruction = f"""
                    You are a decision engine called Command-R.

                    Your task is to analyze multiple LaTeX tables extracted from a receipt and select the most accurate and complete one. 

                    ### **Receipt Item Details**
                    - Each item is represented in the receipt as a line containing:
                    - `Item Code`: A unique numeric or alphanumeric code.
                    - `Item Name`: The name of the product.
                    - `Item Price`: The price of the product in the format `X.XX EUR`.
                    
                    ### **Accuracy Validation Criteria**
                    1. The LaTeX table must match the receipt format:
                    - Each row in the table must correspond to an item in the receipt with `Item Code`, `Item Name`, and `Item Price`.
                    - The **Total Price** must match the sum of all items in the receipt.
                    2. Items must:
                    - Have unique `Item Codes`.
                    - Contain clear, descriptive `Item Names` without truncations or inaccuracies.
                    - Include correct `Item Prices` as per the receipt.
                    3. The table structure must:
                    - Start with the headers: `Item Code`, `Item Name`, `Item Price`, `Total Price`.
                    - Include a **final row** with the Total Price clearly labeled.

                    ### **Instructions**
                    - Analyze the LaTeX tables below.
                    - Compare the tables for both logical accuracy and structural correctness.
                    - Return only the LaTeX code of the best table. If no table meets the criteria, return an empty response.
                    - Do not include any explanations or additional text.

                    ### **LaTeX Tables to Analyze**
{formatted_outputs}

    """

    payload = {
        "model": "cohere/command-r",  # This is the model name for Command-R on OpenRouter
        "messages": [
            {
                "role": "user",
                "content": instruction
            }
        ]
    }

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            data=json.dumps(payload),
            timeout=60
        )

        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            return f"Request failed with status code {response.status_code}: {response.text}"
    except requests.exceptions.RequestException as e:
        return f"Request failed due to an exception: {e}"


# Instructions for the OCR models
instruction = """Extract the following information from this receipt:
1. Item Code: A unique code that corresponds to each individual item. Each item has only one unique code.
2. Item Name: The name of the item as it appears on the receipt.
3. Item Price: The price of each item as listed on the receipt.
4. Total Price: The final price at the bottom of the receipt.

Organize the extracted data into a LaTeX table format with the following headers: `Item Code`, `Item Name`, `Item Price`, and `Total Price`. Ensure that:
- Each row corresponds to a specific item on the receipt.
- Each item has only one unique code.
- The final row includes the total price.

Return only the LaTeX code for the table. Do not include any preamble, explanations, or additional text.

"""

# Path to the receipt image
image_path = 'data/receipt4.jpeg'

# List of models to be tested
models = [
    {"name": "mistralai/pixtral-12b", "description": "PixTral 12B"},
    {"name": "qwen/qwen-2-vl-7b-instruct", "description": "Qwen-2V"},
    {"name": "meta-llama/llama-3.2-11b-vision-instruct:free", "description": "LLaMA-3.2"}
]

# Collect LaTeX output from each model
outputs = []
for model in models:
    print(f"\n=== Processing with {model['description']} ===")
    output = describe_image_with_model(image_path, model['name'], instruction)
    outputs.append(output)
    print(f"\nLaTeX output from {model['description']}:\n{output}\n")

# Select the best LaTeX output using Command-R
print("\n=== Selecting the best LaTeX output using Command-R ===\n")
best_output = select_best_output_with_command_r(outputs)

print("\n=== Best LaTeX Table ===\n")
print(best_output)
