import json
import os

import dotenv
import openai
import tyro
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from tqdm import tqdm

dotenv.load_dotenv()


def clean_prompt(prompt: str):
    return prompt.replace("\n", " ").replace("  ", " ").replace("[", "").replace("]", "").lower()


def parse_response(response: str):
    list_of_response = response.split("Round 2", 1)[1].split("\n")
    sequence_instructions = []
    for response in list_of_response[:5]:
        response = response.split(":", 1)[1].strip()
        prompt, editing_type_id = response.split(" (editing_type_id: ", 1)
        sequence_instructions.append(
            {
                "prompt": prompt.strip().lower(),
                "editing_type_id": editing_type_id.strip(")")[0].strip(),
            }
        )
    return sequence_instructions


@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, min=2, max=10))
def submit_prompt(original_prompt: str, editing_prompt: str, model: str = "gpt-4o"):
    """Follow the instructions from Fig. 9 in the paper."""
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Strictly follow 1. Respond in English 2. Use provide formatting 3. Keep instructions actionable",
            },
            {
                "role": "user",
                "content": f"""
                    Complete these tasks:
                    1. Analyze the original text prompt in English, original prompt is: {original_prompt}
                    2. Generate FIVE sequential edit instructions following this FIRST instruction: {editing_prompt}
                    3. Each instruction should have an associated editing type ID.
                    4. The sentence should not be an instruction but a sentence with a structure similar to the provide one

                    Editing Type IDs:
                    0. Random editing
                    1. Change object: change an object to another, e.g., dog to cat.
                    2. Add object: add an object, e.g., add flowers.
                    3. Delete object: delete an object, e.g., delete the clouds in the image.
                    4. Change something's content: change the content of sth, e.g., change a smiling man to an angry man by editing his facial expression.
                    5. Change something's pose: change the pose of sth, e.g., change a standing dog to a running dog.
                    6. Change something's color: change the color of sth, e.g., change a red heart to a pink heart.
                    7. Change something's material: change the material of sth, e.g., change a wooden table to a glass table. 40 images in total.
                    8. Change image background: change the image background, e.g., change white background to grasses. 80 images in total.
                    9. Change image style: change the image style, e.g., change a photo to watercolor.

                    Requirements:
                    - Each instruction modifies ONE distinct feature
                    - Maintain consistency with previous modifications
                    - Use short imperative phrases
                    - Provide an appropriate editing type ID for each instruction
                    - Ensure the sentence structure remains consistent with the original prompt and first editing prompt

                    Example:
                    - Original Prompt: "a dog wearing space suit"
                    - First Editing Prompt: "a dog wearing space suit with flowers in mouth"
                    - Correct Next Prompt: "a dog wearing space suit with a ball in mouth"
                    - Incorrect Next Prompt: "add a ball in mouth"

                    Format:
                    Text Analysis:[analysis]
                    - Round 2 Instruction:[instruction] (editing_type_id: [id])
                    - Round 3 Instruction:[instruction] (editing_type_id: [id])
                    - Round 4 Instruction:[instruction] (editing_type_id: [id])
                    - Round 5 Instruction:[instruction] (editing_type_id: [id])
                    - Round 6 Instruction:[instruction] (editing_type_id: [id])
                    ...
                    """,
            },
        ],
        temperature=0.5,
        max_tokens=2000,
    )

    return parse_response(response.choices[0].message.content)


def main(
    data_path: str,
    save_path: str,
    model: str = "gpt-4o",
):
    with open(data_path, "r") as f:
        original_data = json.load(f)

    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            saved_data = json.load(f)
    else:
        saved_data = {}

    for data_id, data in tqdm(original_data.items(), ncols=0, total=len(original_data)):
        if data_id in saved_data:
            continue

        original_prompt = clean_prompt(data["original_prompt"])
        editing_prompt = clean_prompt(data["editing_prompt"])
        image_path = data["image_path"]
        editing_type = os.path.dirname(image_path).split("_", 1)[0]
        try:
            sequence_instructions = submit_prompt(original_prompt, editing_prompt, model)
        except Exception as e:
            print(f"Error submitting prompt for {data_id}: {e}")
            continue

        sequence_instructions = [
            {"prompt": editing_prompt, "editing_type_id": editing_type}
        ] + sequence_instructions

        saved_data[data_id] = {
            "image_path": image_path,
            "original_prompt": original_prompt,
            "editing_prompts": sequence_instructions,
            "mask": data["mask"],
        }
        # break

    with open(save_path, "w") as f:
        json.dump(saved_data, f)


if __name__ == "__main__":
    tyro.cli(main)
