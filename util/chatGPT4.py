import json
import time
from pathlib import Path
import io
import base64
import spacy
import os
from openai import OpenAIError, OpenAI

# Load English language model from Spacy
nlp = spacy.load("en_core_web_sm")

# Set up the OpenAI API key
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


class TextpromptGen:
    
    def __init__(self, root_path, control=False):
        super(TextpromptGen, self).__init__()
        self.model = "gpt-4" 
        self.save_prompt = True
        self.scene_num = 0
        self.base_content = (
            "Please generate scene description based on the given information:" if control
            else "Please generate next scene based on the given scene/scenes information:"
        )
        self.content = self.base_content
        self.root_path = root_path

    def write_json(self, output, save_dir=None):
        if save_dir is None:
            save_dir = Path(self.root_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        try:
            output['background'][0] = self.generate_keywords(output['background'][0])
            with open(save_dir / f'scene_{str(self.scene_num).zfill(2)}.json', "w") as json_file:
                print("saving scene")
                json.dump(output, json_file, indent=4)
        except Exception as e:
            pass

    def write_all_content(self, save_dir=None):
        if save_dir is None:
            save_dir = Path(self.root_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / 'all_content.txt', "w") as f:
            f.write(self.content)

    def regenerate_background(self, style, entities, scene_name, background=None):
        content = (
            f"Please generate a brief scene background with Scene name: {scene_name}; "
            f"Entities: {str(entities)}; Style: {style}"
        )
        if background:
            content = f"Please generate a brief scene background with Scene name: {scene_name}; Background: {str(background).strip('.')}; Entities: {str(entities)}; Style: {style}"

        messages = [
            {"role": "system", "content": "You are an intelligent scene generator. Generate a background prompt without mentioning entities. Please use the format below: (the output should be json format)\n \
                        {'scene_name': ['scene_name'], 'entities': ['entity_1', 'entity_2', 'entity_3'], 'background': ['background prompt']}"},
            {"role": "user", "content": content}
        ]

        response = client.chat.completions.create(
            model=self.model,
            messages=messages
        )

        return response.choices[0].message.content

    def run_conversation(self, style=None, entities=None, scene_name=None, background=None, control_text=None):
        if control_text:
            self.scene_num += 1
            scene_content = f"\nScene information: {control_text.strip('.')}; Style: {style}"
            self.content = self.base_content + scene_content
        elif style and entities:
            assert background or scene_name, 'At least one of background or scene_name must be provided'
            self.scene_num += 1
            scene_content = (
                f"\nScene {self.scene_num}: {{Background: {str(background).strip('.')} . Entities: {str(entities)}; Style: {style}}}"
                if background else
                f"\nScene {self.scene_num}: {{Scene name: {str(scene_name).strip('.')} . Entities: {str(entities)}; Style: {style}}}"
            )
            self.content += scene_content
        else:
            assert self.scene_num > 0, 'No scene content available to regenerate.'

        messages = [
            {"role": "system", "content": "You are an intelligent scene generator. Generate a background without entities. Please use the format below: (the output should be json format)\n \
                        {'scene_name': ['scene_name'], 'entities': ['entity_1', 'entity_2', 'entity_3'], 'background': ['background prompt']}"},
            {"role": "user", "content": self.content}
        ]
        output = None
        for _ in range(10):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages
                )
                output = response.choices[0].message.content
                try:
                    print(output)
                    output = eval(response)
                    _, _, _ = output['scene_name'], output['entities'], output['background']
                    if isinstance(output, tuple):
                        output = output[0]
                    if isinstance(output['scene_name'], str):
                        output['scene_name'] = [output['scene_name']]
                    if isinstance(output['entities'], str):
                        output['entities'] = [output['entities']]
                    if isinstance(output['background'], str):
                        output['background'] = [output['background']]
                    break
                except Exception as e:
                    print("An error occurred when transfering the output of gpt-4 into a dict, let's try again!", str(e))
                    continue
            except OpenAIError as e:
                print(f"API error: {e}")
                time.sleep(1)
                continue

        if self.save_prompt:
            self.write_json(output)

        return output

    def generate_keywords(self, text):
        doc = nlp(text)
        keywords = [token.text for token in doc if token.pos_ in ("NOUN", "ADJ")]
        return ", ".join(keywords)

    def generate_prompt(self, style, entities, background=None, scene_name=None):
        assert background or scene_name, 'At least one of background or scene_name must be provided'
        if background:
            background = self.generate_keywords(background)
            prompt_text = f"Style: {style}. Entities: {', '.join(entities)}. Background: {background}"
        else:
            prompt_text = f"Style: {style}. {scene_name} with {', '.join(entities[:-1])}, and {entities[-1]}"
        return prompt_text

    def encode_image_pil(self, image):
        with io.BytesIO() as buffer:
            image.save(buffer, format='PNG')
            return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def evaluate_image(self, image, eval_blur=True):
        print("evaluate_image")
        base64_image = self.encode_image_pil(image)

        # Use OpenAI's latest image model or API for image analysis if available
        # Assuming OpenAI now handles image evaluations directly via the API
        response = client.chat.completions.create(
            image=base64_image,
            model="gpt-4-vision",
        )
        return response
