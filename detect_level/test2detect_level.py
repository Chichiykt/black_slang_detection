import json
import os

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class DetectLabelModel:
    """文本匹配模型，判断问题是否可回答"""

    def __init__(self, base_model_path, model_dict_path, tokenizer_path, lora_weight_path, device_index):
        self.device_index = device_index

        self.device = torch.device(f"cuda:{self.device_index}" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_type_id = self.tokenizer.eos_token_id

        # 加载基础模型
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model_path,
            num_labels=3,
            problem_type="single_label_classification",
            torch_dtype=torch.float16
        )

        self.model = PeftModel.from_pretrained(model, lora_weight_path).to(self.device)
        self.model.load_state_dict(torch.load(model_dict_path, map_location=self.device))
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.eval()
        print("模型加载完成")

    def predict(self, dialogue: str):
        messages = [
            {"role": "user", "content": f"在下面的选项中选择符合这段对话的风险等级。"
                                        f"风险等级可选项：medium_high,high,low_or_contextual。"
                                        f"\n对话内容:\n{dialogue}"}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        with torch.no_grad():
            inputs.to(self.device)
            outputs = self.model(**inputs)
        return  torch.softmax(outputs.logits, dim=1)[0].cpu().numpy().argmax()


if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_model_path = os.path.join(str(current_dir)[:str(current_dir).find("dmj") + 3],
                                   "models/base_model")
    dataset_path = os.path.join(str(current_dir)[:str(current_dir).rfind("SF") + 2],
                                "dataset/industry_slang_detection_dataset_10000.json")

    output_dir = os.path.join(current_dir, "model_dict")

    last_model_dict_path = os.path.join(output_dir, "last_model_dict/model_dict.bin")
    last_model_tokenizer_path = os.path.join(output_dir, "last_model_tokenizer")
    last_model_lora_path = os.path.join(output_dir, "last_model_lora")

    detectModel = DetectLabelModel(base_model_path, last_model_dict_path, last_model_tokenizer_path,
                                   last_model_lora_path, 2)

    datas = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        datas.extend(json.load(f))
    count = 0
    acc = 0

    level2label = {
        "medium_high": 0,
        "high": 1,
        "low_or_contextual": 2
    }
    for data in datas:
        if data["split"] == "test":
            count += 1
            print(f"对话-------------{data['dialogue']}")
            print(f"实际标签-------------{data['risk_level']}---{level2label[data['risk_level']]}")
            predict_label = detectModel.predict(data['dialogue'])
            print(f"预测标签-------------{predict_label}")
            if predict_label == level2label[data['risk_level']]:
                acc += 1
            if count % 100 == 0:
                print(f"{acc}/{count}={(acc / count) * 100}%")

