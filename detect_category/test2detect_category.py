import json
import os

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class DetectCategoryModel:
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
            num_labels=12,
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
            {"role": "user", "content": f"以下对话属于哪类场景？"
                                        f"场景包括: loan_fraud,money_mule_laundering,refund_or_impersonation_fraud,"
                                        f"gambling_slang,romance_investment_fraud,brushing_fraud,"
                                        f"legal_or_compliance_consultation,anti_fraud_education,"
                                        f"normal_finance_or_service,daily_life_hard_negative,"
                                        f"research_or_annotation_discussion,news_or_case_discussion。"
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

    detectModel = DetectCategoryModel(base_model_path, last_model_dict_path, last_model_tokenizer_path,
                                   last_model_lora_path, 2)

    datas = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        datas.extend(json.load(f))
    count = 0
    acc = 0

    category2label = {
        "loan_fraud": 1,
        "money_mule_laundering": 2,
        "refund_or_impersonation_fraud": 3,
        "gambling_slang": 4,
        "romance_investment_fraud": 5,
        "brushing_fraud": 6,
        "legal_or_compliance_consultation": 7,
        "anti_fraud_education": 8,
        "normal_finance_or_service": 9,
        "daily_life_hard_negative": 10,
        "research_or_annotation_discussion": 11,
        "news_or_case_discussion": 0
    }
    for data in datas:
        if data["split"] == "test":
            count += 1
            print(f"对话-------------{data['dialogue']}")
            print(f"实际标签-------------{data['category']}---{category2label[data['category']]}")
            predict_label = detectModel.predict(data['dialogue'])
            print(f"预测标签-------------{predict_label}")
            if predict_label == category2label[data['category']]:
                acc += 1
            if count % 100 == 0:
                print(f"{acc}/{count}={(acc / count) * 100}%")

"""
对话-------------A：你先别急。 甲：这笔钱别直接碰，先过一下走账。 B：我只负责收还是还要再转？ 甲：你先按我说的做，剩下的有人对接。 A：你先配合，后面自然有人接。
实际标签-------------money_mule_laundering---2
预测标签-------------2
对话-------------A：这个事情可以继续谈。 用户A：兼职很简单，先做几单小额任务就行。 乙：是不是需要自己先出钱？ 用户A：先垫付一笔，系统确认后本金和佣金一起返。 乙：那如果我现在不想做了呢？ 用户A：别急，先把这一步过完，不然就容易卡单。 A：别在外面多说，照着做就行。
实际标签-------------brushing_fraud---6
预测标签-------------6
对话-------------A：先说重点。 用户A：新手先从低金额做，后面单子会更稳定。 用户B：中间会不会有风险？ 用户A：先刷手一笔，系统确认后本金和佣金一起返。 用户B：那如果我现在不想做了呢？ 用户A：别急，先把这一步过完，不然就容易连单。 A：你先配合，后面自然有人接。
实际标签-------------brushing_fraud---6
预测标签-------------6
对话-------------A：目前还来得及处理。 A：今晚菠菜的节奏不错，别乱停。 用户B：要充多少才合适？ A：别停，情绪一乱更容易出错。 A：先把这一步做完，再看后面。
实际标签-------------gambling_slang---4
预测标签-------------4
对话-------------A：我大概明白你的情况了。 A：目前款项还没释放，需要先做校验。 B：为什么还要我再转一笔？ A：先走一下专员通道，完成后会一并返还。 A：先把这一步做完，再看后面。
实际标签-------------refund_or_impersonation_fraud---3
预测标签-------------3
对话-------------A：你先别急。 甲：这把没中就继续，先把前面的补回来。 B：还要继续吗？ 甲：先别慌，按计划来。 甲：等会先上分，后面再慢慢补。 A：先把这一步做完，再看后面。
实际标签-------------gambling_slang---4
预测标签-------------4
对话-------------A：我大概明白你的情况了。 A：这笔钱别直接碰，先过一下水房。 乙：后面谁来处理？ A：流程分开做，别把事情都放在一处。 A：等这边走完，再安排车手。 A：今天处理效率最高，拖久了不好说。
实际标签-------------money_mule_laundering---2
预测标签-------------2
对话-------------A：目前还来得及处理。 甲：先别急着谈收益，关系要慢慢聊熟。 B：这样聊多久比较合适？ 甲：对，先把关系做稳，再慢慢往收益上引。 A：今天处理效率最高，拖久了不好说。
实际标签-------------romance_investment_fraud---5
预测标签-------------5
对话-------------A：你先别急。 A：不要把款留在一个地方太久，先拆一下。 用户B：后面谁来处理？ A：你先按我说的做，剩下的有人对接。 A：最后一段交给过卡，你不用露面太多。 A：这事知道的人越少越好。
实际标签-------------money_mule_laundering---2
预测标签-------------2
对话-------------A：目前还来得及处理。 A：你这个情况也能做，我们先帮你做资料，后面额度更容易下来。 B：为什么听起来不太对劲？ A：除了做资料，空放也得一起处理。 A：先把流程做完，到账后你就明白了。 A：今天处理效率最高，拖久了不好说。
实际标签-------------loan_fraud---1
预测标签-------------1
对话-------------甲：最近做课件时，我把垫付写进了典型案例。 用户B：这类词不一定都违法，但确实该高度警惕。 B：不能只看一个词就下结论。
实际标签-------------anti_fraud_education---8
预测标签-------------8
对话-------------用户A：兼职很简单，先做几单小额任务就行。 乙：具体怎么操作？ 用户A：现在不能停，停了就算卡单，前面的钱不好退。 乙：那如果我现在不想做了呢？ 用户A：别急，先把这一步过完，不然就容易垫付。 A：别在外面多说，照着做就行。
实际标签-------------brushing_fraud---6
预测标签-------------6
对话-------------甲：这单走得快，不过前面会有一点包装资料，行业里都这么操作。 乙：这个不是有风险吗？ 甲：很多人都是这么走的，不影响拿款。 A：先把这一步做完，再看后面。
实际标签-------------loan_fraud---1
预测标签-------------1
对话-------------A：征信花了也别急，我们这边能处理刷流水这个环节。 乙：为什么听起来不太对劲？ A：除了刷流水，背债也得一起处理。 A：先把流程做完，到账后你就明白了。 A：你先配合，后面自然有人接。
实际标签-------------loan_fraud---1
预测标签-------------1

"""