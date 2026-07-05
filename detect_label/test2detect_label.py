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
            num_labels=2,
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
            {"role": "user",
             "content": f"以下对话包含黑话吗？\n{dialogue}"}
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
        return torch.softmax(outputs.logits, dim=1)[0].cpu().numpy()[1] > 0.5, torch.softmax(outputs.logits, dim=1)[0].cpu().numpy()[1]


if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_model_path = os.path.join(str(current_dir)[:str(current_dir).find("dmj") + 3],
                              "models/deepSeek_R1_Distill_Qwen_1.5b")
    dataset_path = os.path.join(str(current_dir)[:str(current_dir).rfind("SF") + 2],
                                "dataset/industry_slang_detection_dataset_10000.json")

    output_dir = os.path.join(current_dir, "model_dict")

    last_model_dict_path = os.path.join(output_dir, "last_model_dict/model_dict.bin")
    last_model_tokenizer_path = os.path.join(output_dir, "last_model_tokenizer")
    last_model_lora_path = os.path.join(output_dir, "last_model_lora")

    detectModel = DetectLabelModel(base_model_path, last_model_dict_path, last_model_tokenizer_path, last_model_lora_path, 2)

    datas = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        datas.extend(json.load(f))
    count = 0
    acc = 0
    for data in datas:
        if data["split"] == "test":
            count += 1
            print(f"对话-------------{data['dialogue']}")
            print(f"实际标签-------------{data['detect_label']}")
            predict_label = detectModel.predict(data['dialogue'])
            print(f"预测标签-------------{predict_label}")
            if predict_label == data['detect_label']:
                acc += 1
            if count % 100 == 0:
                print(f"{acc}/{count}={(acc/count)*100}%")

    """
对话-------------A：你先别急。 A：今晚上分的节奏不错，别乱停。 B：还要继续吗？ A：先别慌，按计划来。 A：按照菠菜的方式走，别乱改。 A：你先配合，后面自然有人接。
实际标签-------------1
预测标签-------------(True, 1.0)
对话-------------A：我大概明白你的情况了。 用户A：退票的钱什么时候能到账？ B：一般一到三个工作日。
实际标签-------------0
预测标签-------------(False, 9.47e-05)
对话-------------A：先说重点。 A：案例通报里把杀猪盘列成了重点识别词。 用户B：做案例复盘时，这些词特别值得单独整理。 B：正常情况下按正规流程处理就行。
实际标签-------------0
预测标签-------------(False, 2.1e-06)
对话-------------A：你先别急。 A：警方通报里又出现了跑分这类词。 用户B：做案例复盘时，这些词特别值得单独整理。 B：所以还是要看具体上下文。
实际标签-------------0
预测标签-------------(False, 1e-06)
对话-------------A：先说重点。 A：不要把款留在一个地方太久，先拆一下。 B：后面谁来处理？ A：你先按我说的做，剩下的有人对接。 A：先把这一步做完，再看后面。
实际标签-------------1
预测标签-------------(True, 0.9995)
对话-------------A：先说重点。 A：系统提示账户异常，还要走一步确认流程。 用户B：不是说会原路退回吗？ A：你先做这一步，不然系统不会放款。 A：别在外面多说，照着做就行。
实际标签-------------1
预测标签-------------(True, 1.0)
对话-------------甲：今天还有名额，任务完成后佣金秒结。 用户B：是不是需要自己先出钱？ 甲：现在不能停，停了就算返现，前面的钱不好退。 用户B：看起来像越做越多。 甲：最后都会返，你先不要中途退出。
实际标签-------------1
预测标签-------------(True, 1.0)
对话-------------A：目前还来得及处理。 A：培训材料把养猪列成了高风险词。 用户B：这类词不一定都违法，但确实该高度警惕。 B：所以还是要看具体上下文。
实际标签-------------0
预测标签-------------(False, 1.67e-06)
对话-------------A：目前还来得及处理。 甲：前面先把聊天氛围铺好，投资话题后面再带。 乙：你的意思是先收网？ 甲：别太明显，先让对方觉得你是真关心。 乙：那做信任是在后面做吗？ 甲：对，等信任足够再推进。 A：你先配合，后面自然有人接。
实际标签-------------1
预测标签-------------(True, 1.0)
对话-------------A：目前还来得及处理。 用户A：信用卡临时额度什么时候生效？ 用户B：通过审批后一般会马上更新。 B：正常情况下按正规流程处理就行。
实际标签-------------0
预测标签-------------(False, 1.15e-05)
对话-------------甲：我先垫付会议室费用。 B：后面走对公报销就行。 B：这种情况还是以官方说明为准。
实际标签-------------0
预测标签-------------(False, 1.264e-05)
对话-------------A：目前还来得及处理。 用户A：别自己乱冲，跟着节奏走。 乙：现在越追越慌。 用户A：先别慌，按计划来。 用户A：按照上分的方式走，别乱改。 A：别在外面多说，照着做就行。
实际标签-------------1
预测标签-------------(True, 1.0)
对话-------------A：你先别急。 用户A：如果替别人做车手，法律责任是不是很大？ 用户B：通常要高度怀疑，尤其不能随便提交证件和银行卡。 B：正常情况下按正规流程处理就行。
实际标签-------------0
预测标签-------------(False, 1.514e-05)
对话-------------A：目前还来得及处理。 甲：合同里如果出现跑分这种费用，是否合规？ 乙：这类情况建议先咨询专业人士，不要轻易参与。
实际标签-------------0
预测标签-------------(False, 4.977e-05)
对话-------------A：先别急着谈收益，关系要慢慢聊熟。 B：你的意思是先感情铺垫？ A：成熟了再推进，别让人一下子起疑。 B：那养猪是在后面做吗？ A：对，等信任足够再推进。 A：这事知道的人越少越好。
实际标签-------------1
预测标签-------------(True, 1.0)
对话-------------A：这个事情可以继续谈。 用户A：今天还有名额，任务完成后佣金秒结。 用户B：要我先做什么？ 用户A：你这边如果垫付了，就要继续补一单才能走完。 用户B：能不能只做一单？ 用户A：通常不行，这边要按任务链完成。  A：别在外面多说，照着做就行。
实际标签-------------1
预测标签-------------(True, 1.0)
对话-------------A：先说重点。 用户A：今晚狗庄的节奏不错，别乱停。 乙：我刚刚已经输了两把。 用户A：别停，情绪一乱更容易出错。
实际标签-------------1
预测标签-------------(True, 1.0)
对话-------------A：目前还来得及处理。 用户A：如果替别人做料卡，法律责任是不是很大？ 乙：如果涉及转账、代办、借名之类的操作，就更要谨慎。 B：所以还是要看具体上下文。
实际标签-------------0
预测标签-------------(False, 6.6e-06)
对话-------------甲：前面先把聊天氛围铺好，投资话题后面再带。 乙：是不是得先把信任做足？ 甲：成熟了再推进，别让人一下子起疑。 甲：后面会慢慢过渡到做信任，别一步到位。 A：别在外面多说，照着做就行。
实际标签-------------1
预测标签-------------(True, 1.0)
对话-------------A：你先别急。 甲：社区讲座提醒，遇到菠菜要先核验。 B：这类词不一定都违法，但确实该高度警惕。 甲：像背债这种词也一样，出现时要格外小心。 B：不能只看一个词就下结论。
实际标签-------------0
预测标签-------------(False, 2e-06)
对话-------------A：先说重点。 甲：你的退款现在显示补单，需要先处理一下。 用户B：不是说会原路退回吗？ 甲：先走一下专员通道，完成后会一并返还。 A：先把这一步做完，再看后面。
实际标签-------------1
预测标签-------------(True, 0.999)
对话-------------A：你先别急。 A：别自己乱冲，跟着节奏走。 B：要充多少才合适？ A：跟好这波，不要自己改节奏。 A：等会先上分，后面再慢慢补。 A：别在外面多说，照着做就行。
实际标签-------------1

    
    """