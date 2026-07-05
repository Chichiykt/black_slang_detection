import os

import torch
from fastapi import FastAPI
from langchain_ollama import OllamaLLM
from peft import PeftModel
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

from detect_category.test2detect_category import DetectCategoryModel
from detect_emotion.detect_emotion import generate_emotion_indicators_from_dialogue
from detect_risk.detect_risk import generate_risk_indicators_from_dialogue
from detect_interpretable_tasks.detect_interpretable_tasks import generate_black_terms_indicators_from_dialogue
from global_ball.granular_ball_classifier import GranularBallClassifier

app=FastAPI(
    title='黑话接口',
    description='通过对话内容调用黑话检测的模型',
    version='1.0'
)

class FrontendParams(BaseModel):
    dialogue_str: str

class DetectLabelModel:
    def __init__(self, base_model_path='/app/base_model',
                 model_dict_path='/app/detect_label/model_dict/last_model_dict/model_dict.bin',
                 tokenizer_path='/app/detect_label/model_dict/last_model_tokenizer',
                 lora_weight_path='/app/detect_label/model_dict/last_model_lora', device_index=0):
        self.device_index = device_index

        self.device = torch.device(f"cuda:{self.device_index}" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_type_id = self.tokenizer.eos_token_id

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

    def get_hidden(self, dialogue: str):
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
            padding=False,
            return_tensors='pt'
        )

        with torch.no_grad():
            inputs.to(self.device)
            outputs = self.model(**inputs, output_hidden_states=True)
        # hidden_first = outputs.hidden_states[-1][0][0].cpu().tolist() # 1536
        # hidden_last = outputs.hidden_states[-1][0][-1].cpu().tolist()
        hidden_second_last = outputs.hidden_states[-1][0][-2]
        return F.normalize(hidden_second_last, p=2, dim=1)

class BlackTermsDetectPipeline:

    def __init__(self,base_model_path, model_d_dict_path, tokenizer_d_path, lora_d_weight_path):
        self.label_detect = DetectLabelModel()

        self.category_detect = DetectCategoryModel(base_model_path,model_d_dict_path,tokenizer_d_path,lora_d_weight_path,0)

        self.llm = OllamaLLM(base_url='http://localhost:8080', model='deepseek-r1:14b', temperature=0.6)

        self.risk_list = ["high","medium_high","low_or_contextual"]
        self.judgment_list = ['中高风险，建议人工复核', '低风险，需结合上下文复核', '存在词面风险，但当前更像正常语境', '无明显黑话嫌疑', '较高风险，存在明显黑话嫌疑', '高风险，黑话与异常语境高度吻合']
        self.emotion_tag_list = ['不甘心', '专业', '伪专业', '伪关心', '低声提醒', '侥幸', '催促', '兴奋', '分析', '利诱', '安抚', '平静', '拉近关系', '提醒', '施压', '日常', '温和施压', '焦虑', '煽动', '理性', '礼貌', '科普', '紧张', '规避', '警惕', '警示', '讨论', '诱导', '谨慎', '轻松']
        self.black_terms_list = ['上分', '下分', '专员通道', '中转', '倍投', '做信任', '做资料', '养熟', '养猪', '刷单', '刷手', '刷流水', '包装资料', '卡单', '回流款', '垫付', '复核', '带单', '征信', '感情铺垫', '收网', '料卡', '断单', '杀猪', '杀猪盘', '水房', '洗征信', '狗庄', '砍头息', '空放', '背债', '菠菜', '补仓', '补单', '解冻', '走账', '跑分', '跟单', '车手', '过卡', '返现', '连单', '退款', '额度', '风控校验', '验证', '验证金']


    def analyze_dialogue(self, datas):
        """核心调用函数：传入一段对话，输出所有任务的综合分析结果"""
        
        result_list=[]
        label2category = {
            1: "loan_fraud",
            2: "money_mule_laundering",
            3: "refund_or_impersonation_fraud",
            4: "gambling_slang",
            5: "romance_investment_fraud",
            6: "brushing_fraud",
            7: "legal_or_compliance_consultation",
            8: "anti_fraud_education",
            9: "normal_finance_or_service",
            10: "daily_life_hard_negative",
            11: "research_or_annotation_discussion",
            0: "news_or_case_discussion"
        }

        for data in datas:
            dialogue_str = data['dialogue']
            features = self.label_detect.get_hidden(dialogue_str)
            clf = GranularBallClassifier("粒球聚类结果")
            predict_label = clf.predict(features.numpy())
            if predict_label == 1:
                predict_label_str = "suspicious"
            else:
                predict_label_str = "normal"

            raw_category_id = self.category_detect.predict(dialogue_str)
            predict_category = label2category.get(int(raw_category_id), "unknown_category")

            result_a =generate_risk_indicators_from_dialogue(self.llm,[dialogue_str],self.risk_list,self.judgment_list,temperature=0.6)
            print('result_a')
            print(result_a)
            answer_a = result_a[0]
            _, judgment_list, risk_level, suspicion_score = (
                answer_a["dialogue"],
                answer_a["judgment"],
                answer_a["risk_level"],
                answer_a["suspicion_score"]
            )

            result_b = generate_emotion_indicators_from_dialogue(self.llm,[dialogue_str],self.emotion_tag_list)
            print('result_b')
            print(result_b)
            answer_b = result_b[0]
            _, emotion_tag, sentiment_score, arousal_score = (
                answer_b["dialogue"],
                answer_b["emotion_tag"],
                answer_b["sentiment_score"],
                answer_b["arousal_score"]
            )

            result_c = generate_black_terms_indicators_from_dialogue(self.llm,[dialogue_str],self.black_terms_list)
            print('result_c')
            print(result_c)
            answer_c = result_c[0]
            _, black_terms, rationale = (
                answer_c["dialogue"],
                answer_c["black_terms"],
                answer_c["rationale"],
            )
            if black_terms :
                has_black_terms = 1
            else:
                has_black_terms = 0
            group ={
                "label":predict_label_str,
                "detect_label":predict_label,
                "category":predict_category,
                "risk_level":risk_level,
                "emotion_tag":emotion_tag,
                "sentiment_score":sentiment_score,
                "arousal_score":arousal_score,
                "suspicion_score":suspicion_score,
                "has_black_terms":has_black_terms,
                "black_terms":black_terms,
                "dialogue":dialogue_str,
                "judgment":judgment_list,
                "rationale":rationale

            }
            result_list.append(group)

        return result_list

@app.post('/detect')
def main(params:FrontendParams):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_model_path = os.path.join(current_dir,
                                   "base_model/base_model")

    # 多分类的模型参数地址
    base_d_path = "/app/detect_category"


    last_model_d_dict_path = os.path.join(base_d_path, "model_dict/last_model_dict/model_dict.bin")
    last_model_d_tokenizer_path = os.path.join(base_d_path, "model_dict/last_model_tokenizer")
    last_model_d_lora_path = os.path.join(base_d_path, "model_dict/last_model_lora")

    detect_way = BlackTermsDetectPipeline(base_model_path,
                                          last_model_d_dict_path, last_model_d_tokenizer_path, last_model_d_lora_path,
                                          )
    dialogue_str = params.dialogue_str
    dia_str = detect_way.analyze_dialogue([{"dialogue": dialogue_str}])
    return {
                "label":dia_str[0]["label"],
                "detect_label":dia_str[0]["detect_label"],
                "category":dia_str[0]["category"],
                "risk_level":dia_str[0]["risk_level"],
                "emotion_tag":dia_str[0]["emotion_tag"],
                "sentiment_score":dia_str[0]["sentiment_score"],
                "arousal_score":dia_str[0]["arousal_score"],
                "suspicion_score":dia_str[0]["suspicion_score"],
                "has_black_terms":dia_str[0]["has_black_terms"],
                "black_terms":dia_str[0]["black_terms"],
                "dialogue":dia_str[0]["dialogue"],
                "judgment":dia_str[0]["judgment"],
                "rationale":dia_str[0]["rationale"]
    }


if __name__ == '__main__':


    import uvicorn

    uvicorn.run(app="main:app", host="0.0.0.0", port=3306)
