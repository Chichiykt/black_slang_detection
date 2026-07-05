import json
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def func_result(result):
    try:

        if "</think>" in result:
            f_comments_str = result.split("</think>")[-1].strip()
        else:
            f_comments_str = result.strip()


        pattern = r'```json(.*?)```'
        matches = re.search(pattern, f_comments_str, re.DOTALL)

        if matches:
            json_str = matches.group(1).strip()
        else:
            # 没有代码块 → 直接尝试把全文当JSON（兼容模型直接输出JSON）
            json_str = f_comments_str


        json_dict = json.loads(json_str)
        return json_dict

    except Exception as e:

        print(f"解析失败，使用默认值: {str(e)[:100]}")
        return {
            "judgment": "unknown",
            "risk_level": "low_or_contextual",
            "suspicion_score": 0
        }

def generate_risk_indicators_from_dialogue(model,dialogue_str_list,risk_list,judgment_list,temperature=0.6):
    risk_str = "\n".join([f"- {item}" for item in risk_list])
    judgment_str = "\n".join([f"- {item}" for item in judgment_list])
    template="""你是一个专业的风控审查与对话分析专家。你的任务是分析提供的对话内容（dialogue），并提取或推断出三个维度的信息：判别结论（judgment）、风险层级（risk_level）和可疑分数（suspicion_score）。

请严格遵守以下约束条件进行分析：

1. 字段约束
judgment:基于综合标注信息给出的自然语言判别结论,必须且只能从以下列表中选择一个：{judgment_list}。

risk_level:必须且只能从以下列表中选择一个：{risk_list}。


2. 打分指南 (suspicion_score: 0-100)
请根据对话的可疑程度给出 0 到 100 的整数评分:
0-20分 (正常/安全):纯粹的科普、正常的金融客服服务、反诈宣传。例如：“我想咨询正规银行的消费贷。/可以，我们先核验收入和征信情况。” (得分参考: 5)
21-50分 (低风险/需结合上下文):出现了敏感词汇，但语境是探讨、咨询合规边界或日常生活。例如：“合同里如果出现料卡这种费用，是否合规？/通常要高度怀疑。” (得分参考: 25)
51-80分 (中高风险/疑似黑话):对话存在诱导、催促，暗示规避审查或非常规操作。例如：“先补一单才能走完，别急，先把这一步过完，不然就容易卡单。” (得分参考: 75)
81-100分 (高风险/典型黑话):明显的黑产交易、杀猪盘、洗钱或诈骗话术，极力规避追踪。例如：“不要把款留在一个地方太久，先拆一下，后面会安排车手接，不用你问太细。” (得分参考: 95)

3. 输出格式
只输出"纯JSON字符"
不要加任何解释、不要加```json、不要加```、不要加多余文字
直接返回：{{"judgment":"...","risk_level":"...","suspicion_score":0}}

待分析对话:
{dialogue_text}
"""
    prompt_template = ChatPromptTemplate.from_template(template)

    # model = OllamaLLM(base_url=__base_url, model=__llm_model, temperature=temperature)

    chain = prompt_template | model | StrOutputParser()

    dialogue_list=[]

    for dialogue_text in dialogue_str_list:

        result = chain.invoke({"judgment_list":judgment_str,
                                "risk_list":risk_str,
                               "dialogue_text":dialogue_text})
        result=func_result(result)
        dialogue_text={
            "dialogue":dialogue_text,
            **result
        }

        dialogue_list.append(dialogue_text)

    return dialogue_list

