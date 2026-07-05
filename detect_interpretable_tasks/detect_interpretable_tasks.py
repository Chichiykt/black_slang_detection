import json
import os
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

def generate_black_terms_indicators_from_dialogue(model,dialogue_str_list,black_terms_list,temperature=0.6):
    black_terms_str = "\n".join([f"- {item}" for item in black_terms_list])
    template="""你是一个专业的风控审查与对话分析专家。你的任务是分析提供的对话内容（dialogue），并提取出两个维度的信息：命中的黑话词（black_terms）和判别理由（rationale）。

请严格遵守以下约束条件进行分析：

1. 命中黑话词 (black_terms):
请仔细阅读对话，优先从以下已知的行业黑话词库中匹配对话里出现的黑话词：
{black_terms_list}
如果对话中存在明显的黑产、违规、诱导等黑话，但不在上述列表中，请从对话中精准提取出来。
- 如果提取到多个黑话词，请严格用英文分号 ";" 分隔。
- 如果对话是正常的业务咨询、日常交流或反诈科普，没有包含任何黑话，请输出空字符串 ""。

2. 判别理由 (rationale):
请对该样本为何被判定为存在风险（或为何包含这些黑话），或者为何被判定为正常安全进行简要解释。要求逻辑清晰、客观专业。

3. 输出格式要求:
请严格以 JSON 格式输出，不要包含任何额外的解释性文本（如“好的”、“根据分析”等），不要使用 Markdown 代码块包裹，直接输出 JSON 字符串:
{{
  "black_terms": "词1;词2",
  "rationale": "简要的判别理由解释"
}}

待分析对话:
{dialogue_text}
"""
    prompt_template = ChatPromptTemplate.from_template(template)
    chain = prompt_template | model | StrOutputParser()

    dialogue_list=[]

    for dialogue_text in dialogue_str_list:

        result = chain.invoke({"black_terms_list":black_terms_str,
                               "dialogue_text":dialogue_text})
        result=func_result(result)
        group={
            "dialogue":dialogue_text,
            **result
        }
        dialogue_list.append(group)

    return dialogue_list