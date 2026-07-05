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
            "emotion_tag": "unknown",
            "sentiment_score": 0,
            "arousal_score": 0
        }


def generate_emotion_indicators_from_dialogue(model,dialogue_list,emotion_tag_kind):
    emotion_tag_str = "\n".join([f"- {item}" for item in emotion_tag_kind])

    template="""你是一个专业的文本情绪分析与风控心理学专家。你的任务是阅读一段对话，并精准提取出该对话的三个情绪与语境参数。

请你严格按照以下预设的列表和评分标准进行判断，严禁编造列表外的内容。

1. 参数限定范围与评分标尺

emotion_tag (情绪标签): 
从以下列表中选择一个最符合当前语境主导情绪的词,若列表里面没有贴切的，严格按照语境对话生成一个贴切的情绪词：
{emotion_tag_list}

sentiment_score (情绪极性分值): 
表示对话整体情绪倾向，必须从以下离散整数中选择：[-2, -1, 0, 1, 2]
-2: 强烈的负向情绪
-1: 轻微的负向情绪
0: 中立（如客观陈述、平静）
1: 轻微的正向情绪
2: 强烈的正向情绪

arousal_score (情绪激烈程度分值): 
表示情绪激烈程度，分值越高代表语气越紧张、催促、刺激或情绪波动越明显。必须从以下整数中选择：[1, 2, 3, 4, 5]
1: 情绪平稳，无明显波澜
2: 轻微的情绪波动
3: 有一定情绪波动，带有一定的情感色彩
4: 情绪较为激烈，伴随明显的紧张、催促或刺激
5: 情绪极度紧张、刺激或产生剧烈波动

2. 分析示例 (Few-Shot Examples)
输入对话："A：你先别急。 甲：不要把款留在一个地方太久，先拆一下。 乙：这样会不会有风险？ 甲：后面会安排人接，不用你问太细。 A：先把这一步做完，再看后面。"
输出结果：
{{
  "emotion_tag": "规避",
  "sentiment_score": -2,
  "arousal_score": 4
}}

输入对话："A：我大概明白你的情况了。 甲：我想咨询正规银行的消费贷。 乙：可以，我们先核验收入和征信情况。 B：所以还是要看具体上下文。"
输出结果：
{{
  "emotion_tag": "平静",
  "sentiment_score": 1,
  "arousal_score": 1
}}

输入对话："A：目前还来得及处理。 A：新手先从低金额做，后面单子会更稳定。 B：具体怎么操作？ A：你这边如果连单了，就要继续补一单才能走完。 B：那如果我现在不想做了呢？ A：别急，先把这一步过完，不然就容易连单。"
输出结果：
{{
  "emotion_tag": "施压",
  "sentiment_score": -1,
  "arousal_score": 4
}}

### 3. 输出格式要求
请仔细阅读下方的[待分析对话]，并严格以 JSON 格式输出上述三个参数。不要包含任何 markdown 代码块标记（如 ```json ），不要输出任何解释性文字，只输出合法的 JSON 字符串。
直接返回：{{"emotion_tag":"...","sentiment_score":0,"arousal_score":0}}

待分析对话:
{dialogue_text}
"""
    prompt_template = ChatPromptTemplate.from_messages([
        ("user", template),
    ])

    result_list=[]

    chain = prompt_template | model | StrOutputParser()
    for dialogue_text in dialogue_list:

        result = chain.invoke({"dialogue_text":dialogue_text,
                               "emotion_tag_list":emotion_tag_str})
        # print(result)
        result=func_result(result)
        group={
            "dialogue":dialogue_text,
            **result
        }
        # print(group)
        result_list.append(group)

    return result_list

