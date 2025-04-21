from datetime import datetime

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import SelectorGroupChat

from agents.tools.bing_search import bing_search_tool
from agents.tools.fetch_webpage import fetch_webpage_tool
from config import get_model_client

model_client = get_model_client()

MAX_MESSAGES  = 50

PROMPT_RESERACH = """你是一个专注于个性化教学的助手，负责根据具体学生的学习记录创建定制化的教学内容。

你的主要任务是分析该学生的学习记录，并创建定制化的教学计划：
1. 仔细分析该学生在课件中的表现记录，找出他们的知识缺口和不理解的概念
2. 注意该学生表现出兴趣的话题和领域
3. 使用bing_search工具查找相关资料，如果需要网页更完整的内容使用fetch_webpage工具获取网页完整内容来补充学生学习中的知识缺口
4. 根据该学生的兴趣点，搜索额外的相关知识进行拓展
5. 在标题中突出学生名字，兴趣点和问题点内容。

创建一个完整的教学计划，包括：
- 第一部分：针对性复习
  * 列出该学生掌握不好的关键知识点
  * 为每个知识点提供清晰简洁的解释
  * 设计简单的例子帮助理解

- 第二部分：兴趣拓展
  * 基于该学生表现出兴趣的点进行知识拓展
  * 提供与课程相关但更深入或更广泛的内容
  * 包含有趣的实例、应用场景或小故事

所有内容应以中文呈现，适合一对一教学场景。
"""

PROMPT_VERIFIER = """你是一个个性化教学内容审核专家。
你的任务是：
1. 确保教学计划直接针对该学生的具体知识缺口
2. 验证兴趣拓展部分确实基于该学生表现出的兴趣点
3. 检查内容是否适合一对一教学
4. 评估教学计划是否包含了必要的互动元素来验证学生理解
5. 确保内容的难度和表达方式适合该目标学生

你的回应应包含：
- 针对性评估（内容是否直接解决该学生的知识缺口）
- 兴趣匹配度评估（拓展内容是否符合该学生兴趣）
- 时间安排合理性（内容是否适合40分钟课程）
- 互动元素评估（是否有效验证该学生的理解）
- 改进建议（如有需要）
- 结论：以"继续完善"或"已批准"结束

所有内容应以中文呈现。
"""

PROMPT_SUMMARY = """你是个性化教学材料的整合者。你的任务是将创建的针对该学生教学内容整合为一个完整的教案。

请创建一个结构清晰的针对该学生学习情况下，进一步加强的教案文档，包括：
1. 课程标题和学习目标
2. 第一部分：知识查缺补漏
   - 该学生的答错的问题和具体知识点及解释
   - 该学生的答错的问题进一步示例和练习
3. 第二部分：兴趣点拓展
   - 该学生在学习过程中感兴趣或者提出问题的拓展知识内容
   - 以及这些兴趣点和问题的相关实例或应用
4. 互动环节设计（穿插在两部分中）
   - 3-5个简答题，用于验证兴趣点是否得到加深理解
   - 每个问题的预期答案和评估标准
5. 教学流程时间线（精确到10分钟）
6. 教学资源和参考材料

使用清晰的markdown格式，使教案易于教师直接使用。
你的最终文档应以"TERMINATE"一词结束，表示完成。

所有内容应以中文呈现。
"""

PROMPT_SELECTOR = """
你正在协调一个个性化教学团队，通过选择下一位成员发言/行动。可用的团队成员角色有：
{roles}。
course_content_creator负责分析该学生记录并创建针对性的教学内容。
content_reviewer评估教学计划是否满足针对该学生的错题点和感兴趣点等的个性化需求和时间要求。
materials_compiler整合所有内容为完整的教案，仅在内容被批准后执行。

根据当前情况，选择最合适的下一位发言人。
course_content_creator应分析学生记录并创建教学内容。
content_reviewer应评估教学计划的针对性和完整性（如需验证/评估进度，选择此角色）。
仅当content_reviewer批准内容后，才选择materials_compiler角色。

基于以下因素做出选择：
1. 当前教学内容创建阶段
2. 上一位发言者的发现或建议
3. 是否需要验证或需要新信息
阅读以下对话，然后从{participants}中选择下一个角色。只返回角色名称。

{history}

阅读上述对话，然后从{participants}中选择下一个角色。只返回角色名称。
"""

text_mention_termination = TextMentionTermination("TERMINATE")
max_messages_termination = MaxMessageTermination(max_messages=MAX_MESSAGES)
termination = text_mention_termination | max_messages_termination

def create_catch_up_team()->SelectorGroupChat:
    research_assistant = AssistantAgent(
        "course_content_creator",
        description="分析学生学习记录，创建针对性教学内容和互动环节的智能体。",
        model_client=model_client,
        model_client_stream=True,
        system_message=PROMPT_RESERACH,
        tools=[fetch_webpage_tool, bing_search_tool])

    verifier = AssistantAgent(
        "content_reviewer",
        description="审核个性化教学内容的针对性、完整性和时间安排的智能体。",
        model_client=model_client,
        model_client_stream=True,
        system_message=PROMPT_VERIFIER)

    summary_agent = AssistantAgent(
        name="materials_compiler",
        description="将所有教学内容整合为完整40分钟教案的智能体。",
        model_client=model_client,
        model_client_stream=True,
        system_message=PROMPT_SUMMARY)
    
    return SelectorGroupChat(
        [research_assistant, verifier, summary_agent],
        termination_condition=termination,
        model_client=model_client,
        selector_prompt=PROMPT_SELECTOR,
        allow_repeated_speaker=True)
