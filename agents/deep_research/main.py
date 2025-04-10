from datetime import datetime

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import SelectorGroupChat

from agents.deep_research.tools.bing_search import bing_search_tool
from agents.deep_research.tools.fetch_webpage import fetch_webpage_tool
from config import get_model_client

model_client = get_model_client()

MAX_MESSAGES  = 50

PROMPT_RESERACH = """You are an educational content creation assistant focused on developing comprehensive teaching materials.
The **TIME NOW** is {{time_now}}

Your primary role is to create high-quality course materials based on the outline provided by the teacher.
For each topic in the outline:
1. Use the bing_search tool to find accurate and up-to-date information.
2. Search for relevant examples, case studies, and visual references that can be included.
3. Create engaging educational content structured as:
   - Key learning objectives
   - Core content with clear explanations 
   - Visual aids suggestions (diagrams, charts, or images)
   - Interactive elements (discussions, group activities, hands-on exercises)
   - Learning assessments (quizzes, questions, problem sets)

Break down complex topics into understandable sections. Verify information across multiple sources.
When you find relevant educational resources, extract teaching methodologies and adapt them for the current course.
Present the created materials in markdown format with clear sections.
All content should be in Chinese.
""".replace("{{time_now}}", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

PROMPT_VERIFIER = """You are an educational content verification specialist.
Your role is to:
1. Verify that the course materials are accurate, comprehensive, and aligned with the provided outline
2. Ensure learning objectives are clearly defined and assessments align with these objectives
3. Evaluate if interactive elements are appropriate and engaging for the target audience
4. Check that content is organized logically with clear progression
5. Suggest improvements for clarity, engagement, or pedagogical effectiveness
6. When the content creation is complete, respond with "APPROVED" or if changes are needed, end with "CONTINUE DEVELOPMENT"

Your responses should be structured as:
- Content Assessment (accuracy, completeness, alignment with outline)
- Pedagogical Assessment (effectiveness of teaching approach)
- Interactive Elements Review (engagement potential)
- Assessment Methods Review (appropriateness and alignment with objectives)
- Suggestions for Improvement (if needed)
- CONTINUE DEVELOPMENT or APPROVED

All content should be in Chinese.
"""

PROMPT_SUMMARY = """You are a course materials compiler. Your role is to organize the created educational content into a comprehensive course package. 

Create a well-structured course materials document that includes:
1. Course title and overview
2. Detailed lesson plans with timing suggestions
3. All content sections with clear headings and subheadings
4. Interactive activities with instructions
5. Assessment materials with answer keys where appropriate
6. Visual aids and presentation slides content
7. Additional resources and references

Format the document in clean markdown with appropriate sections, tables, and formatting to make it easy for the teacher to use directly in class.
Your final package should end with the word "TERMINATE" to signal completion.

All content should be in Chinese.
"""

PROMPT_SELECTOR = """
You are coordinating a research team by selecting the team member to speak/act next. The following team member roles are available:
{roles}.
The course_content_creator creates educational content with interactive elements and learning assessments.
The content_reviewer evaluates progress and ensures completeness.
The materials_compiler provides a comprehensive course package, only when content creation is APPROVED.

Given the current context, select the most appropriate next speaker.
The course_content_creator should create and analyze.
The content_reviewer should evaluate progress and guide the content creation (select this role if there is a need to verify/evaluate progress). 
You should ONLY select the materials_compiler role if the content creation is APPROVED by content_reviewer.

Base your selection on:
1. Current stage of content creation
2. Last speaker's findings or suggestions
3. Need for verification vs need for new information
Read the following conversation. Then select the next role from {participants} to play. Only return the role.

{history}

Read the above conversation. Then select the next role from {participants} to play. ONLY RETURN THE ROLE.

"""

text_mention_termination = TextMentionTermination("TERMINATE")
max_messages_termination = MaxMessageTermination(max_messages=MAX_MESSAGES)
termination = text_mention_termination | max_messages_termination

def create_team()->SelectorGroupChat:
    research_assistant = AssistantAgent(
        "course_content_creator",
        description="An agent that creates educational content with interactive elements and learning assessments in Chinese.",
        model_client=model_client,
        model_client_stream=True,
        system_message=PROMPT_RESERACH,
        tools=[fetch_webpage_tool, bing_search_tool])

    verifier = AssistantAgent(
        "content_reviewer",
        description="An agent that reviews educational content for accuracy, effectiveness, and alignment with learning goals in Chinese.",
        model_client=model_client,
        model_client_stream=True,
        system_message=PROMPT_VERIFIER)

    summary_agent = AssistantAgent(
        name="materials_compiler",
        description="Compile and format all educational materials into a comprehensive course package in Chinese.",
        model_client=model_client,
        model_client_stream=True,
        system_message=PROMPT_SUMMARY)
    
    return SelectorGroupChat(
        [research_assistant, verifier, summary_agent],
        termination_condition=termination,
        model_client=model_client,
        selector_prompt=PROMPT_SELECTOR,
        allow_repeated_speaker=True)
