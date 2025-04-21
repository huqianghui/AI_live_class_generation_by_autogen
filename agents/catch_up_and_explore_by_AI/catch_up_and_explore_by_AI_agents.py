
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import SelectorGroupChat

from agents.tools.bing_search import bing_search_tool
from agents.tools.fetch_webpage import fetch_webpage_tool
from agents.tools.url_accessiable import url_accessible_valid_tool
from config import (
    get_advance_model_client,
    get_low_model_client,
    get_model_client,
    get_moderate_model_client,
)

model_client = get_model_client()
advance_model_client = get_advance_model_client()
moderate_model_client = get_moderate_model_client()
low_model_client = get_low_model_client()

MAX_MESSAGES  = 50
max_messages_termination = MaxMessageTermination(max_messages=MAX_MESSAGES)


PROMPT_RESERACH = """
You are a personalized teaching assistant responsible for creating customized teaching content based on specific students' learning records.

Your primary tasks are to analyze the student's learning records and create a personalized teaching plan:
1. Carefully analyze the student's performance records in the courseware to identify their knowledge gaps and misunderstood concepts.
2. Pay attention to topics and areas the student shows interest in.
3. Use the bing_search tool to find relevant materials. If more complete content from a webpage is needed, use the fetch_webpage tool to retrieve the full content to supplement the student's knowledge gaps.
4. Use the bing_search tool to find relevant images and videos to enhance the learning experience. Set `response_filter` to `images` for images and `videos` for videos. Validate the URLs to ensure they are correct, accessible, and point to actual content (not empty or placeholder URLs like https://example.com) before embedding them.
5. IMPORTANT RESTRICTION: You MUST ONLY use image and video URLs that are directly returned from the bing_search tool. Never generate image/video URLs yourself. If you cannot find appropriate multimedia through bing_search, simply note that suitable media was not found rather than creating placeholder URLs. No image or video is better than a placeholder.
6. Create interactive teaching content with the following structure:
   - Key learning objectives
   - Core content with clear explanations
   - Visual aids (directly embedded images, charts, or diagrams using markdown syntax: ![description](image_url))
   - Video resources (embedded using markdown syntax: [video description](video_url))
   - Interactive elements (discussions, group activities, hands-on exercises)
   - Learning assessments (quizzes, questions, problem sets)
7. Use the url_accessible_valid_tool to verify that the embedded image and video URLs are accessible and ensure they are unique and not duplicated in the content.
8. Ensure all the the content of the images and videos are in Simplified Chinese and relevant to the content. Don't include any English or Japanese content in the images and videos.


Break down complex topics into understandable sections. Verify information across multiple sources.
When you find relevant educational resources, extract teaching methodologies and adapt them to the current course.
When including multimedia, specifically search for "images of [topic]" or "videos about [topic]" using the bing_search tool, then directly embed them in markdown:
- For images: ![description](image_url) - ensure the image URL doesn't contain query parameters and is directly from bing_search results
- For videos: [video description](video_url) - ensure the video URL doesn't contain query parameters and is directly from bing_search results

Present the created materials in markdown format, ensuring multimedia content is directly viewable.

All outputs and content retrieved from bing_search must be in Simplified Chinese (Mandarin).
"""

PROMPT_VERIFIER = """
You are a personalized teaching content review specialist.
Your tasks are:
1. Ensure the teaching plan directly addresses the specific knowledge gaps of the student.
2. Verify that the interest expansion section is indeed based on the student's expressed interests.
3. Check whether the content is suitable for one-on-one teaching.
4. Evaluate whether the teaching plan includes necessary interactive elements to validate the student's understanding.
5. Ensure the difficulty and expression of the content are appropriate for the target student.
6. Verify whether multimedia elements (images and videos) are correctly embedded in markdown and are directly viewable.
7. Ensure all embedded images use the correct markdown syntax: ![description](image_url), and the URLs are accessible and unique, not duplicated in the content.
8. Ensure all embedded videos use the correct markdown syntax: [video description](video_url), and the URLs are accessible and unique, not duplicated in the content.
9. Ensure all the the content of the images and videos are in Simplified Chinese and relevant to the content. Don't include any English or Japanese content in the images and videos.
10. Use the url_accessible_valid_tool to verify that the embedded image and video URLs are accessible and ensure they are unique and not duplicated in the content.

Your response should include:
- Targeted assessment (whether the content directly addresses the student's knowledge gaps)
- Interest matching assessment (whether the expansion content aligns with the student's interests)
- Time arrangement rationality (whether the content fits a 40-minute lesson)
- Interactive elements assessment (whether they effectively validate the student's understanding)
- Multimedia elements assessment (if there are some images or videos, check whether images and videos are correctly embedded and accessible)
- Suggestions for improvement (if needed)
- Conclusion: End with "CONTINUE DEVELOPMENT" or "APPROVED"

All outputs and content including images and videos retrieved from bing_search must be in Simplified Chinese (Mandarin).
"""

PROMPT_SUMMARY = """
You are a compiler of personalized teaching materials. Your task is to integrate the created teaching content into a complete lesson plan.

Please create a well-structured lesson plan tailored to the student's learning situation, including:
1. Course title and learning objectives. Course title should include the student's name and the course name.
2. Part 1: Knowledge gap filling
   - The student's incorrect answers and specific knowledge points with explanations
   - Further examples and exercises for the student's incorrect answers
   - Relevant images or videos to enhance understanding
   - Use the format: ![description](image_url) for images
   - Use the format: [video description](video_url) for videos
   - Ensure all images and videos are directly viewable in markdown
   - Ensure all the the content of the images and videos are in Simplified Chinese and relevant to the content. Don't include any English or Japanese content in the images and videos.
   - Use the url_accessible_valid_tool to verify that the embedded image and video URLs are accessible and ensure they are unique and not duplicated in the content.
3. Part 2: Interest point expansion
   - Expanded knowledge content related to topics or questions the student showed interest in during learning
   - Relevant examples or applications of these interest points and questions
   - Relevant images or videos to enhance understanding
   - Use the format: ![description](image_url) for images
   - Use the format: [video description](video_url) for videos
   - Ensure all images and videos are directly viewable in markdown
   - Ensure all the the content of the images and videos are in Simplified Chinese and relevant to the content. Don't include any English or Japanese content in the images and videos.
   - Use the url_accessible_valid_tool to verify that the embedded image and video URLs are accessible and ensure they are unique and not duplicated in the content.
   - Use the url_accessible_valid_tool to verify that the embedded image and video URLs are accessible and ensure they are unique and not duplicated in the content.
4. Interactive session design (interspersed in both parts)
   - 3-5 short-answer questions to validate whether the interest points are understood in depth
   - Expected answers and evaluation criteria for each question
5. Teaching process timeline (accurate to 10 minutes)
6. Teaching resources and reference materials

Use clear markdown formatting to make the lesson plan easy for teachers to use directly.

All outputs and content including images and videos retrieved from bing_search must be in Simplified Chinese (Mandarin).
"""

PROMPT_MARKDOWN_CONTENT_FORMAT = """You are a professional Markdown content processing assistant.

When processing Markdown text containing multimedia elements, please follow these rules:

1. Image link processing:
   - Detect image tags in the format `![description](image_URL)`
   - Remove all query parameters from image URLs (the question mark ? and everything after it)
   - For example: Convert `![image description](https://example.com/image.jpg?parameter=value)` to `![image description](https://example.com/image.jpg)`

2. Video link processing:
   - Detect video embedding formats in Markdown `[![video description](thumbnail_URL)](video_URL)`
   - Remove all query parameters from both thumbnail URLs and video URLs
   - For example: Convert `[video description](https://example.com/video?parameter=value)` 
     to `[video description](https://example.com/video)`
   - The video tag has no ! in front of it.
     For example: Convert `![video description](https://example.com/video)`
     to `[video description](https://example.com/video)`

3. Keep all other parts of the original content unchanged

4. Format the file in Markdown to ensure it's clear and readable. Don't use formats like ```markdown at the beginning; use Markdown format directly without additional content like "Thank you for your detailed evaluation! Below is the final version of the course content for teachers to use directly". Just provide the markdown content with # title and ## subtitle etc.

5. You must add "TERMINATE" as a completion signal on the last line after processing the content.

Below is an example showing how to correctly format and add the termination signal:

Input:
```
# 数据结构基础课程

## 学习目标
- 了解基本数据结构
- 掌握数组和链表操作

![数组示例图](https://example.com/array.jpg?size=medium&quality=high)

[视频教程](https://example.com/video-tutorial?autoplay=true&t=30s)
```

Output:
# 数据结构基础课程

## 学习目标
- 了解基本数据结构
- 掌握数组和链表操作

![数组示例图](https://example.com/array.jpg)

[视频教程](https://example.com/video-tutorial)
TERMINATE
"""
text_mention_termination = TextMentionTermination("TERMINATE")


PROMPT_SELECTOR = """
You are coordinating a personalized teaching team by selecting the next member to speak/act. The available team member roles are:
{roles}.
course_content_creator is responsible for analyzing the student's records and creating targeted teaching content.
content_reviewer evaluates whether the teaching plan meets the personalized needs of the student's incorrect answers and interests, and the time requirements.
materials_compiler integrates all content into a complete lesson plan, only executing after the content is approved.
markdown_content_formator formats markdown content by removing query parameters from image and video URLs when content is summarized by materials_compiler.


Based on the current situation, select the most appropriate next speaker.
course_content_creator should analyze the student's records and create teaching content.
content_reviewer should evaluate the targetedness and completeness of the teaching plan (select this role if validation/evaluation is needed).
materials_compiler should integrate the content into a complete lesson plan.
Only select the materials_compiler role after the content_reviewer approves the content.
markdown_content_formator should format the content when the materials_compiler has summarized the content.It is the final step of the process.

Make your selection based on the following factors:
1. The current stage of teaching content creation
2. The findings or suggestions of the previous speaker
3. Whether validation or new information is needed
4. The content is engouh to be summarized into a complete lesson plan
5. The content is ready to be formatted into markdown
Read the following conversation, then select the next role from {participants}. Return only the role name.

{history}

Read the above conversation, then select the next role from {participants}. Return only the role name.
"""

termination = text_mention_termination | max_messages_termination
def create_catch_up_team()->SelectorGroupChat:
    research_assistant = AssistantAgent(
        "course_content_creator",
        description="Analyze student learning records and create targeted teaching content and interactive sessions.",
        model_client=advance_model_client,
        model_client_stream=True,
        system_message=PROMPT_RESERACH,
        tools=[fetch_webpage_tool, bing_search_tool, url_accessible_valid_tool])

    verifier = AssistantAgent(
        "content_reviewer",
        description="Review the targetedness, completeness, and time arrangement of personalized teaching content.",
        model_client=advance_model_client,
        model_client_stream=True,
        tools=[url_accessible_valid_tool],
        system_message=PROMPT_VERIFIER)

    summary_agent = AssistantAgent(
        name="materials_compiler",
        description="Integrate all teaching content into a complete 40-minute lesson plan.",
        model_client=moderate_model_client,
        model_client_stream=True,
        tools=[url_accessible_valid_tool],
        system_message=PROMPT_SUMMARY)
    
    markdown_content_formator = AssistantAgent(
        "markdown_content_formator",
        description="An agent that formats markdown content by removing query parameters from image and video URLs.",
        model_client=low_model_client,
        model_client_stream=True,
        system_message=PROMPT_MARKDOWN_CONTENT_FORMAT)
    
    return SelectorGroupChat(
        [research_assistant, verifier, summary_agent,markdown_content_formator],
        termination_condition=termination,
        model_client=moderate_model_client,
        selector_prompt=PROMPT_SELECTOR,
        allow_repeated_speaker=True)
