import asyncio
import os
import tempfile
import time
import traceback
from datetime import datetime

import chainlit as cl
from autogen_agentchat.base import TaskResult
from autogen_agentchat.messages import (
    ModelClientStreamingChunkEvent,
    StopMessage,
    TextMessage,
)
from autogen_agentchat.teams import SelectorGroupChat
from autogen_core import CancellationToken

from agents.deep_research.main import create_team
from agents.file_processor.main import process_file
from config import DEEP_RESEARCH_AGENT


@cl.set_chat_profiles
async def chat_profile():
    return [
        cl.ChatProfile(
            name=DEEP_RESEARCH_AGENT,
            markdown_description="生成中国小学语文教学内容，包括诗词鉴赏、阅读理解和互动练习。",
            icon="public/icons/deep_research.png",
            starters=[
                cl.Starter(
                    label="李白《静夜思》教学",
                    message="请为小学三年级学生创建一节关于李白《静夜思》的课程。包括诗词背景介绍、重点字词解释、诗句赏析、朗读指导、互动活动和学习检测题目。",
                ),
                cl.Starter(
                    label="杜甫《春夜喜雨》教学",
                    message="为小学四年级学生设计一节杜甫《春夜喜雨》的教学课件，包含诗人简介、诗词解析、情景想象活动、诗词朗诵技巧、课堂互动环节和课后习题。",
                ),
                cl.Starter(
                    label="成语故事《守株待兔》教学",
                    message="为小学二年级学生创建一节关于成语故事《守株待兔》的教学内容，包括故事原文、生字词解释、故事寓意分析、角色扮演活动、课堂提问和课后练习。",
                ),
            ],
        ),
    ]

@cl.on_chat_start
async def on_chat_start():
    deep_research_team = create_team()
    
    cl.user_session.set(DEEP_RESEARCH_AGENT, deep_research_team)

@cl.on_message  # type: ignore
async def chat(message: cl.Message) -> None:
    # Check if there are files uploaded
    files = message.elements
    
    if files:
        try:
            # Process uploaded files with timeout handling
            await cl.Message(content="正在处理上传的文件，请稍候...").send()
            # Use a longer timeout for file processing
            await asyncio.wait_for(process_uploaded_files(files, message), timeout=120.0)
        except asyncio.TimeoutError:
            await cl.Message(content="文件处理超时，请尝试将文件拆分为较小的部分或减少文件数量。").send()
        except Exception as e:
            error_msg = f"文件处理失败: {str(e)}\n\n"
            error_trace = traceback.format_exc()
            print(f"File processing error: {error_trace}")
            await cl.Message(content=error_msg + "请重试或联系系统管理员。").send()
    else:
        # Process text request directly
        deep_research_team = cl.user_session.get(DEEP_RESEARCH_AGENT)
        await run_stream_team(
            deep_research_team,
            message,
        )

async def process_uploaded_files(files, message: cl.Message):
    deep_research_team = cl.user_session.get(DEEP_RESEARCH_AGENT)
    
    combined_content = ""
    file_count = len(files)
    
    await cl.Message(content=f"开始处理 {file_count} 个文件...").send()
    
    # Process each file
    for i, file in enumerate(files):
        try:
            temp_dir = tempfile.mkdtemp()
            temp_file_path = os.path.join(temp_dir, file.name)
            
            # Show progress
            await cl.Message(content=f"处理文件 {i+1}/{file_count}: {file.name}").send()
            
            # Save the uploaded file directly to temp directory
            try:
                # Use the file's path attribute instead of trying to get bytes
                original_file_path = None
                if hasattr(file, 'path') and file.path:
                    # Option 1: Copy the file from its current location to our temp path
                    import shutil
                    shutil.copy(file.path, temp_file_path)
                    # Store the original file path for saving markdown alongside it
                    original_file_path = file.path
                else:
                    # Fallback for versions where path might not be available
                    raise Exception(f"Cannot access file: File path not available")
                
                # Get file size for limit check
                file_size = os.path.getsize(temp_file_path) / (1024 * 1024)  # Size in MB
                
            except Exception as e:
                error_msg = f"无法保存文件: {str(e)}"
                print(f"File saving error: {traceback.format_exc()}")
                await cl.Message(content=error_msg).send()
                continue
                
            # Check if file is too large
            if file_size > 50:  # 50MB limit
                await cl.Message(content=f"文件 {file.name} 太大 ({file_size:.1f}MB)，请上传50MB以下的文件。").send()
                continue
            
            # Process the file and convert to markdown - pass the original file path
            await cl.Message(content=f"正在转换 {file.name} 为教学内容...").send()
            error, content = process_file(temp_file_path, original_file_path=original_file_path)
            
            if error:
                await cl.Message(content=f"处理 {file.name} 时出错: {error}").send()
                continue
            
            combined_content += f"\n\n## Content from {file.name}\n\n{content.markdown}"
            
            # Clean up
            os.remove(temp_file_path)
            os.rmdir(temp_dir)
            
            await cl.Message(content=f"✅ 文件 {file.name} 处理成功").send()
            
        except Exception as e:
            await cl.Message(content=f"处理文件 {file.name} 时发生错误: {str(e)}").send()
            print(f"Error processing file {file.name}: {traceback.format_exc()}")
    
    if combined_content:
        # Send the extracted content to the file processor agent
        async with cl.Step(name="处理文件内容") as step:
            # Fix: Create a separate message instead of passing string to step.send()
            await cl.Message(content=f"从 {file_count} 个文件中提取教学内容要求...").send()
            
            try:
                # Fix: Create a separate message instead of passing string to step.send()
                await cl.Message(content="文件处理完成。正在基于提取的内容生成教学材料...").send()
                
                # Now run the deep research team with the processed content
                new_message = cl.Message(content=combined_content)
                await run_stream_team(deep_research_team, new_message)
            except asyncio.TimeoutError:
                await cl.Message(content="内容处理超时，请尝试减少文件数量或拆分为较小的请求。").send()
            except Exception as e:
                error_msg = f"内容处理失败: {str(e)}"
                print(f"Content processing error: {traceback.format_exc()}")
                await cl.Message(content=error_msg).send()
    else:
        await cl.Message(content="无法从上传的文件中提取内容。请确保文件格式正确且内容可读。").send()

async def run_stream_team(team=SelectorGroupChat, message: cl.Message | None = None):
    executing = False

    async with cl.Step(name="Executing") as executing_step:
        start = time.time()

        final_answer = cl.Message(content="")

        try:
            # Create a clean cancellation token
            cancellation_token = CancellationToken()
            
            # Use the async generator directly instead of trying to wrap it in a task
            async for msg in team.run_stream(
                task=[TextMessage(content=message.content, source="user")],
                cancellation_token=cancellation_token,
            ):
                try:
                    # Check message type properly
                    if isinstance(msg, ModelClientStreamingChunkEvent):
                        # Only process message if content exists
                        if not hasattr(msg, 'content') or msg.content is None:
                            continue
                            
                        # Extract content and handle TERMINATE keyword
                        content = msg.content
                        if isinstance(content, str) and "TERMINATE" in content:
                            # Remove TERMINATE and everything after it
                            content = content.split("TERMINATE")[0].strip()
                                
                        # Process based on source
                        if msg.source != "materials_compiler":
                            executing = True
                            if content:  # Only stream non-empty content
                                await executing_step.stream_token(content)
                        else:
                            executing = False
                            executed_for = round(time.time() - start)
                            executing_step.name = f"Executed for {executed_for}s"
                            await executing_step.update()
                            if content:  # Only stream non-empty content
                                await final_answer.stream_token(content)
                    
                    elif isinstance(msg, StopMessage):
                        # Handle stop messages properly
                        print(f"Received StopMessage: {msg.content if hasattr(msg, 'content') else 'No content'}")
                        # Don't try to serialize the entire StopMessage object
                        if hasattr(msg, 'content') and isinstance(msg.content, str):
                            content = msg.content
                            if "TERMINATE" in content:
                                content = content.split("TERMINATE")[0].strip()
                            if content:
                                await final_answer.stream_token(f"\n\n{content}")
                                
                    elif isinstance(msg, TaskResult):
                        # Process task results if needed
                        pass
                    
                    elif executing_step is not None and msg is not None:
                        # Handle any other message types safely
                        try:
                            await executing_step.send()
                        except Exception as send_error:
                            print(f"Error sending executing step: {str(send_error)}")
                        
                except Exception as token_error:
                    # Log the error but continue processing
                    print(f"Error processing message chunk: {str(token_error)}")
                    print(traceback.format_exc())
                    continue
                    
        except Exception as stream_error:
            # Handle other stream errors
            print(f"Error in message stream: {str(stream_error)}")
            print(traceback.format_exc())
            await cl.Message(content=f"生成内容时出错: {str(stream_error)}").send()
            
    # Only try to create MD and PDF if we have content
    if final_answer.content:
        try:
            # Clean up content before saving
            clean_content = final_answer.content
            if "TERMINATE" in clean_content:
                clean_content = clean_content.split("TERMINATE")[0].strip()
            
            # Create timestamp for filenames
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save markdown to public/md directory
            os.makedirs("public/md", exist_ok=True)
            md_filename = f"public/md/course_materials_{timestamp}.md"
            with open(md_filename, "w", encoding="utf-8") as md_file:
                md_file.write(clean_content)
            
            # Generate PDF
            pdf_file = md_to_pdf(clean_content)
            
            # Add both links to the response
            await final_answer.stream_token(f"\n\nMarkdown: [{os.path.basename(md_filename)}]({md_filename})")
            await final_answer.stream_token(f"\n\nPDF: [{os.path.basename(pdf_file)}]({pdf_file})")
        except Exception as file_error:
            print(f"Error creating files: {file_error}")
            print(traceback.format_exc())
            await final_answer.stream_token("\n\n无法创建文件，请检查生成的内容。")
    
    # Make sure to send the final answer even if empty
    await final_answer.send()

def md_to_pdf(md: str) -> str:
    from markdown_pdf import MarkdownPdf, Section

    os.makedirs("public/pdfs", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"public/pdfs/course_materials_{timestamp}.pdf"

    # Clean up the content - no need to check for TERMINATE again as it should be handled earlier
    content = md
    
    # Ensure we have content
    if not content:
        content = "# 无内容 \n\n请检查生成过程，内容生成失败。"
        
    pdf = MarkdownPdf()
    
    # Set only standard PDF metadata fields
    pdf.meta["title"] = "中国小学语文教学内容"
    pdf.meta["keywords"] = "教学内容,小学语文,教案"
    pdf.meta["subject"] = "自动生成的中国小学语文教学内容"
    
    # Add content as a section with table of contents
    pdf.add_section(
        Section(
            content,
            toc=True,
        )
    )
    
    try:
        pdf.save(filename)
        return filename
    except Exception as e:
        # Log error and save a simplified version if there are issues
        print(f"Error saving PDF: {str(e)}")
        print(traceback.format_exc())
        
        # Create a simple text file instead
        text_filename = filename.replace('.pdf', '.txt')
        with open(text_filename, "w", encoding="utf-8") as f:
            f.write(content)
        return text_filename
