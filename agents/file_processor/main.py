import os
import tempfile
import traceback
from pathlib import Path

from autogen_agentchat.agents import AssistantAgent
from markitdown import MarkItDown

from config import get_model_client


def process_file(file_path, original_file_path=None):
    """Convert various file formats to markdown and save alongside original files"""
    # Check if file exists and is readable
    if not os.path.isfile(file_path):
        return f"Error: File does not exist at path: {file_path}", None
    
    file_extension = Path(file_path).suffix.lower()
    file_name = Path(file_path).name
    file_stem = Path(file_path).stem

    markitdown = MarkItDown()

    try:
        # Use markitdown to convert different formats to markdown
        if file_extension in ['.docx', '.doc']:
            try:
                result = markitdown.convert(file_path)
                if result is None:
                    return f"Failed to convert DOCX file: {file_path}. Empty result returned.", None
            except Exception as docx_err:
                print(f"DOCX conversion error: {str(docx_err)}")
                print(traceback.format_exc())
                return f"Error converting DOCX file: {str(docx_err)}", None
        elif file_extension in ['.pptx', '.ppt']:
            try:
                result = markitdown.convert(file_path)
                if result is None:
                    return f"Failed to convert PPTX file: {file_path}. Empty result returned.", None
            except Exception as pptx_err:
                print(f"PPTX conversion error: {str(pptx_err)}")
                print(traceback.format_exc())
                return f"Error converting PPTX file: {str(pptx_err)}", None
        elif file_extension == '.pdf':
            try:
                result = markitdown.convert(file_path)
                if result is None:
                    return f"Failed to convert PDF file: {file_path}. Empty result returned.", None
            except Exception as pdf_err:
                print(f"PDF conversion error: {str(pdf_err)}")
                print(traceback.format_exc())
                return f"Error converting PDF file: {str(pdf_err)}", None
        elif file_extension in ['.md', '.markdown']:
            # If already markdown, just read the file
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                # Create a simple object that mimics markitdown result structure
                class MarkdownResult:
                    def __init__(self, content):
                        self.text_content = content
                        self.markdown = content
                result = MarkdownResult(content)
            except UnicodeDecodeError:
                # Try with a different encoding if utf-8 fails
                try:
                    with open(file_path, 'r', encoding='latin-1') as f:
                        content = f.read()
                    class MarkdownResult:
                        def __init__(self, content):
                            self.text_content = content
                            self.markdown = content
                    result = MarkdownResult(content)
                except Exception as encoding_err:
                    return f"Error reading file with alternative encoding: {str(encoding_err)}", None
        else:
            return f"Unsupported file format: {file_extension}", None
            
        # Save markdown content alongside the original file if path is provided
        if original_file_path and hasattr(result, 'text_content'):
            try:
                # Get the directory of the original file
                original_dir = os.path.dirname(original_file_path)
                
                # Create filename with .md extension
                markdown_filename = f"{file_stem}.md"
                markdown_path = os.path.join(original_dir, markdown_filename)
                
                # Write the markdown content to file
                with open(markdown_path, 'w', encoding='utf-8') as md_file:
                    md_file.write(result.text_content)
                    
                print(f"Saved markdown file to: {markdown_path}")
            except Exception as save_err:
                print(f"Error saving markdown file: {str(save_err)}")
                print(traceback.format_exc())
                # Continue even if saving fails - we'll still return the result
            
        return None, result
    except Exception as e:
        print(f"General file processing error: {str(e)}")
        print(traceback.format_exc())
        return f"Error processing file: {str(e)}", None


def create_file_processor_agent():
    model_client = get_model_client()
    
    PROMPT_FILE_PROCESSOR = """You are a file content analyzer and requirements extractor.
    
Your role is to:
1. Analyze the content extracted from uploaded files (Word, PowerPoint, PDF, etc.)
2. Identify key educational requirements, learning objectives, and content structure
3. Format this information into a clear, structured request for course material generation
4. Preserve important content from the original document while organizing it logically

For Chinese elementary school language education content:
- Identify key vocabulary, poems, stories or themes
- Extract teaching requirements or educational standards if present
- Note any specific activities, assessments, or teaching methods mentioned
- Format content to ensure Chinese characters and special formatting are preserved

Present your analysis as a well-structured request that can be used to generate 
comprehensive teaching materials for Chinese elementary language education.
"""

    file_processor_agent = AssistantAgent(
        "file_processor",
        description="An agent that processes uploaded files and extracts requirements for course generation.",
        model_client=model_client,
        model_client_stream=True,
        system_message=PROMPT_FILE_PROCESSOR
    )
    
    return file_processor_agent
