from fastapi import FastAPI, HTTPException, File, UploadFile, Request, Form
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import json
import requests
import aiohttp
import re
import asyncio
import pathlib
from typing import List, AsyncGenerator, Optional, Dict, Set

load_dotenv()

# 创建 FastAPI 应用实例
app = FastAPI()

# 添加CORS中间件，允许前端进行API调用
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头
)

# 设置静态文件和模板
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# xAI API配置
XAI_API_KEY = os.getenv("XAI_API_KEY", "") # 从 .env 读取 API Key
XAI_API_URL = "https://api.x.ai/v1/chat/completions"
XAI_MODEL = os.getenv("XAI_MODEL", "grok-3-mini-fast-beta") # 从 .env 读取模型，默认为 fast-beta

print(f"Using xAI Model: {XAI_MODEL}") # 启动时打印使用的模型

# 常量 Prompt
TRANSLATION_PROMPT = """
你是一个.srt字幕文件翻译助手。你的任务是将以下**当前字幕**内容翻译成自然流畅的中文，表达优美且地道。
**请勿重复翻译上下文。你只需要翻译当前字幕内容。请勿翻译下文！**

这是一个关于{video_title}的视频，请确保准确翻译所有技术术语和专有名词。

特别需要注意的术语列表：
{special_terms}

请注意，你**不能**使用 Markdown 语法。

字幕内容如下:
当前字幕：{current_subtitle}
前{context_translated_count}句已翻译字幕：{context_translated_str}
前{context_untranslated_count}句未翻译字幕：{context_untranslated_str}


请按照以下步骤操作：
1. **直译**：逐句翻译**当前字幕**内容，确保基本语义准确。
2. **术语检查**：确保所有技术术语和专有名词保持原样，特别是上面列出的特殊术语。
3. **反思错误**：
   - 检查是否存在语言不通顺的地方。
   - 分析语法、语义和文化差异导致的翻译问题。
4. **意译优化**：根据错误反思调整译文，使其更加符合中文表达习惯。
5. **生成结果**：将完成的翻译内容输出到 `<r>` 标签中。

翻译时避免机械化，确保语言流畅自然。使用优雅的表达，不加入过多语气词。
永远将翻译内容以 `<r>` 标签包裹并放在输出的最后。当输出内容被截断时，直接续写，无需重复之前的内容。

开始翻译：
"""

# 术语提取Prompt
TERMS_EXTRACTION_PROMPT = """
你是一个专业的内容分析和术语提取专家。我需要你从以下视频字幕内容中提取重要的专业术语、缩写、专有名词和技术概念。

这个视频的标题是：{title}

这些术语将用于确保字幕翻译的准确性，因此需要特别注意以下类型的术语：
1. 计算机语言名称（如Python、JavaScript、C++等）
2. 框架和库名称（如React、TensorFlow、LLVM等）
3. 缩写词和首字母缩略词（如API、CPU、GPU、LLM等）
4. 专有名词和品牌名称
5. 技术概念和行业术语
6. 寻找主题关联的术语、比如LLVM关联IR、Block、Function等
请分析以下字幕内容（包括标题），并以列表形式返回重要术语：

{srt_content}

请使用以下格式返回结果：
<terms>
- 术语1
- 术语2
- 术语3
</terms>

仅返回术语列表，不需要解释每个术语的含义。确保不要遗漏任何可能在翻译过程中需要特别注意的术语。
"""


def parse_srt(srt_content: str) -> List[dict]:
    """Parses an SRT file content and returns a list of subtitle dictionaries."""
    lines = srt_content.strip().split("\n")
    subtitles = []
    i = 0
    while i < len(lines):
        try:
            # Ensure index is an integer and strip whitespace
            line_index = lines[i].strip()
            if not line_index.isdigit():
                i += 1
                continue
            index = int(line_index)
            i += 1

            # Check for valid time format (simple check)
            if '-->' not in lines[i]:
                i += 1
                continue
            time_str = lines[i].strip()
            i += 1

            # Collect text lines
            text_lines = []
            while i < len(lines) and lines[i].strip() != "":
                text_lines.append(lines[i])
                i += 1
            text = "\n".join(text_lines).strip()

            if not text: # Skip entries with no text
                i += 1 # Also skip potential empty line after text
                continue

            subtitles.append({
                "index": index,
                "time": time_str,
                "text": text
            })
            # Skip the empty line AFTER a valid entry
            if i < len(lines) and lines[i].strip() == "":
                 i += 1

        except (ValueError, IndexError):
            # Skip malformed entries gracefully
            i += 1
            continue
    return subtitles


def get_context(subtitles: List[dict], current_srt_index: int, translated_subtitles: List[dict],
                context_size: int = 2) -> dict:
    """Gets the context subtitles before the current subtitle based on SRT index."""
    context_translated = []
    context_untranslated = []

    # Find the actual list index corresponding to the current SRT index
    current_list_idx = -1
    for idx, sub in enumerate(subtitles):
        if sub['index'] == current_srt_index:
            current_list_idx = idx
            break

    if current_list_idx == -1:
        print(f"Warning: Could not find list index for SRT index {current_srt_index}")
        return {"translated": [], "untranslated": []}

    # Get translated context (looking backwards from the *last* translated item)
    start_translated_idx = max(0, len(translated_subtitles) - context_size)
    context_translated = [sub['translated'] for sub in translated_subtitles[start_translated_idx:]]

    # Get untranslated context (looking backwards from the current *list* index)
    start_untranslated_idx = max(0, current_list_idx - context_size)
    for idx in range(start_untranslated_idx, current_list_idx):
         # Check if this original subtitle index was already translated
         # This requires checking if an entry with subtitles[idx]['index'] exists in translated_subtitles
         original_sub_index = subtitles[idx]['index']
         is_already_translated = any(ts['index'] == original_sub_index for ts in translated_subtitles)
         if not is_already_translated:
             context_untranslated.append(subtitles[idx]['text'])

    return {
        "translated": context_translated,
        "untranslated": context_untranslated
    }


async def extract_terms(srt_content: str, title: str) -> list:
    """Uses AI (non-streaming) to extract technical terms from title and content."""
    parsed_subs = parse_srt(srt_content)
    if not parsed_subs:
        return []
    # Limit sample size to avoid overly long prompts
    sample_content = '\n'.join([sub['text'] for sub in parsed_subs[:50]])

    prompt = TERMS_EXTRACTION_PROMPT.format(
        title=title,
        srt_content=sample_content
    )

    try:
        terms_response = await translate_with_xai_async(prompt) # Use non-streaming call

        if terms_response.startswith("翻译失败"):
             print(f"Term extraction failed (API Error): {terms_response}")
             return [] # Return empty list on API failure

        # Attempt to extract terms using regex first
        match = re.search(r'<terms>(.*?)</terms>', terms_response, re.DOTALL | re.IGNORECASE)
        if match:
            terms_text = match.group(1).strip()
            terms_list = [term.strip().strip('- ') for term in terms_text.split('\n') if term.strip()]
            print(f"AI Extracted Terms (Regex): {terms_list}")
            return terms_list

        # Fallback: Try extracting list items if regex fails
        terms_list = []
        lines = terms_response.split('\n')
        in_terms_section = False # Handle cases where <terms> tag might be missing but list exists
        for line in lines:
             stripped_line = line.strip()
             if stripped_line.lower() == "<terms>":
                  in_terms_section = True
                  continue
             if stripped_line.lower() == "</terms>":
                  in_terms_section = False
                  continue
             # Check for list format or if inside tags
             if stripped_line.startswith('- ') or (in_terms_section and stripped_line):
                  term = stripped_line.strip().strip('- ')
                  if term:
                       terms_list.append(term)

        if terms_list:
            print(f"AI Extracted Terms (Fallback List): {terms_list}")
            return terms_list

        print(f"Could not extract terms from response: {terms_response[:200]}...")
        return []
    except Exception as e:
        print(f"Error processing term extraction response: {e}")
        return []


async def translate_with_xai_stream(prompt: str, max_retries: int = 3, initial_delay: float = 1.0) -> AsyncGenerator[str, None]:
    """
    Uses xAI API for translation with retries, attempting streaming but handling non-streaming JSON responses.
    Yields content chunks or a single complete chunk. Yields errors prefixed with '__ERROR__:'.
    Retries on TimeoutError or ClientError with exponential backoff.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {XAI_API_KEY}"
    }
    data = {
        "messages": [
            {"role": "system", "content": "You are a professional subtitle translator specialized in technical content."},
            {"role": "user", "content": prompt}
        ],
        "model": XAI_MODEL,
        "stream": True, # Attempt streaming
        "temperature": 0.2
    }

    retries = 0
    delay = initial_delay

    while retries <= max_retries:
        try:
            timeout = aiohttp.ClientTimeout(total=180, connect=30) # 保持超时设置
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(XAI_API_URL, headers=headers, json=data) as response:
                    if response.status != 200:
                        response_text = await response.text()
                        # 对 4xx 错误不重试，直接返回错误
                        if 400 <= response.status < 500:
                             yield f"__ERROR__:翻译失败: API客户端错误 {response.status} - {response_text[:500]}"
                             return
                        # 对于 5xx 或其他错误，允许重试
                        else:
                            raise aiohttp.ClientResponseError(
                                response.request_info,
                                response.history,
                                status=response.status,
                                message=response_text[:500],
                                headers=response.headers,
                            )

                    content_type = response.headers.get('Content-Type', '').lower()

                    # Handle Streaming Response (text/event-stream)
                    if 'text/event-stream' in content_type:
                        print(f"尝试次数 {retries + 1}: 处理为流...")
                        async for line in response.content:
                            if line:
                                line_text = line.decode('utf-8').strip()
                                if line_text == 'data: [DONE]':
                                    continue
                                if line_text.startswith('data: '):
                                    try:
                                        json_data = json.loads(line_text[6:])
                                        if 'error' in json_data:
                                            error_detail = json_data['error'].get('message', '未知API流错误')
                                            yield f"__ERROR__:翻译失败: API返回流错误 - {error_detail}"
                                            return # 明确的错误，不重试
                                        content = json_data.get('choices', [{}])[0].get('delta', {}).get('content', '')
                                        if content:
                                            yield content
                                    except json.JSONDecodeError:
                                        print(f"警告: 无法解析流式JSON: {line_text}")
                                        continue # 忽略格式错误的行
                        # 流式传输成功完成
                        return

                    # Handle Non-Streaming JSON Response (application/json)
                    elif 'application/json' in content_type:
                        print(f"尝试次数 {retries + 1}: 处理为JSON...")
                        try:
                            result = await response.json()
                            if 'error' in result:
                                error_detail = result['error'].get('message','未知API错误')
                                yield f"__ERROR__:翻译失败: API返回错误 - {error_detail}"
                                return # 明确的错误，不重试
                            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                            if content:
                                yield content # Yield the single complete result
                                return # 非流式成功完成
                            else:
                                nested_error = result.get("choices", [{}])[0].get("error")
                                if nested_error:
                                     error_detail = nested_error.get('message','未知嵌套API错误')
                                     yield f"__ERROR__:翻译失败: API choices错误 - {error_detail}"
                                else:
                                     yield "__ERROR__:翻译失败: API响应JSON中无有效内容或错误信息"
                                return # 明确的错误或无内容，不重试
                        except json.JSONDecodeError:
                             err_text = await response.text()
                             yield f"__ERROR__:翻译失败: 无法解析API JSON响应 - {err_text[:200]}..."
                             return # 解析错误，不重试
                        except Exception as e_json:
                             yield f"__ERROR__:翻译失败: 处理非流式JSON响应时出错 - {str(e_json)}"
                             return # 其他JSON处理错误，不重试

                    # Handle unexpected content type
                    else:
                        response_text = await response.text()
                        yield f"__ERROR__:翻译失败: 未知的API响应类型 '{content_type}' - {response_text[:200]}..."
                        return # 未知类型，不重试

            # 如果代码执行到这里，意味着请求成功并且所有内容都已yield/return
            # 对于流式响应，上面的 'return' 会退出循环
            # 对于非流式响应，上面的 'return' 也会退出循环
            break # 成功完成，退出重试循环

        except (asyncio.TimeoutError, aiohttp.ClientError) as e:
            retries += 1
            if retries > max_retries:
                if isinstance(e, asyncio.TimeoutError):
                    yield "__ERROR__:翻译失败: API请求超时 (已达最大重试次数)"
                else:
                    yield f"__ERROR__:翻译失败: 网络连接错误 (已达最大重试次数) - {str(e)}"
                return # 达到最大重试次数，退出

            print(f"请求失败 (尝试 {retries}/{max_retries+1}): {type(e).__name__}. 等待 {delay:.2f} 秒后重试...")
            await asyncio.sleep(delay)
            delay *= 2 # 指数退避
            continue # 继续下一次重试

        except Exception as e_outer:
            # 捕获其他意外错误
            yield f"__ERROR__:翻译失败: 调用翻译函数时发生未知错误 - {str(e_outer)}"
            return # 发生未知错误，退出


async def translate_with_xai_async(prompt: str, max_retries: int = 3, initial_delay: float = 1.0) -> str:
    """
    Uses xAI API (non-streaming) with retries - Primarily for term extraction.
    Retries on TimeoutError or ClientError with exponential backoff.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {XAI_API_KEY}"
    }
    data = {
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "model": XAI_MODEL, # Use the globally defined model
        "stream": False,
        "temperature": 0.1
    }

    retries = 0
    delay = initial_delay
    while retries <= max_retries:
        try:
            timeout = aiohttp.ClientTimeout(total=60, connect=20) # 保持超时设置
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(XAI_API_URL, headers=headers, json=data) as response:
                    if response.status != 200:
                        response_text = await response.text()
                        # 对 4xx 错误不重试，直接返回错误
                        if 400 <= response.status < 500:
                            return f"翻译失败: API客户端错误 {response.status} - {response_text[:500]}"
                        # 对于 5xx 或其他错误，允许重试
                        else:
                             raise aiohttp.ClientResponseError(
                                response.request_info,
                                response.history,
                                status=response.status,
                                message=response_text[:500],
                                headers=response.headers,
                            )
                    try:
                        result = await response.json()
                    except json.JSONDecodeError:
                        try:
                            err_text = await response.text()
                            return f"翻译失败: 无法解析API响应 - {err_text[:200]}..."
                        except:
                            return "翻译失败: 无法解析API响应且无法读取原始文本"

                    # Check for error in response
                    if 'error' in result:
                        return f"翻译失败: API返回错误 - {result['error'].get('message','未知API错误')}"
                    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                    if not content:
                         return "翻译失败：API未返回有效内容"
                    return content # 成功获取内容，返回并退出

            # 如果代码执行到这里，说明请求处理成功并已返回
            break # 确保退出循环（虽然理论上应该已经被上面的 return 退出了）

        except (asyncio.TimeoutError, aiohttp.ClientError) as e:
            retries += 1
            if retries > max_retries:
                if isinstance(e, asyncio.TimeoutError):
                     return "翻译失败: API请求超时 (已达最大重试次数)"
                else:
                     return f"翻译失败: 网络连接错误 (已达最大重试次数) - {str(e)}"

            print(f"请求失败 (尝试 {retries}/{max_retries+1}): {type(e).__name__}. 等待 {delay:.2f} 秒后重试...")
            await asyncio.sleep(delay)
            delay *= 2 # 指数退避
            continue # 继续下一次重试

        except Exception as e_outer:
            # 捕获其他意外错误
            return f"翻译失败: 未知错误 - {str(e_outer)}"


async def translate_srt(srt_content: str, filename: str, display_mode: str = "only_translated", user_terms_list: List[str] = []) -> str:
    """Translates an SRT file content using xAI API (non-streaming). Optionally uses custom terms."""
    subtitles = parse_srt(srt_content)
    if not subtitles:
        return "错误：无法解析SRT文件或文件为空。"
    translated_subtitles = []
    video_title = pathlib.Path(filename).stem

    # Term extraction and merging (similar to stream version)
    ai_extracted_terms = await extract_terms(srt_content, video_title)
    cleaned_user_terms = set(term.strip() for term in user_terms_list if term.strip())
    cleaned_ai_terms = set(term.strip() for term in ai_extracted_terms if term.strip())
    combined_terms_set = cleaned_ai_terms | cleaned_user_terms
    combined_terms_list = sorted(list(combined_terms_set), key=str.lower)
    special_terms_str = "\n".join([f"- {term}" for term in combined_terms_list]) if combined_terms_list else "无"

    for i, subtitle in enumerate(subtitles):
        # Use the correct SRT index for context gathering
        context = get_context(subtitles, subtitle['index'], translated_subtitles)
        context_translated_str = "\n".join(context["translated"])
        context_untranslated_str = "\n".join(context["untranslated"])

        prompt = TRANSLATION_PROMPT.format(
            video_title=video_title,
            special_terms=special_terms_str, # Use combined terms
            current_subtitle=subtitle['text'],
            context_translated_str=context_translated_str,
            context_untranslated_str=context_untranslated_str,
            context_translated_count=len(context["translated"]),
            context_untranslated_count=len(context["untranslated"]),
        )

        # Use non-streaming async function
        response_text = await translate_with_xai_async(prompt)

        translated_text = ""
        if response_text.startswith("翻译失败"):
            translated_text = response_text # Keep error message
        else:
            try:
                # Attempt to extract <r> tag first
                match = re.search(r'<r>(.*?)</r>', response_text, re.DOTALL | re.IGNORECASE)
                if match:
                    translated_text = match.group(1).strip()
                    translated_text = re.sub(r'\n{2,}', '\n', translated_text).strip()
                else:
                    # If no <r> tag, use the whole response
                    translated_text = response_text.strip()
                    if not translated_text:
                        translated_text = "翻译成功但无有效内容"
            except Exception as e:
                translated_text = f"翻译成功但解析结果失败: {str(e)}"

        translated_subtitles.append({
            "index": subtitle['index'],
            "time": subtitle['time'],
            "original": subtitle['text'],
            "translated": translated_text
        })

    # Build final SRT output
    output_srt = ""
    for sub in translated_subtitles:
        output_srt += f"{sub['index']}\n"
        output_srt += f"{sub['time']}\n"
        if display_mode == "only_translated":
            output_srt += f"{sub['translated']}\n\n"
        elif display_mode == "original_above_translated":
            output_srt += f"{sub['original']}\n"
            output_srt += f"{sub['translated']}\n\n"
        elif display_mode == "translated_above_original":
            output_srt += f"{sub['translated']}\n"
            output_srt += f"{sub['original']}\n\n"
    return output_srt


async def translate_srt_stream(srt_content: str, filename: str, display_mode: str = "only_translated", user_terms_list: List[str] = []):
    """Translates an SRT file content using xAI API with context and streaming (handling non-stream responses)."""
    subtitles = parse_srt(srt_content)
    if not subtitles:
         yield json.dumps({"type": "error", "message": "无法解析SRT文件或文件为空。"}) + "\n"
         return

    video_title = pathlib.Path(filename).stem

    # --- Term Extraction and Merging Logic ---
    yield json.dumps({"type": "info", "message": "正在提取和合并术语..."}) + "\n"
    ai_extracted_terms = []
    combined_terms_list = []
    special_terms_str = "无"
    try:
        ai_extracted_terms = await extract_terms(srt_content, video_title)
        cleaned_user_terms = set(term.strip() for term in user_terms_list if term.strip())
        cleaned_ai_terms = set(term.strip() for term in ai_extracted_terms if term.strip())
        combined_terms_set = cleaned_ai_terms | cleaned_user_terms
        combined_terms_list = sorted(list(combined_terms_set), key=str.lower)
        special_terms_str = "\n".join([f"- {term}" for term in combined_terms_list]) if combined_terms_list else "无"
        yield json.dumps({"type": "terms", "terms": combined_terms_list}) + "\n"
    except Exception as e:
         yield json.dumps({"type": "error", "message": f"术语处理时出错: {e}，将仅使用用户自定义术语（如有）。"}) + "\n"
         cleaned_user_terms = set(term.strip() for term in user_terms_list if term.strip())
         combined_terms_list = sorted(list(cleaned_user_terms), key=str.lower)
         special_terms_str = "\n".join([f"- {term}" for term in combined_terms_list]) if combined_terms_list else "无"
         yield json.dumps({"type": "terms", "terms": combined_terms_list}) + "\n"
    # --- End Term Logic ---

    yield json.dumps({"type": "info", "message": f"开始翻译，共 {len(subtitles)} 个字幕", "total": len(subtitles)}) + "\n"

    translated_subtitles = []
    total_subtitles = len(subtitles)

    for i, subtitle in enumerate(subtitles):
        current_progress = i + 1
        percent_complete = round(current_progress / total_subtitles * 100, 1)

        yield json.dumps({
            "type": "progress", "current": current_progress, "total": total_subtitles, "percent": percent_complete
        }) + "\n"

        # Use the correct SRT index for context gathering
        context = get_context(subtitles, subtitle['index'], translated_subtitles)
        context_translated_str = "\n".join(context["translated"])
        context_untranslated_str = "\n".join(context["untranslated"])

        prompt = TRANSLATION_PROMPT.format(
            video_title=video_title, special_terms=special_terms_str, current_subtitle=subtitle['text'],
            context_translated_str=context_translated_str, context_untranslated_str=context_untranslated_str,
            context_translated_count=len(context["translated"]), context_untranslated_count=len(context["untranslated"]),
        )

        yield json.dumps({
            "type": "current", "index": subtitle['index'], "time": subtitle['time'], "text": subtitle['text']
        }) + "\n"

        translated_text = "翻译失败：未知错误"
        api_error = None
        accumulated_response = ""
        has_translation_started = False

        try:
            # Call the modified stream function which handles both stream/non-stream
            async for chunk in translate_with_xai_stream(prompt):
                if chunk.startswith("__ERROR__"):
                    api_error = chunk.replace("__ERROR__:", "").strip()
                    translated_text = api_error # Use error as the "translation"
                    break # Stop processing chunks for this subtitle

                # If no error, accumulate the response chunk
                accumulated_response += chunk
                has_translation_started = True

                # Send cumulative chunk to frontend for potential real-time display
                yield json.dumps({
                    "type": "translation_chunk",
                    "text": accumulated_response
                }) + "\n"

            # After the loop (or break on error), finalize the translation text
            if not api_error:
                if has_translation_started and accumulated_response:
                    try:
                        # Attempt to extract <r> tag content
                        match = re.search(r'<r>(.*?)</r>', accumulated_response, re.DOTALL | re.IGNORECASE)
                        if match:
                            translated_text = match.group(1).strip()
                            # Clean extra newlines
                            translated_text = re.sub(r'\n{2,}', '\n', translated_text).strip()
                        else:
                             # Use the whole response if no <r> tag found
                             translated_text = accumulated_response.strip()
                             if not translated_text:
                                  translated_text = "翻译成功但无有效内容"
                    except Exception as parse_err:
                         translated_text = f"翻译成功但解析最终结果失败: {parse_err}"
                elif not has_translation_started: # Stream ended without error but no content received
                     translated_text = "翻译失败：API未返回任何内容"
                # else: accumulated_response is empty, keep default error

        except Exception as e:
            # Catch errors during the async for loop itself
            api_error = f"处理字幕 {subtitle['index']} 流时发生内部错误: {str(e)}"
            translated_text = api_error
            print(f"Error processing stream for subtitle {subtitle['index']}: {e}")

        # Send error event if an API error occurred
        if api_error:
            yield json.dumps({
                "type": "error", "index": subtitle['index'], "message": api_error
            }) + "\n"

        # Send final translation result (could be the translation or an error message)
        yield json.dumps({
            "type": "translation_complete", "index": subtitle['index'], "time": subtitle['time'],
            "original": subtitle['text'], "translated": translated_text
        }) + "\n"

        # Append to list for context generation, using the final text (translation or error)
        translated_subtitles.append({
            "index": subtitle['index'], "time": subtitle['time'],
            "original": subtitle['text'], "translated": translated_text
        })

        # Build and send the SRT chunk based on display mode
        output_srt_chunk = f"{subtitle['index']}\n{subtitle['time']}\n"
        if display_mode == "only_translated":
            output_srt_chunk += f"{translated_text}\n\n"
        elif display_mode == "original_above_translated":
             output_srt_chunk += f"{subtitle['text']}\n{translated_text}\n\n"
        elif display_mode == "translated_above_original":
            output_srt_chunk += f"{translated_text}\n{subtitle['text']}\n\n"
        else: # Default to only translated
            output_srt_chunk += f"{translated_text}\n\n"

        yield json.dumps({"type": "srt", "text": output_srt_chunk}) + "\n"

    # Signal completion after processing all subtitles
    yield json.dumps({"type": "complete", "message": "翻译完成"}) + "\n"


# WebUI 路由
@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# /upload route handles form submission from index.html
@app.post("/upload", response_class=HTMLResponse)
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    display_mode: str = Form("only_translated"),
    stream: Optional[bool] = Form(False),
    custom_terms: Optional[str] = Form(None) # Receive custom terms from form
):
    try:
        contents = await file.read()
        srt_content = None
        detected_encoding = "utf-8" # Default

        # Attempt decoding with common encodings
        encodings_to_try = ['utf-8', 'utf-8-sig', 'latin-1', 'gbk', 'big5', 'shift_jis']
        for encoding in encodings_to_try:
            try:
                srt_content = contents.decode(encoding)
                detected_encoding = encoding
                print(f"Successfully decoded with {encoding}")
                break
            except UnicodeDecodeError:
                continue

        if srt_content is None:
             raise ValueError("无法解码SRT文件内容，请确保文件编码正确。")

        if stream:
            # Render stream.html, passing necessary data including custom terms
            return templates.TemplateResponse(
                "stream.html",
                {
                    "request": request,
                    "filename": file.filename,
                    "display_mode": display_mode,
                    "srt_content": srt_content,
                    "custom_terms": custom_terms or "" # Pass custom terms string
                }
            )
        else:
            # Non-streaming translation
            # Parse custom terms for non-streaming translation as well
            user_terms_list = [term.strip() for term in custom_terms.splitlines() if term.strip()] if custom_terms else []
            translated_srt = await translate_srt(
                srt_content,
                file.filename,
                display_mode,
                user_terms_list=user_terms_list # Pass terms to non-streaming function
            )
            filename_dl = f"translated_{file.filename}"
            return templates.TemplateResponse(
                "result.html",
                {
                    "request": request,
                    "translated_text": translated_srt,
                    "original_filename": file.filename,
                    "translated_filename": filename_dl
                }
            )
    except Exception as e:
        print(f"Upload error: {e}")
        # Optionally log the full traceback here
        # import traceback
        # traceback.print_exc()
        return templates.TemplateResponse(
            "error.html",
            {"request": request, "error": str(e)}
        )


# API Endpoints (less critical for UI functionality but good practice)

@app.post("/api/translate", response_class=JSONResponse)
async def translate_endpoint(
    file: UploadFile = File(...),
    display_mode: str = Form("only_translated"),
    custom_terms: Optional[str] = Form(None) # Add custom terms here too
):
    """API endpoint for translating SRT file (non-streaming)."""
    try:
        contents = await file.read()
        srt_content = None
        # ... (decoding logic as in /upload) ...
        encodings_to_try = ['utf-8', 'utf-8-sig', 'latin-1', 'gbk', 'big5', 'shift_jis']
        for encoding in encodings_to_try:
            try: srt_content = contents.decode(encoding); break
            except UnicodeDecodeError: continue
        if srt_content is None: raise HTTPException(status_code=400, detail="无法解码SRT文件内容。")

        user_terms_list = [term.strip() for term in custom_terms.splitlines() if term.strip()] if custom_terms else []
        # Call translate_srt with custom terms
        translated_srt = await translate_srt(srt_content, file.filename, display_mode, user_terms_list=user_terms_list)
        return JSONResponse(content={"translated_srt": translated_srt})
    except Exception as e:
        # Log the error server-side for API calls as well
        print(f"API /api/translate error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/api/translate-stream")
async def translate_stream_endpoint(
    file: UploadFile = File(...),
    display_mode: str = Form("only_translated"),
    custom_terms: Optional[str] = Form(None) # Receive custom terms
):
    """API endpoint for streaming translation of SRT file."""
    try:
        contents = await file.read()
        srt_content = None
        # ... (decoding logic as in /upload) ...
        encodings_to_try = ['utf-8', 'utf-8-sig', 'latin-1', 'gbk', 'big5', 'shift_jis']
        for encoding in encodings_to_try:
            try: srt_content = contents.decode(encoding); break
            except UnicodeDecodeError: continue
        if srt_content is None:
             async def error_stream(): yield json.dumps({"type": "error", "message": "无法解码SRT文件内容。"}) + "\n"
             return StreamingResponse(error_stream(), media_type="text/event-stream")

        # Parse custom terms string into a list, clean them
        user_terms_list = [term.strip() for term in custom_terms.splitlines() if term.strip()] if custom_terms else []

        # Pass user_terms_list to the generator
        return StreamingResponse(
            translate_srt_stream(srt_content, file.filename, display_mode, user_terms_list=user_terms_list),
            media_type="text/event-stream"
        )
    except Exception as e:
        # Log the error server-side for API calls as well
        print(f"API /api/translate-stream error: {e}")
        async def error_stream(): yield json.dumps({"type": "error", "message": f"处理流请求时出错: {str(e)}"}) + "\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")


# Download Endpoint
@app.post("/download")
async def download_translation(translated_text: str = Form(...), filename: str = Form(...)):
    # Sanitize filename more thoroughly
    # Remove directory separators and limit length
    safe_filename = re.sub(r'[/\\]', '', filename) # Remove slashes
    safe_filename = re.sub(r'[^\w\.\-]', '_', safe_filename) # Replace other invalid chars
    safe_filename = safe_filename[:100] # Limit length
    if not safe_filename.lower().endswith(".srt"):
         safe_filename += ".srt" # Ensure .srt extension
    if not safe_filename or safe_filename == ".srt":
        safe_filename = "translated_download.srt"

    # Encode the text to UTF-8 bytes for the response body
    response_body = translated_text.encode('utf-8')

    return StreamingResponse(
        iter([response_body]),
        media_type="text/plain; charset=utf-8",
        headers={
            # Use RFC 5987 encoding for filenames with potentially non-ASCII chars
            "Content-Disposition": f"attachment; filename*=UTF-8''{safe_filename}"
            }
    )


if __name__ == "__main__":
    import uvicorn
    # Ensure necessary directories exist (optional, good practice)
    # os.makedirs("templates", exist_ok=True)
    # os.makedirs("static/css", exist_ok=True)
    # os.makedirs("static/js", exist_ok=True)
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)