"""
CompetitorSmart — FastAPI 后端
功能：LangGraph ReAct Agent + DuckDuckGo联网搜索 + 网页抓取
部署：Railway
"""

import os
import re
import textwrap
import asyncio
from typing import Annotated, Literal, AsyncGenerator

import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="CompetitorSmart API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Skills Prompt（来自Lenny PM技能库6个核心技能）────────────────────────────
SKILLS_PROMPT = """
## 分析框架参考（Lenny PM技能库，6个核心技能）

### Skill #09 竞品分析核心原则
- 从「竞争替代」开始：如果你的产品不存在，客户会做什么？
- 竞品不只是直接竞品，还包括客户用来解决问题的所有变通方案
- 从外部视角出发：始终从客户、市场、竞争对手视角，而非内部视角
- 理解行业经济学：真正的竞争优势需要理解底层行业经济关系
- 包含「模拟」替代：分析传统、非数字替代方案，不只看直接数字竞争对手

### Skill #60 产品定位框架（April Dunford方法论）
- 定位 = 为什么我们是特定客户的最佳选择
- 5个定位要素：竞争替代、独特属性、价值、目标客户、市场类别
- 好的定位让销售变容易，差的定位让销售变成推销
- 定位需要定期更新，市场变化时定位也需要随之调整

### Skill #05 北极星指标分析
- 北极星指标 = 最能预测产品长期成功的单一关键指标
- 好的北极星指标：反映用户价值、可测量、可行动、预测收入
- 分析竞品时关注其北极星指标有助于理解其战略重心

### Skill #07 路线图优先级判断
- 用RICE框架（覆盖度/影响力/信心/投入）评估优先级
- 关注竞品的功能迭代节奏，判断其战略重心转移

### Skill #04 产品愿景评估
- 好的产品愿景：鼓舞人心、有野心但可实现、以用户为中心
- 分析竞品愿景有助于预判其未来战略方向

### Claude竞品分析结构框架
分析维度：市场定位 → 核心能力 → 商业模式 → 增长策略 → 用户画像 → SWOT → 壁垒 → 机会
每个竞品必须回答：它解决了什么问题？谁在付钱？为什么用户不离开？
""".strip()

# ── System Prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = f"""你是一名专业的竞争情报（Competitive Intelligence）分析师。

{SKILLS_PROMPT}

---

工作流程：
阶段一：搜集竞品信息（每个竞品搜索2次 + 抓取1次官网）
阶段二：直接输出完整报告，不要输出任何过渡句

来源标注规则：
- 禁止将第三方媒体文章标注为"官网"
- 官网链接仅限该产品官方域名
- 关键结论后附来源，格式：（来源：[标题](URL)）

报告格式要求：
- 以 "# 竞品调研报告：{{市场}}" 开头
- 必须包含以下11个章节：

  ## 一、报告概述（Executive Summary）
  ## 二、市场与赛道分析（Market Context）
  （市场规模、增速、竞争格局、趋势判断，≥3条要点）
  ## 三、竞品选择与分层（Competitive Landscape）
  ## 四、核心能力拆解（Product Capability Analysis）
  （每个竞品用 ### 单独列小节，必须包含9个字段：
  产品定位、核心功能、技术特点、定价策略、用户规模、更新节奏、分发渠道、商业模式、近期动态）
  ## 五、商业模式分析（Monetization）
  （收费方式、客单价、付费转化路径，≥3条要点）
  ## 六、增长与分发策略（Growth Strategy）
  ## 七、用户与场景分析（User & Use Case）
  ## 八、优劣势对比（SWOT / 对比矩阵）
  （每个竞品：优势≥3、劣势≥3、机会≥2、威胁≥2；附整体对比矩阵表格）
  ## 九、关键差异与壁垒（Moat Analysis）
  ## 十、机会点与策略建议（Opportunities）
  （≥3条具体可执行建议，每条说明依据）
  ## 十一、数据附录（Appendix）

- 使用表格对比关键数据
- 总字数不少于4000字
- 每个判断有数据或事实支撑"""

# ── 工具 ──────────────────────────────────────────────────────────────────────
@tool
def search_web(query: str) -> str:
    """搜索网络，获取竞争对手的公开信息。输入搜索词，返回多条搜索结果摘要。"""
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        results = list(DDGS().text(query, max_results=5))
        if not results:
            return "未找到搜索结果，请换一个更具体的搜索词。"
        lines = []
        for r in results:
            title = r.get("title", "").strip()
            body = r.get("body", "").strip()
            href = r.get("href", "").strip()
            lines.append(f"### {title}\n{body}\n来源: {href}")
        return "\n\n---\n\n".join(lines)
    except Exception as e:
        return f"搜索失败: {str(e)}"


@tool
def fetch_webpage(url: str) -> str:
    """抓取指定网页并提取纯文本内容（最多返回3000字符）。"""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        html = resp.text
        html = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r"<style[^>]*>.*?</style>", " ", html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r"<[^>]+>", " ", html)
        html = re.sub(r"&[a-z]+;", " ", html)
        html = re.sub(r"\s{2,}", "\n", html).strip()
        return textwrap.shorten(html, width=3000, placeholder="…（内容已截断）")
    except requests.exceptions.Timeout:
        return f"请求超时: {url}"
    except Exception as e:
        return f"无法获取页面: {str(e)}"


# ── Agent ─────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def build_agent(api_key: str, base_url: str, model: str):
    tools = [search_web, fetch_webpage]
    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0.3,
        max_tokens=4096,
    ).bind_tools(tools)
    tool_node = ToolNode(tools)
    MAX_TOOL_CALLS = 25

    def agent_node(state: AgentState) -> dict:
        messages = list(state["messages"])
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        tool_call_count = sum(
            1 for m in messages
            if isinstance(m, AIMessage) and hasattr(m, "tool_calls") and m.tool_calls
        )
        if tool_call_count >= MAX_TOOL_CALLS:
            messages.append(SystemMessage(
                content="信息已足够，请立即输出完整竞争情报报告，不要再调用任何工具。"
            ))
        return {"messages": [llm.invoke(messages)]}

    def route(state: AgentState) -> Literal["tools", "agent", "__end__"]:
        last: AIMessage = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            tool_call_count = sum(
                1 for m in state["messages"]
                if isinstance(m, AIMessage) and hasattr(m, "tool_calls") and m.tool_calls
            )
            if tool_call_count >= MAX_TOOL_CALLS:
                return END
            return "tools"
        content = getattr(last, "content", "") or ""
        if "# 竞品调研报告" in content or "# 竞争情报报告" in content:
            return END
        return "agent"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent", route,
        {"tools": "tools", "agent": "agent", "__end__": END}
    )
    graph.add_edge("tools", "agent")
    return graph.compile()


# ── API 请求模型 ──────────────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    competitors: str
    market: str
    api_key: str
    base_url: str
    model: str = "gpt-4o"


# ── SSE 流式生成器 ────────────────────────────────────────────────────────────
async def run_agent_stream(req: AnalyzeRequest) -> AsyncGenerator[str, None]:
    """运行Agent，通过SSE流式返回进度和最终报告"""

    def sse(data: str) -> str:
        # SSE格式
        return f"data: {data}\n\n"

    compiled = build_agent(req.api_key, req.base_url, req.model)

    user_task = (
        f"请对以下竞争对手进行全面研究，生成竞争情报报告：\n\n"
        f"**竞争对手：** {req.competitors}\n"
        f"**市场/产品类别：** {req.market}\n"
        f"**地理范围：** 中国市场为主\n\n"
        f"请先用工具搜集每个竞品的信息，搜集完毕后输出完整报告。"
    )

    tool_count = 0
    final_report = ""

    try:
        for chunk in compiled.stream(
            {"messages": [("user", user_task)]},
            stream_mode="updates",
        ):
            for node_name, node_output in chunk.items():
                msgs: list[BaseMessage] = node_output.get("messages", [])

                if node_name == "agent":
                    for msg in msgs:
                        if not isinstance(msg, AIMessage):
                            continue
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tc in msg.tool_calls:
                                tool_count += 1
                                args = tc.get("args", {})
                                arg_val = next(iter(args.values()), "") if args else ""
                                preview = str(arg_val)[:60]
                                icon = "🔍" if tc["name"] == "search_web" else "🌐"
                                log = f"[{tool_count:02d}] {icon} {tc['name']}: {preview}"
                                yield sse(f'{{"type":"log","text":{repr(log)}}}')
                                await asyncio.sleep(0)

                        elif msg.content and (
                            "# 竞品调研报告" in msg.content
                            or "# 竞争情报报告" in msg.content
                        ):
                            final_report = msg.content
                            yield sse(f'{{"type":"log","text":"✅ 报告生成完成！共调用工具 {tool_count} 次"}}')
                            await asyncio.sleep(0)

                elif node_name == "tools":
                    for msg in msgs:
                        content = getattr(msg, "content", "") or ""
                        preview = content[:50].replace("\n", " ")
                        yield sse(f'{{"type":"log","text":{repr("   ↳ " + preview + "...")}}}')
                        await asyncio.sleep(0)

    except Exception as e:
        yield sse(f'{{"type":"error","text":{repr(str(e))}}}')
        return

    if final_report:
        import json
        yield sse(f'{{"type":"report","content":{json.dumps(final_report)}}}')
    else:
        yield sse('{"type":"error","text":"Agent未能生成完整报告，请重试"}')

    yield sse('{"type":"done"}')


# ── 路由 ──────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "service": "CompetitorSmart API"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/api/analyze")
async def analyze(req: AnalyzeRequest):
    """启动Agent分析，SSE流式返回进度和报告"""
    return StreamingResponse(
        run_agent_stream(req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
