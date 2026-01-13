import os
import sys
import base64
import asyncio

import chainlit as cl

# ---------------------- Ensure local imports resolve ----------------------
CURRENT_DIR = os.path.dirname(__file__)
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from langgraph_pipeline import app, get_lazy_retriever

# ---------------------- Helpers: sources & formatting ----------------------
def _basename(p: str) -> str:
    try:
        return os.path.basename(p) or p
    except Exception:
        return str(p)

def _rag_sources(st: dict, k: int = 3) -> list[str]:
    seen, out = set(), []
    for s in (st or {}).get("sources") or []:
        src = s.get("source")
        if src:
            name = _basename(str(src))
            if name not in seen:
                seen.add(name)
                out.append(name)
        if len(out) >= k:
            break
    return out

def _viz_source(st: dict) -> str | None:
    files = (st or {}).get("chart_files") or []
    if files:
        return _basename(files[0])
    ctx = (st or {}).get("chart_context") or ""
    if "Best file selected:" in ctx:
        return _basename(ctx.split("Best file selected:", 1)[-1].strip())
    return None

def _sources_footer(final_state: dict) -> str:
    qtype = final_state.get("query_type")
    if qtype == "visualization":
        ds = _viz_source(final_state)
        return f"\n\n*Source:* `{ds}`" if ds else ""
    tops = _rag_sources(final_state, k=2)
    if not tops:
        return ""
    label = "Source" if len(tops) == 1 else "Sources"
    return f"\n\n*{label}:* " + ", ".join(f"`{s}`" for s in tops)

# ---------------------- Node name to user-friendly description ----------------------
NODE_DESCRIPTIONS = {
    "classifier": "🔍 Analyzing your request...",
    "chat": "💬 Generating conversational response...",
    "rag": "📚 Searching knowledge base...",
    "chart_rag": "📊 Finding relevant datasets...",
    "visualization": "🎨 Creating visualization...",
}

def get_step_details(node_name: str, state: dict) -> str:
    """Generate detailed step description based on node and state."""
    if node_name == "classifier":
        qt = state.get("query_type", "unknown")
        return f"Classified query as `{qt}`"
    
    elif node_name == "rag":
        tops = _rag_sources(state, k=3)
        if tops:
            return f"Retrieved documents: " + ", ".join(f"`{s}`" for s in tops)
        return "Searching documents..."
    
    elif node_name == "chart_rag":
        chart_files = state.get("chart_files", [])
        if chart_files:
            short = [_basename(p) for p in chart_files]
            return f"Selected dataset: `{short[0]}`"
        return "Finding suitable datasets..."
    
    elif node_name == "visualization":
        # By the time we see this node complete, the chart is ready
        # So we show what actually happened during execution
        return "Generating Python code and executing....."
    
    elif node_name == "chat":
        return "Crafting response..."
    
    return f"**Processing:** {node_name}"

# ---------------------- Chat lifecycle ----------------------
@cl.on_chat_start
async def start():
    msg = await cl.Message(content="Preparing environment…").send()
    try:
        await asyncio.to_thread(get_lazy_retriever)
    finally:
        try:
            await msg.remove()
        except Exception:
            pass

    await cl.Message(
        content="S.A.G.E is online.\n\nYour project knowledge, simplified. What can I help you with today?"
    ).send()

    cl.user_session.set(
        "state",
        {
            "messages": [],
            "last_user_text": "",
            "query_type": None,
            "chart_context": "",
            "chart_files": [],
            "chart_image": None,
            "chart_summary": "",
            "sources": [],
        },
    )

# ---------------------- Main handler with REAL-TIME STREAMING ----------------------
@cl.on_message
async def main(message: cl.Message):
    # ---- Load & update conversation state ----
    state = cl.user_session.get("state") or {}
    state.setdefault("messages", [])
    state["last_user_text"] = message.content
    state["messages"].append({
        "role": "user",
        "content": [{"type": "text", "text": message.content}],
    })

    # ---- Create streaming progress message ----
    progress_msg = cl.Message(content="")
    await progress_msg.send()
    
    steps_completed = []
    final_state = None

    async def stream_graph():
        """Stream through graph execution and yield node updates."""
        nonlocal final_state
        try:
            # Use app.stream() to get updates as each node completes
            for step_output in await asyncio.to_thread(lambda: list(app.stream(state))):
                # step_output is a dict like: {"classifier": <state_after_classifier>}
                node_name = list(step_output.keys())[0]
                node_state = step_output[node_name]
                
                yield node_name, node_state
            
            # Get final state after all nodes complete
            final_state = await asyncio.to_thread(app.invoke, state)
            
        except Exception as e:
            raise e

    try:
        # Initial message
        await progress_msg.stream_token("⏳ Starting...\n\n")
        
        # Stream through graph execution
        async for node_name, node_state in stream_graph():
            # Skip special nodes
            if node_name in ["__start__", "__end__"]:
                continue
            
            # Get friendly description
            description = NODE_DESCRIPTIONS.get(node_name, f"Processing {node_name}...")
            
            # Add to steps
            step_detail = get_step_details(node_name, node_state)
            steps_completed.append(step_detail)
            
            # Clear and rebuild progress display
            progress_text = f"{description}\n\n" + "\n".join(steps_completed) + "\n"
            
            # Update the message content by creating a new one
            try:
                await progress_msg.remove()
            except:
                pass
            
            progress_msg = cl.Message(content=progress_text)
            await progress_msg.send()
            
            # ✅ NEW: If chart is ready, display it immediately!
            if node_name == "visualization" and node_state.get("chart_image"):
                await asyncio.sleep(0.5)  # Brief pause to show completion
                try:
                    await progress_msg.remove()
                except:
                    pass
                
                # Display chart immediately
                chart_b64 = node_state.get("chart_image")
                chart_summary = node_state.get("chart_summary")
                
                # Extract fallback text
                assistant_msgs = [
                    m for m in node_state.get("messages", []) if m.get("role") == "assistant"
                ]
                fallback_text = None
                if assistant_msgs:
                    last = assistant_msgs[-1]
                    content = last.get("content", "")
                    if isinstance(content, list) and content and isinstance(content[0], dict):
                        fallback_text = content[0].get("text", "")
                    else:
                        fallback_text = content

                text = chart_summary or fallback_text or "Here is the visualization:"
                text = text + _sources_footer(node_state)

                elements = []
                try:
                    img_bytes = base64.b64decode(chart_b64)
                    elements.append(
                        cl.Image(
                            name="chart.png",
                            display="inline",
                            content=img_bytes,
                            size="large",
                        )
                    )
                except Exception as e:
                    text += f"\n\n*(Chart display error: {e})*"

                await cl.Message(content=text, elements=elements).send()
                return  # Exit early - chart is shown!
            
            # Small delay for visual effect
            await asyncio.sleep(0.3)
        
        # Remove progress message
        try:
            await progress_msg.remove()
        except Exception:
            pass

    except Exception as e:
        try:
            await progress_msg.remove()
        except Exception:
            pass
        await cl.Message(f"❌ Error: {e}").send()
        return

    # ---------------------- Present final answer (for non-chart responses) ----------------------
    if not final_state:
        await cl.Message("No response generated.").send()
        return

    # If we got here, it's a chat or RAG response (chart already handled above)
    assistant_msgs = [
        m for m in final_state.get("messages", []) if m.get("role") == "assistant"
    ]
    if assistant_msgs:
        last = assistant_msgs[-1]
        content = last.get("content", "")
        if isinstance(content, list) and content and isinstance(content[0], dict):
            text = content[0].get("text", "")
        else:
            text = content
        text = text or "Response generated but empty."
        await cl.Message(content=text + _sources_footer(final_state)).send()
    else:
        await cl.Message("No response produced.").send()

    # ---- Trim & persist history ----
    if len(final_state.get("messages", [])) > 40:
        final_state["messages"] = final_state["messages"][-40:]
    cl.user_session.set("state", final_state)
