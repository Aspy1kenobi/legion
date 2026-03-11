from datetime import datetime
import logging
logger = logging.getLogger(__name__)

def log_call(func):
    def wrapper(*args, **kwargs):
        logger.debug("[%s] called with: %s %s", func.__name__, args, kwargs)
        return func(*args, **kwargs)
    return wrapper

@log_call
def agent_planner(topic: str, context: str = "") -> str:
    output = (
        f"[Planner]\n"
        f"Goal: Make progress on: {topic}\n"
    )

    if context:
        output += f"\nConsidering your recent activity:{context}\n"

    output += (
        f"Plan:\n"
        f"1) Define what success looks like\n"
        f"2) List constraints and resources\n"
        f"3) Break into 3 small tasks\n"
        f"4) Choose the next action you can do in 15 minutes\n"
    )
    return output


@log_call
def agent_engineer(topic: str, context: str = "") -> str:
    output = f"[Engineer]\n"

    if context:
        output += f"Looking at what you've been working on:{context}\n"
        
    output += (
        f"Implementation ideas for: {topic}\n"
        f"- Start with a minimal prototype\n"
        f"- Add logging + tests early\n"
        f"- Keep modules small and readable\n"
    )
    return output


@log_call
def agent_skeptic(topic: str, context: str = "") -> str:
    output = f"[Skeptic]\n"

    if context:
        output += f"Based on your notes:{context}\n"

    output += (
        f"Questions / risks about: {topic}\n"
        f"- What assumptions might be wrong?\n"
        f"- What could fail silently?\n"
        f"- How do we know it works?\n"
    )
    return output


@log_call
def agent_ethicist(topic: str, context: str = "") -> str:
    output = f"[ethicist]\n"
    
    if context:
        output += f"What precedent does this set: {context}\n"

    output +=(
        f"Safety / ethics check for: {topic}\n"
        f"- Does this increase harm or risk?\n"
        f"- Are there privacy issues?\n"
        f"- Can we add a human approval step?\n"
    )
    return output

@log_call
def agent_imagination(topic: str, context: str = "") -> str:
    """Creative agent powered by your LLM"""
    from llm_bridge import generate_text, is_available, initialize_llm

    if not is_available():
        return "[Imagination agent]\nLLM not available"
    
    initialize_llm()

    #Create a prompt for the LLM based on the topic
    prompt = f"Creative ideas about {topic}: "

    # Generate
    text = generate_text(prompt, length=200, temperature=0.8)

    if text:
        return f"[Imagination Agent - Powered by TinyLM]\n{text}"
    else:
        return "[Imagination Agent]\nFailed to generate ideas"
    


def run_lab_meeting(topic: str, context: str = "") -> str:
    outputs = [
        agent_planner(topic, context),
        agent_engineer(topic, context),
        agent_skeptic(topic, context),
        agent_imagination(topic, context),
        agent_ethicist(topic, context),
    ]
    return "\n\n".join(outputs)