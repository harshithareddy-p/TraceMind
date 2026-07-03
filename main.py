import streamlit as st
from ollama import chat
import datetime

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="AI Assistant", page_icon="🤖", layout="centered")
st.title("🤖 AI Assistant")
st.caption("Powered by your local Ollama model")

# ---------------------------------------------------------------------------
# Sidebar: model + generation settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    model_name = st.selectbox(
        "Model",
        options=["gemma3:1b", "gemma3:4b", "llama3.2:1b", "llama3.2:3b", "mistral"],
        index=0,
        help="Must already be pulled locally via `ollama pull <model>`",
    )
    temperature = st.slider("Creativity (temperature)", 0.0, 1.5, 0.7, 0.1)
    word_limit = st.slider("Answer length limit (words)", 50, 500, 100, 10)
    st.divider()
    if st.button("🗑️ Clear history"):
        st.session_state.history = []
        st.rerun()

if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------------------------------------------------------
# Task selector
# ---------------------------------------------------------------------------
task = st.radio(
    "What do you want to do?",
    options=["Q&A", "Summarizer", "Sentiment Analyzer", "MCQ Generator"],
    horizontal=True,
)

st.divider()

# ---------------------------------------------------------------------------
# Task-specific system prompts and inputs
# ---------------------------------------------------------------------------
if task == "Q&A":
    system_prompt = f"""You are an AI expert designed to answer any user-asked question.
Rules:
1. Answer within {word_limit} words.
2. Mention the type of source/knowledge you're drawing from (e.g., general knowledge, reasoning).
3. Structure the output in two sections: **Details** and **Example**.
"""
    user_input = st.text_input("Ask me anything", placeholder="e.g., What is quantum entanglement?")
    submit_label = "Ask"

elif task == "Summarizer":
    system_prompt = f"""You are a professional summarization assistant.
Rules:
1. Summarize the given text within {word_limit} words.
2. Preserve key facts, names, and numbers.
3. Structure the output in two sections: **Summary** and **Key Points** (bullet list).
"""
    user_input = st.text_area("Paste text to summarize", height=200, placeholder="Paste an article, email, or report here...")
    submit_label = "Summarize"

elif task == "Sentiment Analyzer":
    system_prompt = """You are a sentiment analysis assistant.
Rules:
1. Classify the overall sentiment as Positive, Negative, or Neutral.
2. Give a confidence score out of 100.
3. Structure the output in two sections: **Verdict** (sentiment + score) and **Explanation** (short reasoning, max 60 words).
"""
    user_input = st.text_area("Paste text to analyze", height=150, placeholder="e.g., a review, tweet, or feedback message...")
    submit_label = "Analyze"

else:  # MCQ Generator
    col1, col2 = st.columns([3, 1])
    with col1:
        topic_input = st.text_area("Topic or passage", height=150, placeholder="e.g., Photosynthesis, or paste a passage...")
    with col2:
        num_q = st.number_input("Number of MCQs", min_value=1, max_value=10, value=3)
        difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"])
    user_input = topic_input
    system_prompt = f"""You are an MCQ (Multiple Choice Question) generator for educational purposes.
Rules:
1. Generate exactly {num_q} multiple-choice questions at {difficulty} difficulty based on the given topic/passage.
2. Each question must have 4 options labeled A-D, with exactly one correct answer.
3. Structure the output in two sections: **Questions** (numbered, with options) and **Answer Key** (list of correct letters).
"""
    submit_label = "Generate MCQs"

submitted = st.button(f"🚀 {submit_label}", type="primary")

# ---------------------------------------------------------------------------
# Run the model
# ---------------------------------------------------------------------------
if submitted and user_input.strip():
    with st.spinner("Thinking..."):
        try:
            response = chat(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input},
                ],
                options={"temperature": temperature},
            )
            output = response["message"]["content"]

            st.session_state.history.insert(0, {
                "task": task,
                "input": user_input,
                "output": output,
                "time": datetime.datetime.now().strftime("%H:%M:%S"),
            })

        except Exception as e:
            st.error(f"⚠️ Something went wrong: {e}\n\nMake sure Ollama is running and `{model_name}` is pulled.")

elif submitted:
    st.warning("Please enter some input first.")

# ---------------------------------------------------------------------------
# Display latest result + history
# ---------------------------------------------------------------------------
if st.session_state.history:
    latest = st.session_state.history[0]
    st.subheader(f"✨ Result — {latest['task']}")
    st.markdown(latest["output"])
    st.download_button(
        "⬇️ Download response",
        data=latest["output"],
        file_name=f"{latest['task'].lower().replace(' ', '_')}_result.txt",
        mime="text/plain",
    )

    if len(st.session_state.history) > 1:
        with st.expander(f"📜 History ({len(st.session_state.history) - 1} earlier)"):
            for item in st.session_state.history[1:]:
                st.markdown(f"**[{item['time']}] {item['task']}**")
                st.caption(item["input"][:150] + ("..." if len(item["input"]) > 150 else ""))
                st.markdown(item["output"])
                st.divider()
else:
    st.info("Fill in the fields above and click the button to get started.")
