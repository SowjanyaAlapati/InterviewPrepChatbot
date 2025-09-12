import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# ---- Load data & model once ----
@st.cache_data
def load_data():
    return pd.read_csv("../data/interview_questions.csv")

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

df = load_data()
model = load_model()

# ---- Initialize session state ----
if "step" not in st.session_state:
    st.session_state.step = 0        # current question index
    st.session_state.score = []      # list of scores
    st.session_state.category = None
    st.session_state.filtered = None

st.title("💼 AI Interview Simulator")

# --- Step 0: choose category & count ---
if st.session_state.step == 0:
    st.subheader("Setup")
    cats = ["Any"] + sorted(df.Category.unique())
    category = st.selectbox("Choose category:", cats)
    num = st.number_input("How many questions?", 1, len(df), 5)
    if st.button("Start Interview"):
        if category == "Any":
            st.session_state.filtered = df.sample(num).reset_index(drop=True)
        else:
            st.session_state.filtered = (
                df[df.Category == category].sample(num).reset_index(drop=True)
            )
        st.session_state.step = 1
        st.rerun()

# --- Step ≥1: questions loop ---
elif st.session_state.step <= len(st.session_state.filtered):
    qidx = st.session_state.step - 1
    row = st.session_state.filtered.iloc[qidx]
    st.subheader(f"Question {st.session_state.step}: {row.Question}")
    user_ans = st.text_area("Your Answer", key=f"ans_{qidx}")

    if st.button("Submit Answer", key=f"btn_{qidx}"):
        # evaluate
        ideal = row["IdealAnswer"]
        emb1 = model.encode([user_ans], convert_to_tensor=True)
        emb2 = model.encode([ideal], convert_to_tensor=True)
        score = float(util.pytorch_cos_sim(emb1, emb2)[0][0]) * 10
        st.session_state.score.append(score)
        st.session_state.step += 1
        st.rerun()

# --- Finished ---
else:
    st.success("Interview Complete!")
    avg = sum(st.session_state.score) / len(st.session_state.score)
    st.write(f"Your Average Score: **{avg:.2f}/10**")
    for i, row in st.session_state.filtered.iterrows():
        st.markdown(f"**Q{i+1}:** {row.Question}")
        st.markdown(f"- IdealAnswer: {row['IdealAnswer']}")
        st.markdown(f"- Your Score: {st.session_state.score[i]:.2f}/10")
    if st.button("Restart"):
        for k in ["step", "score", "category", "filtered"]:
            st.session_state.pop(k, None)
        st.rerun()
