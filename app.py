import streamlit as st
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

st.set_page_config(page_title="RAG with Agents", page_icon="🤖")

st.title("🤖 RAG with Agents")
st.write("An AI-powered document question-answering system.")

api_key = st.sidebar.text_input("OpenRouter API Key", type="password")

documents = st.text_area(
"Enter your documents",
placeholder="Paste your document content here..."
)

question = st.text_input("Ask your question")

@st.cache_resource
def load_model():
return SentenceTransformer("all-MiniLM-L6-v2")

def retrieve_context(documents, question):
chunks = [
documents[i:i+500]
for i in range(0, len(documents), 500)
]

model = load_model()
embeddings = model.encode(chunks)
query_embedding = model.encode([question])

scores = cosine_similarity(
    query_embedding, embeddings
)[0]

best_index = int(np.argmax(scores))
return chunks[best_index]

def research_agent(context, question, client):
response = client.chat.completions.create(
model="openai/gpt-4o-mini",
messages=[
{
"role": "system",
"content": "You are a research agent. Analyze the retrieved context and answer accurately."
},
{
"role": "user",
"content": f"Context: {context}\nQuestion: {question}"
}
]
)
return response.choices[0].message.content

def review_agent(answer, question, client):
response = client.chat.completions.create(
model="openai/gpt-4o-mini",
messages=[
{
"role": "system",
"content": "You are a review agent. Improve the answer for clarity and correctness using only the supplied answer. Do not add unsupported facts."
},
{
"role": "user",
"content": f"Question: {question}\nAnswer: {answer}"
}
]
)
return response.choices[0].message.content

if st.button("Generate Answer"):

if not api_key or not documents.strip() or not question.strip():
    st.warning("Please enter API key, documents and question.")

else:
    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )

        with st.spinner("Research agent is retrieving information..."):
            context = retrieve_context(documents, question)

        with st.spinner("Review agent is preparing the answer..."):
            initial_answer = research_agent(
                context, question, client
            )

            final_answer = review_agent(
                initial_answer, question, client
            )

        st.subheader("Final Answer")
        st.write(final_answer)

        with st.expander("Retrieved Context"):
            st.write(context)

    except Exception as e:
        st.error(f"Error: {e}")
