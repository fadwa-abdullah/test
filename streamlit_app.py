import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.title("📚 RAG Q&A System (FastAPI + Streamlit)")

col1, col2 = st.columns([3, 1])
with col2:
    if st.button("Update Index"):
        with st.spinner("Updating index..."):
            res = requests.post(f"{API_URL}/update-index")
        st.success(res.json().get("status", "Index updated!"))

query = st.text_input("Ask a question about your documents:")

if query:
    with st.spinner("Searching documents..."):
        res = requests.post(f"{API_URL}/ask", json={"query": query})
        answer = res.json().get("answer", "No answer found.")
    st.write("### Answer:")
    st.write(answer)
