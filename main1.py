from flask import Flask, render_template, request, jsonify

import langchain
import pickle

app = Flask(__name__)

# Load the FAISS index from the pickle file
with open("faiss_store_openai.pkl", "rb") as f:
    vectorstore = pickle.load(f)

# Create a LangChain chain from the pre-trained LLM
llm = OpenAI(temperature=0.9, max_tokens=500)
chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())

@app.route("/answer", methods=["POST"])
def answer():
    """Answers the user query using the LangChain chain."""

    # Get the query from the request body
    query = request.get_json()["question"]

    # Generate the answer using the LangChain chain
    result = chain({"question": query}, return_only_outputs=True)

    # Return the answer as JSON
    return jsonify({"answer": result["answer"], "sources": result.get("sources", "")})

if __name__ == "__main__":
    app.run()
