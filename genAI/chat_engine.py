# chat_engine.py
"""
Handles the main conversational logic using OpenAI models and
contextual retrieval from ChromaDB.
"""

import openai
from chroma_store import get_vector_store

def chat_with_docs(prompt, api_key, model, history):
    """
    Generate a conversational response using OpenAI API and contextual retrieval.

    Args:
        prompt (str): User query or message.
        api_key (str): OpenAI API key.
        model (str): Selected OpenAI model (e.g., 'gpt-4o', 'gpt-3.5-turbo').
        history (list): List of previous chat messages.

    Returns:
        tuple: (response, updated_history)
    """
    openai.api_key = api_key

    # Retrieve relevant context from vector store
    vectordb = get_vector_store()
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(prompt)

    # Combine retrieved documents as context
    context = "\n\n".join([d.page_content for d in docs])
    full_prompt = f"Context:\n{context}\n\nUser: {prompt}\nAssistant:"

    # Call OpenAI ChatCompletion API
    completion = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful and knowledgeable assistant."},
            {"role": "user", "content": full_prompt}
        ]
    )

    # Extract response
    response = completion.choices[0].message["content"]

    # Append to chat history
    history.append({"user": prompt, "assistant": response})

    return response, history
