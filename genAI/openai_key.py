# openai_key.py
"""
This module validates an OpenAI API key by performing a simple API call.
"""

import openai
import streamlit as st

def validate_openai_key(api_key: str) -> bool:
    """
    Validate a provided OpenAI API key by attempting to list available models.

    Args:
        api_key (str): The OpenAI API key provided by the user.

    Returns:
        bool: True if the key is valid, False otherwise.
    """
    try:
        openai.api_key = api_key
        openai.models.list()
        return True
    except Exception as e:
        st.error(f"❌ Invalid OpenAI API Key: {e}")
        return False
