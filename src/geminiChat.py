"""
At the command line, only need to run once to install the package via pip:

$ pip install google-generativeai
"""

import google.generativeai as genai

# genai.configure(api_key="AIzaSyD0YK3pcVXsFxBw7ehQcJzc55xGBI-s5Tc")
genai.configure(api_key="AIzaSyCDtWoevnUrXQ_G9-41EE10KDNyTozQEuM")


# Set up the model
generation_config = {
  "temperature": 0.9,
  "top_p": 1,
  "top_k": 0,
  "max_output_tokens": 2048,
}

safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_NONE"
  },
]

model = genai.GenerativeModel(model_name="gemini-1.0-pro",
                              generation_config=generation_config,
                              safety_settings=safety_settings)

convo = model.start_chat(history=[
  {
    "role": "user",
    "parts": ["Now you are an AI book assistant. The assistant should be able to perform various tasks related to books."]
  },
  {
    "role": "model",
    "parts": ["**AI Book Assistant**"]
  },
])

convo.send_message("find me best 5 mangas")
print(convo.last.text)
print(convo.history)