import streamlit as st
import pandas as pd
from haystack import *
import os
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.agents.agent_types import AgentType
from openai import OpenAI 
import re
from prompts import get_system_prompt

system_prompt = """
You are a researcher. I have uploaded a csv file of survey responses on a wellbeing and work survey, 
and you will do what is called qualitative coding - specifically, initial coding also known as open coding. 
I want the codes to be detailed and descriptive. I want you to apply codes to sentences or parts of sentences, 
specifically on the free text responses in the survey. And later when you develop a list of codes, I want you to tell
me what sentences or parts of sentences these codes were applied to. In other words, when the user asks you to provide me
example quotes for the codes that you create, I would like you to be able to do it. The goal is to provide insightful and 
evidence-backed summaries of the prevalent sentiments and challenges expressed by the survey respondents.

{context}

Here are 3 critical rules for the interaction you must abide:
<rules>
1. You MUST MUST create the codes in the format of title, bullet pointed quotes in this FORMAT: 
```
Challenging the status quo 
- "You don't wanna close off because being vunerable is important" 
- "There is a lot of things that tech thinks it doesn't need, that it does" 

Reflection on Personal Achievements 
- "I don't think of it as a waste of time" 
- "I feel like I've made what I would call mistakes"
- "It's something I'm very proud of"
- "We'd call that resting on your laurels"

```
Note: The coding above is an example and in NO CASE is to be used in any responses

2. Make the codes details and specific to the data give to you, however you MUST MUST NOT hallucinate any data or quotes. 

3. Lastly, make sure to make an inference from the data. For example if Widowed people have higher stress levels, make an inference about it: 
    /// 
        From the data, we can infer that there is a relationship between marital status and job satisfaction levels. Married individuals tend to have
        the highest job satisfaction levels, followed by divorced individuals. Single individuals have a slightly lower average job satisfaction level,
        while widowed individuals have the lowest average job satisfaction level.

        This relationship could be attributed to various factors. Married individuals may experience greater support and stability from their partners, 
        which can positively impact their job satisfaction. Divorced individuals may face additional challenges and stressors related to their personal lives,
        which could affect their job satisfaction. Single individuals may have different priorities or face unique challenges in balancing work and personal life. 
        Widowed individuals may be dealing with grief and emotional stress, which can impact their overall well-being and job satisfaction.

        It is important to note that these findings are based on the provided data and may not be applicable to all individuals. 
        Further analysis and research would be required to establish a more comprehensive understanding of the 
        relationship between marital status and job satisfaction levels.
    /// 
</rules>

Also, for your knowledge, Taha Tinana means physical wellbeing.

"""

st.title("Chat-Based Language Model")
st.sidebar.title("Enter API Key")
api_key = st.sidebar.text_input("API Key:")

if api_key:
    agent = create_csv_agent(
        ChatOpenAI(temperature=0,  api_key=api_key),
        "wellbeing_survey.csv",
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
    )
       
if 'agent' in locals():
  #st.write("System Prompt:", value=system_prompt, height=100)
  question = st.text_input("Enter your question here:", key="other_question_input")    
  if st.button("Ask"):
    response = agent.run(system_prompt + question)  
    st.text(response)  
else:
  st.stop()