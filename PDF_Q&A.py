from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import VectorDBQA, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader, PyPDFLoader
import openai
import os

os.environ["OPENAI_API_KEY"] = "***"

os.listdir(".") 

#ファイルを選択

loader = PyPDFLoader('')
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(texts, embeddings)

qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(model_name="gpt-3.5-turbo"), chain_type="stuff", retriever=vectordb.as_retriever())

from langchain import PromptTemplate

template = """
あなたは親切なアシスタントです。下記の質問に日本語で回答してください。
質問：{question}
回答：
"""

prompt = PromptTemplate(
    input_variables=["question"],
    template=template,
)

import gradio as gr

def add_text(history, text):
    history = history + [(text, None)]
    return history, ""

def bot(history):
    query = history[-1][0]
    query = prompt.format(question=query)
    answer = qa.run(query)
    source = qa._get_docs(query)[0]
    source_sentence = source.page_content
    answer_source = source_sentence +"\n"+"source:"+source.metadata["source"] + ", page:" + str(source.metadata["page"])
    history[-1][1] = answer # + "\n\n情報ソースは以下です：\n" + answer_source
    return history

with gr.Blocks() as UI:
    chatbot = gr.Chatbot([], elem_id="chatbot").style(height=400)

    with gr.Row():
        with gr.Column(scale=0.6):
            txt = gr.Textbox(
                show_label=False,
                placeholder="Enter text and press enter",
            ).style(container=False)

    txt.submit(add_text, [chatbot, txt], [chatbot, txt]).then(
        bot, chatbot, chatbot
    )

UI.launch()