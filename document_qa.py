from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain

import chainlit as cl
from chainlit.types import AskFileResponse

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chuck_overlap=200, add_start_index=True)
embeddings = OpenAIEmbeddings()

def process_file(file: AskFileResponse):
    import tempfile
    if file.type == "text/plain":
        Loader = TextLoader
    elif file.type == "application/pdf":
        Loader = PyPDFLoader
    with tempfile.NamedTemporaryFile() as tempfile:
        tempfile.write(file.content)
        loader = Loader(tempfile.name)
        documents = loader.load()
        docs = text_splitter.split_documents(documents)
        for idx, doc in enumerate(docs):
            doc.metadata["source"] = f"source_{idx}"
        return docs
    
def get_docsrch(file: AskFileResponse):
    docs = process_file(file)
    cl.user_session.set("docs", docs)
    docsrch = Chroma.from_documents(docs, embeddings)
    return docsrch

@cl.on_chat_start
async def on_chat_start():
    contents = "You can now chat with your pdf"
    welcome = "welcome! Please upload a pdf or text file"
    await cl.Message(content=contents).send()
    files = None
    while files is None:
        files = await cl.AskFileMessage(content=welcome, accept=["text/plain", "application/pdf"], max_size_mb=100, timeout=180).send()

    file = files[0]
    msg = cl.Message(content=f"Processing `{file.name}` !")
    await msg.send()
    doc_srch = await cl.make_async(get_docsrch(file))

    chain = RetrievalQAWithSourcesChain.from_chain_type(
        ChatOpenAI(temperature=1, streaming=True), chain_type="stuff", retriever=doc_srch.as_retriever(max_token_limit=4097)
    )

    msg.content = f" `{file.name}` processed. You can now ask questions!"
    await msg.update()

    cl.user_session.set("chain", chain)
    