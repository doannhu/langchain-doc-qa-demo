from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain

import mlflow

import chainlit as cl
from chainlit.types import AskFileResponse

from dotenv import load_dotenv
import os
load_dotenv()
secret_key = os.getenv('SECRET_KEY')

#
mlflow.set_experiment("document_qa")
#

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                chunk_overlap=200,
                                                length_function=len,
                                                is_separator_regex=False,)
embeddings = OpenAIEmbeddings(openai_api_key=secret_key)

def process_file(file: AskFileResponse):
    # import tempfile
    if file.type == "text/plain":
        Loader = TextLoader
    elif file.type == "application/pdf":
        Loader = PyPDFLoader

    # with tempfile.NamedTemporaryFile() as tempfile:
        # tempfile.write(file.content)
    loader = Loader(file.path)
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
    doc_srch = await cl.make_async(get_docsrch)(file)

    chain = RetrievalQAWithSourcesChain.from_chain_type(
        ChatOpenAI(openai_api_key=secret_key, temperature=1, streaming=True), chain_type="stuff", retriever=doc_srch.as_retriever(max_token_limit=4097)
    )

    #
    with mlflow.start_run() as run:
        model_info = mlflow.langchain.log_model(
            chain,
            artifact_path="retrieval_qa",
            loader_fn=doc_srch.as_retriever(max_token_limit=4097),
    )
    #

    msg.content = f" `{file.name}` processed. You can now ask questions!"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def on_message(message):
    chain = cl.user_session.get("chain")
    cb = cl.LangchainCallbackHandler(stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"])
    cb.answer_reached = True
    response = await chain.acall(message.content, callbacks=[cb])
    answer = response["answer"]
    src = response["sources"].strip()
    src_elements = []

    docs = cl.user_session.get("docs")
    metadatas = [doc.metadata for doc in docs]
    all_sources = [m["source"] for m in metadatas]

    if src:
        found_sources = []

        for source in src.split(","):
            src_name = src.strip().replace(".", "")
            try:
                index = all_sources.index(src_name)
            except ValueError:
                continue
            text = docs[index].page_content
            found_sources.append(src_name)
            src_elements.append(cl.Text(content=text, name=src_name))

        if found_sources:
            answer += f"\nSources: {', '.join(found_sources)}"
        else:
            answer += "\n No source found"

        if cb.has_streamed_final_answer:
            cb.final_stream.elements = src_elements
            await cb.final_stream.update()
        else:
            await cl.Message(content=answer, elements=src_elements).send()    
