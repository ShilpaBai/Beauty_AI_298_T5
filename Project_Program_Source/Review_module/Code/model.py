import asyncio
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl
from langchain_community.chat_models import ChatAnyscale

DB_FAISS_PATH = '/Users/shilpabai/Downloads/DATA298A/Data/visio/vectorstores/db_faiss'

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.



Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

template = """
              Write a summary of the product review delimited by double backticks.
              Return your response which covers the key points of the text.

              Context: {context}
              Question: {question}
              SUMMARY:
           """

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=template,
                            input_variables=['context', 'question'])
    return prompt

# Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain

# Loading the model
def load_llm():
    # Load the locally downloaded model here
    
    llm = ChatAnyscale(model="meta-llama/Llama-2-13b-chat-hf", anyscale_api_key='esecret_5datfql4wjbg1mqgshlvutaary')
    
    return llm

# QA Model Function
async def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

# Output function
async def final_result(query):
    qa_result = await qa_bot()
    response = await qa_result({'query': query})
    return response

# chainlit code
@cl.on_chat_start
async def start():
    chain = await qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to Beauty Chat Bot. What is your query?"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]
    #sources = res["source_documents"]

    #if sources:
       # answer += f"\nSources:" + str(sources)
    #else:
        #answer += "\nNo sources found"

    await cl.Message(content=answer).send()

if __name__ == "__main__":
    asyncio.run(cl.main())