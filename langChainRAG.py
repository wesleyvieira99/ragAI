import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Certifique-se de configurar sua chave de API
# os.environ["OPENAI_API_KEY"] = "sua-chave-aqui"

# 1. Carregar o documento
loader = TextLoader("documento.txt", encoding="utf-8")
docs = loader.load()

# 2. Dividir em blocos menores (chunking)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(docs)

# 3. Criar embeddings e armazenar no banco de dados vetorial (Chroma)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# 4. Criar um Retriever a partir do vectorstore
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 5. Definir o template do prompt
template = """Você é um assistente prestativo. Use apenas o contexto fornecido abaixo para responder à pergunta. 
Se você não souber a resposta com base no contexto, diga que não sabe.

Contexto:
{context}

Pergunta:
{question}

Resposta:"""

prompt = ChatPromptTemplate.from_template(template)

# 6. Inicializar o Modelo de Linguagem (LLM)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Função auxiliar para formatar documentos recuperados em texto
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 7. Montar o Pipeline usando LCEL (LangChain Expression Language)
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 8. Executar o pipeline com uma pergunta
resposta = rag_chain.invoke("Qual é o assunto principal do documento?")
print(resposta)
