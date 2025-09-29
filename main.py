import os
import textwrap
from dotenv import load_dotenv 

# Imports da Ingestão
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Imports da Consulta e Geração
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq # Conexão com o Groq

# --- Configurações Globais ---
# 1. Carrega as variáveis de ambiente
load_dotenv() 

# 2. Caminhos
# Documento PDF na pasta data
PDF_PATH = "data/codigocivil.pdf" 
VECTOR_DB_PATH = "model/faiss_index" # Onde o banco é salvo/carregado
EMBEDDING_MODEL = "all-MiniLM-L6-v2" # Modelo de embeddings
# Linha de configuração (um sofrimento pra tentar vários modelos):
GROQ_MODEL = "llama-3.1-8b-instant" 

# ---------------------------------------
# FUNÇÕES DA FASE 1: INGESTÃO E INDEXAÇÃO
# ---------------------------------------

def load_and_clean_pdf(file_path: str, pages_to_skip: int = 15):
    """
    Carrega o PDF, verifica a existência do arquivo e remove páginas de ruído.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo PDF não encontrado em: {file_path}")
        
    print(f"Carregando o arquivo: {file_path}")
    loader = PyMuPDFLoader(file_path=file_path)
    docs_por_pagina = loader.load()
    
    # Remove as N primeiras páginas (capa, índice)
    cleaned_docs = docs_por_pagina[pages_to_skip:]
    print(f"Páginas carregadas: {len(docs_por_pagina)}. Páginas úteis (após limpeza): {len(cleaned_docs)}")
    
    return cleaned_docs

def split_documents_intelligently(documents):
    """
    Divide os documentos em chunks, priorizando a quebra por "Art." (Chunking Jurídico).
    """
    # A lista de separadores 
    separators = ["Art.", "art.", "\n\n", "\n", " "]

    text_splitter = RecursiveCharacterTextSplitter(
        separators=separators,
        chunk_size=2500, 
        chunk_overlap=300
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Documentos segmentados em {len(chunks)} chunks (Artigos/Seções).")
    return chunks

def create_and_save_vector_db(chunks):
    """
    Cria os embeddings e salva o índice FAISS (o banco de dados do RAG).
    """
    print("Iniciando a criação de Embeddings. Isso pode levar alguns minutos...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}
    )
    
    # Cria e salva o Vector Store FAISS
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    os.makedirs(os.path.dirname(VECTOR_DB_PATH), exist_ok=True)
    vectorstore.save_local(VECTOR_DB_PATH)
    print(f"Vector Store FAISS criado e salvo com sucesso em: {VECTOR_DB_PATH}")
    return vectorstore

# ----------------------------------------------------------------------
# FUNÇÕES DA FASE 2: CONSULTA E GERAÇÃO
# ----------------------------------------------------------------------

def load_embedding_model(model_path):
    """Carrega o modelo de embedding (necessário para carregar o FAISS e buscar a similaridade)."""
    return HuggingFaceEmbeddings(
        model_name = model_path,
        model_kwargs = {'device':'cpu'},
        encode_kwargs = {'normalize_embeddings': True}
    )

def create_rag_chain(vectorstore):
    """Cria a cadeia RAG usando Groq (Mistral), Retriever e Prompt."""
    
    # Conexão com o Groq (Ele automaticamente pega a GROQ_API_KEY do .env)
    llm = ChatGroq(temperature=0, model_name=GROQ_MODEL)

    # O Retriever busca os 5 chunks mais relevantes do código civil
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # O Prompt Template - (crucial para o RAG)
    template = """### System: Você atua como assistente jurídico especializado no Código Civil.
    Sua função é responder às perguntas do usuário **usando apenas o contexto de lei fornecido**. 
    Sempre cite o número do Artigo ou a seção exata do documento que fundamenta sua resposta.
    Caso o contexto não forneça a resposta, diga que a informação não foi encontrada no Código.
    c
    ### Contexto:{context}
    ### Usuário:{question}
    ### Resposta:"""
    
    prompt = PromptTemplate.from_template(template)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain

def get_response(query, chain):
    """Executa a consulta RAG e formata a saída."""
    print(f"\n--- CONSULTA: {query} ---")
    
    response = chain.invoke({'query': query})

    wrapped_text = textwrap.fill(response['result'], width=100)
    print(f"\nRESPOSTA DO ASSISTENTE:\n{wrapped_text}")

    print("\nFONTES RECUPERADAS (Fundamentação):")
    for doc in response['source_documents']:
        print(f"  - Fonte (Página {doc.metadata.get('page')}): {doc.page_content[:60]}...")
    print("-" * 105)

# ----------------------------------------------------------------------
# EXECUÇÃO PRINCIPAL (ORQUESTRAÇÃO DAS FASES)
# ----------------------------------------------------------------------

if __name__ == "__main__":
    
    # 1. LÓGICA DE INGESTÃO: Cria o banco SÓ SE ele não existir.
    if not os.path.exists(VECTOR_DB_PATH):
        print("Banco de vetores não encontrado. Iniciando a Ingestão...")
        try:
            cleaned_documents = load_and_clean_pdf(PDF_PATH)
            legal_chunks = split_documents_intelligently(cleaned_documents)
            create_and_save_vector_db(legal_chunks)
            print("\n--- INGESTÃO CONCLUÍDA. ---")
        except FileNotFoundError:
            print(f"\nERRO: O arquivo '{PDF_PATH}' não foi encontrado. Coloque o PDF na pasta 'data/' e tente novamente.")
            exit()
    else:
        print(f"Banco de vetores encontrado em '{VECTOR_DB_PATH}'. Pulando Ingestão.")


    # 2. FASE DE CONSULTA: 
    try:
        # Carrega o modelo de embedding
        embed_model = load_embedding_model(EMBEDDING_MODEL)
        
        # Carrega o banco de vetores FAISS
        vector_db = FAISS.load_local(VECTOR_DB_PATH, embed_model, allow_dangerous_deserialization=True)
        
        # Cria a cadeia RAG com o Groq 
        rag_chain = create_rag_chain(vector_db)

        print("\n--- INICIANDO CONSULTAS RAG COM GROQ (Mixtral) ---")
        
        # Demonstração de Consultas
        get_response("Quais são os direitos da personalidade?", rag_chain)
        get_response("Quem deve ser responsabilizado por danos causados a terceiros, de acordo com o Art. 927?", rag_chain)
        get_response("Existe alguma forma de sociedade empresarial no Código Civil?", rag_chain)
        
    except Exception as e:
        print(f"\nERRO CRÍTICO NA CONSULTA: {e}")
        print("Verifique: 1. Sua GROQ_API_KEY. 2. A conexão com a internet. 3. O formato do seu .env.")