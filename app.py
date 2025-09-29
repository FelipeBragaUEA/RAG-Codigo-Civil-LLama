import os
import streamlit as st
from dotenv import load_dotenv 
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

# --- CONFIGURAÇÕES GLOBAIS ---
load_dotenv() 

VECTOR_DB_PATH = "model/faiss_index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# O único que funcionou após 10 tentativas foi o LLama 3.1 8B Instant 
GROQ_MODEL = "llama-3.1-8b-instant" 

# --- FUNÇÕES ESSENCIAIS ---

# Decorador que garante que esta função só será executada uma vez
@st.cache_resource
def load_rag_components():
    """Carrega o modelo de embedding e o banco FAISS."""
    
    # 1. Carrega o modelo de embedding (para buscar no FAISS)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # 2. Carrega o banco de vetores FAISS
    try:
        if not os.path.exists(VECTOR_DB_PATH):
            st.error("ERRO: O banco de vetores FAISS não foi encontrado. Execute o 'main.py' primeiro para criar o índice.")
            return None
            
        vector_db = FAISS.load_local(
            VECTOR_DB_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        return vector_db
        
    except Exception as e:
        st.error(f"Erro ao carregar o banco de dados: {e}")
        return None

def create_rag_chain(vectorstore):
    """Cria a cadeia RAG usando Groq, Retriever e Prompt."""
    
    # Pega a GROQ_API_KEY do .env
    llm = ChatGroq(temperature=0, model_name=GROQ_MODEL)

    # Retriever busca os 5 chunks mais relevantes
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # Prompt Template
    template = """### System: Você atua como assistente jurídico especializado no Código Civil.
    Sua função é responder às perguntas do usuário **usando apenas o contexto de lei fornecido**. 
    Sempre cite o número do Artigo ou o trecho do documento que fundamenta sua resposta.
    Caso o contexto não forneça a resposta, diga que a informação não foi encontrada no Código.
    
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

# --- INTERFACE STREAMLIT ---

st.set_page_config(page_title="Assistente Jurídico RAG - Código Civil", layout="wide")

st.title("⚖️ Assistente Jurídico RAG - Código Civil")
st.markdown("Consulte instantaneamente o Código Civil. As respostas são geradas pelo **Llama 3.1** (via Groq) e fundamentadas no documento. (Tempo de resposta: < 1s)")

# Carrega os componentes do RAG
vector_db = load_rag_components()

if vector_db:
    # Cria a cadeia RAG 
    rag_chain = create_rag_chain(vector_db)

    # Área de entrada para a pergunta do usuário
    user_query = st.text_input("Faça sua pergunta sobre o Código Civil:", placeholder="Ex: Qual a pena para o crime de furto simples?")

    if user_query:
        with st.spinner("Buscando e gerando resposta..."):
            try:
                # Executa a cadeia RAG
                response = rag_chain.invoke({'query': user_query})

                # --- EXIBIÇÃO DA RESPOSTA ---
                
                st.subheader("✅ Resposta do Assistente:")
                st.info(response['result'])
                
                # --- EXIBIÇÃO DA FUNDAMENTAÇÃO ---
                st.subheader("📚 Fundamentação (Fontes do Código Civil):")
                
                for i, doc in enumerate(response['source_documents']):
                    st.markdown(f"**Fonte {i+1} (Pág. {doc.metadata.get('page')})**")
                    st.code(doc.page_content[:500] + "...", language='text') 
                    
            except Exception as e:
                if "401" in str(e) or "GROQ_API_KEY" in str(e):
                    st.error("Erro de Autenticação: Verifique se sua GROQ_API_KEY está correta no arquivo .env.")
                else:
                    st.error(f"Erro na consulta: {e}")