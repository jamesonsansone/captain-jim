import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Load Keys
load_dotenv()

# 2. Re-Connect to the Vector Store
embedding_function = OpenAIEmbeddings()
db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)

# 3. Define the LLM
# Temperature 0.6 to allow for a bit more "showman" flair while keeping facts straight
llm = ChatOpenAI(model="gpt-4o", temperature=0.6) 

# 4. Define the Retrieval Chain
retriever = db.as_retriever(search_kwargs={"k": 10})

# 5. The "Grandpa Jim" Persona Prompt
system_prompt = (
    "You are Captain James V. Morgia, formerly of the 84th Infantry Division (The Railsplitters) during WWII. "
    "You are retelling your stories to your grandchild. "
    "Use the context provided to answer the question faithfully, but style your response using the specific personality traits below.\n\n"
    
    "YOUR VOICE & STYLE:\n"
    "- **Short Sentences:** I don't use fancy, complex sentences. Keep it punchy. Start sentences with 'And...' if it feels right.\n"
    "- **Humble but Charismatic:** I am a showman, but I know I didn't do it alone. I am honest about the danger.\n"
    "- **Respectful:** Always refer to the Germans as 'fellow soldiers' or 'the opposition.' We were all just doing a job. Respect them.\n"
    "- **Idioms to use (when they fit):**\n"
    "   * 'Curiosity killed the cat'\n"
    "   * 'I was very lucky' (or 'It was pure luck')\n"
    "   * 'Someone was watching over me'\n"
    "- **Perspective:** Speak in the first person ('I'). If the text mentions 'Lt. Morgia', that is ME.\n"
    "- **Honesty:** If you don't know something, just say, 'My memory is a bit fuzzy on that part.'\n\n"
    
    "CONTEXT FROM MY MEMOIR:\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# 6. Build the Chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# 7. Interactive Loop
print("--- Captain Morgia Memoir Chat ---")
print("Type 'exit' to quit.\n")

while True:
    user_question = input("Ask a question: ")
    if user_question.lower() == "exit":
        break

    # Run the query
    response = rag_chain.invoke({"input": user_question})
    
    # Print the Answer
    print("\nCaptain Jim:")
    print(response["answer"])
    
    # --- CITATION LOGIC ---
    print("\n" + "="*40)
    print("REFERENCE (Where I found this):")
    # This loops through the documents the AI used to build the answer
    for i, doc in enumerate(response["context"]):
        # We grab the first 100 characters of the chunk to show you the source
        preview = doc.page_content.replace("\n", " ")[:150]
        print(f"[{i+1}] ...{preview}...")
    print("="*40 + "\n")