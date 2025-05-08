import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import torch
import uuid
import os
import langid

st.title("RAG PDF Q&A με DeepSeek-7B και Paraphrasing (μόνο Αγγλικά)")

st.sidebar.header("Ανέβασε PDF")
pdf = st.sidebar.file_uploader("Επίλεξε PDF", type="pdf")

if pdf:
    temp_filename = f"temp_{uuid.uuid4()}.pdf"
    with open(temp_filename, "wb") as f:
        f.write(pdf.read())

    loader = PyPDFLoader(temp_filename)
    pages = loader.load()

    st.sidebar.success(f"Το έγγραφο έχει {len(pages)} σελίδες.")

    # ---- Φόρτωση DeepSeek ----
    @st.cache_resource
    def load_llm():
        MODEL_ID = "deepseek-ai/deepseek-llm-7b-chat"

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        return tokenizer, model

    tokenizer, model = load_llm()

    # ---- Paraphrasing για Αγγλικά ----
    @st.cache_resource
    def load_en_paraphraser():
        return pipeline("text2text-generation", model="ramsrigouthamg/t5_paraphraser")

    paraphraser_en = load_en_paraphraser()

    translation_pipeline = pipeline("translation", model="Helsinki-NLP/opus-mt-en-el")

    total_words = sum(len(page.page_content.split()) for page in pages)
    avg_words_per_page = total_words / len(pages)

    st.sidebar.info(f"Μέσος όρος λέξεων ανά σελίδα: {int(avg_words_per_page)}")

    proposed_chunk_size = 1000
    proposed_overlap = 300

    user_chunk_size = st.sidebar.number_input("Επίλεξε Chunk size", value=proposed_chunk_size, step=50)
    user_overlap = st.sidebar.number_input("Επίλεξε Overlap", value=proposed_overlap, step=50)

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", "! ", "; ", "? ", "  ", " "],
        chunk_size=user_chunk_size,
        chunk_overlap=user_overlap
    )

    docs = text_splitter.split_documents(pages)

    # ---- Προσθήκη custom ids ----
    for idx, doc in enumerate(docs):
        doc.metadata["custom_id"] = idx

    st.success(f"Επεξεργάστηκαν {len(docs)} chunks.")

    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # ---- Ανέβηκε νέο PDF; Καθάρισε vectordb ----
    if st.session_state.get("loaded_filename") != temp_filename:
        st.session_state.vectordb = None
        st.session_state.retriever = None
        st.session_state.docs = None
        st.session_state.loaded_filename = temp_filename

    if not st.session_state.get("vectordb"):
        vectordb = Chroma(
            collection_name=f"rag_pdf_collection_{uuid.uuid4()}",
            embedding_function=embedding_function
        )
        vectordb.add_documents(docs)
        retriever = vectordb.as_retriever(
            search_kwargs={"k": 6}
        )
        st.session_state.vectordb = vectordb
        st.session_state.retriever = retriever
        st.session_state.docs = docs
    else:
        retriever = st.session_state.retriever
        docs = st.session_state.docs

    st.sidebar.success("Έτοιμο το retriever.")

    # ---- Rephrasing ----
    def rephrase_question(original_question, lang):
        variations = [original_question]
        paraphrase_prompt = f"paraphrase: {original_question} </s>"

        try:
            if lang == "el":
                # Δεν υπάρχει διαθέσιμο paraphrasing για ελληνικά προς το παρόν
                rephrases = []
            else:
                output = paraphraser_en(paraphrase_prompt, max_length=64, num_return_sequences=3, do_sample=True)
                rephrases = list({o['generated_text'].strip() for o in output})
            variations.extend(rephrases)
        except Exception as e:
            st.sidebar.warning("Το paraphrasing απέτυχε. Χρησιμοποιούμε μόνο την αρχική ερώτηση.")

        return variations

    def generate_answer(question):
        detected_lang = langid.classify(question)[0]
        if detected_lang == "el":
            lang_instruction = "Απάντησε στα Ελληνικά."
            fallback_response = "Δεν γνωρίζω."
        else:
            lang_instruction = "Answer in English."
            fallback_response = "I do not know."

        variations = rephrase_question(question, detected_lang)

        st.sidebar.info(f"Εναλλακτικές ερωτήσεις: {variations}")

        all_docs = []
        for var in variations:
            docs_found = retriever.get_relevant_documents(var)
            all_docs.extend(docs_found)

        unique_docs = list({doc.page_content: doc for doc in all_docs}.values())
        context = "\n\n".join([doc.page_content for doc in unique_docs])

        chunk_ids = []
        for doc in unique_docs:
            if "custom_id" in doc.metadata:
                chunk_ids.append(doc.metadata["custom_id"])

        prompt = f"""
{lang_instruction}
Χρησιμοποίησε ΜΟΝΟ τα συμφραζόμενα.
Αν δεν υπάρχει απάντηση στα συμφραζόμενα, πες ρητά: {fallback_response}
Συμφραζόμενα:
{context}
Ερώτηση: {question}
Απάντηση:"""

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        output = model.generate(
            **inputs,
            max_new_tokens=250,
            temperature=0.2,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id,
        )

        full_answer = tokenizer.decode(output[0], skip_special_tokens=True)

        if "Απάντηση:" in full_answer:
            clean_answer = full_answer.split("Απάντηση:")[-1].strip()
        else:
            clean_answer = full_answer.strip()

        if clean_answer == "" or any(
            bad in clean_answer.lower()
            for bad in ["απάντησε", "συμφραζόμενα", question.lower()]
        ):
            clean_answer = fallback_response

        if detected_lang == "el" and clean_answer != fallback_response:
            if sum(c.isalpha() and c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ' for c in clean_answer) > len(clean_answer) * 0.4:
                translation = translation_pipeline(clean_answer)[0]['translation_text']
                clean_answer = translation

        return clean_answer, chunk_ids

    question = st.text_input("Γράψε την ερώτησή σου:")
    if question:
        with st.spinner("Ανάκτηση και παραγωγή απάντησης..."):
            answer, chunk_ids = generate_answer(question)
            st.markdown("**Απάντηση:**")
            st.success(answer)
            st.info(f"Χρησιμοποιήθηκαν τα chunks με ID: {chunk_ids}")

    os.remove(temp_filename)

else:
    st.info("Περιμένω να ανεβάσεις ένα PDF.")
