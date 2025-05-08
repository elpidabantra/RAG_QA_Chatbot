---
license: apache-2.0
title: Chatbot Q&A
sdk: streamlit
colorFrom: blue
short_description: A RAG app using DeepSeek to answer questions based on .pdf
---

# 📄 RAG PDF Q&A με Quantized DeepSeek-7B

**Ένας βοηθός που απαντάει στις ερωτήσεις του χρήστη αποκλειστικά με βάση το κείμενο που του δόθηκε από τον χρήστη (PDF) και όχι με βάση την εσωτερική του γνώση.**

---

## 🚀 Βασίζεται σε:

- **LangChain** για chunking και ανάκτηση συμφραζομένων (retrieval)
- **MiniLM (sentence-transformers/all-MiniLM-L6-v2)** για embeddings
- **ChromaDB** ως προσωρινό vector store
- **DeepSeek LLM (deepseek-ai/deepseek-llm-7b-chat)** σε 4-bit quantization
- **Streamlit** για το περιβάλλον χρήστη
- **langid** για αναγνώριση γλώσσας

---

## 🏷️ Χαρακτηριστικά

✅ **Ανέβασμα PDF** (Ελληνικά ή Αγγλικά)  
✅ Αυτόματη εξαγωγή κειμένου  
✅ **Chunking** με μέγεθος 1000 και overlap 300  
✅ Δημιουργία embeddings με **MiniLM**  
✅ **ChromaDB** για προσωρινό vector store  
✅ **Καθαρισμός retriever και vectorstore** όταν αλλάζει PDF (δεν κρατά παλιές πληροφορίες)  
✅ **Αναγνώριση γλώσσας** (langid)  
✅ **Απάντηση στη γλώσσα της ερώτησης** (με αυτόματη μετάφραση όταν χρειάζεται)  
✅ Επιστροφή **IDs των chunks** που χρησιμοποιήθηκαν για τη δημιουργία της απάντησης  
✅ Αν δεν βρεθεί σχετική πληροφορία, απαντά **"I do not know."** ή **"Δεν γνωρίζω."**  
✅ **Quantization 4-bit** του **DeepSeek LLM** για αποδοτική χρήση GPU (A10g)  
✅ **Cache** του LLM και των pipelines για αποφυγή επανυπολογισμών

---

## 🗂️ Αρχεία

| Όνομα αρχείου    | Περιγραφή                      |
|------------------|-------------------------------|
| `app.py`         | Κύριο Streamlit app           |
| `requirements.txt` | Απαιτούμενες βιβλιοθήκες    |
| `Dockerfile`     | Ρυθμίσεις για Hugging Face Space |
| `README.md`      | Αυτό το αρχείο                |

---

**Σημείωση:**  
✔ Υποστηρίζεται paraphrasing μόνο για ερωτήσεις στα Αγγλικά.  
✔ Για Ελληνικά paraphrasing δεν υπάρχει κατάλληλο διαθέσιμο μοντέλο αυτή τη στιγμή στο Hugging Face.

