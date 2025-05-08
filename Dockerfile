FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04

# ============================== #
# Εγκατάσταση εργαλείων συστήματος
# ============================== #

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    poppler-utils \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# ============================== #
# Ρύθμιση Python ως default
# ============================== #
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# ============================== #
# Δημιουργία φακέλου εφαρμογής
# ============================== #
WORKDIR /app

# ============================== #
# Αντιγραφή requirements.txt
# ============================== #
COPY requirements.txt requirements.txt

# ============================== #
# Εγκατάσταση Python βιβλιοθηκών
# ============================== #
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# ============================== #
# Αντιγραφή υπόλοιπου κώδικα
# ============================== #
COPY . .

# ============================== #
# Άνοιγμα πόρτας Streamlit
# ============================== #
EXPOSE 7860

# ============================== #
# Εκκίνηση εφαρμογής Streamlit
# ============================== #
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
