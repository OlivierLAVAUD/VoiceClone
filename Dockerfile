# Utiliser une image de base avec Python et CUDA si disponible
FROM nvidia/cuda:12.1.1-base-ubuntu22.04

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Créer et définir le répertoire de travail
WORKDIR /app

# Copier les fichiers nécessaires
COPY main/app.py .
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Créer le répertoire pour les enregistrements
RUN mkdir -p /app/voice_cloning_recordings

# Exposer le port utilisé par Gradio
EXPOSE 7860

# Commande pour lancer l'application
CMD ["python3", "app.py"]