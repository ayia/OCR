#!/bin/bash

# 1. Vérifier si le dossier venv existe
if [ ! -d "venv" ]; then
    echo "Création de l'environnement virtuel Python..."
    python3 -m venv venv
fi

# 2. Activer l'environnement virtuel
source venv/bin/activate

# 3. Installer les dépendances
echo "Installation des dépendances Python..."
pip install --upgrade pip
pip install -r requirements.txt

# 4. Lancer l'API avec Uvicorn
echo "Lancement de l'API sur http://0.0.0.0:9091 ..."
uvicorn app:app --reload --host 0.0.0.0 --port 9091 