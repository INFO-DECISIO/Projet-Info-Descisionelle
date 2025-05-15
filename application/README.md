# Projet : Analyseur de Sentiment IA

Ce projet a pour objectif de développer un système complet d'analyse de sentiments, capable de déterminer si un texte exprime une émotion positive, négative ou neutre. Il se compose d'une API backend développée avec Flask (Python) pour la logique de prédiction et d'une application frontend interactive construite avec Vue.js.

**Dépôts GitHub Associés :**
*   **API Backend (Flask) :** [https://github.com/INFO-DECISIO/Api-Test](https://github.com/INFO-DECISIO/Api-Test) (et code source principal dans [Projet-Info-Descisionelle/tree/main/application](https://github.com/INFO-DECISIO/Projet-Info-Descisionelle/tree/main/application))
*   **Application Frontend (Vue.js) :** [https://github.com/INFO-DECISIO/Front-end-](https://github.com/INFO-DECISIO/Front-end-)

---

## Table des Matières

1.  [Introduction Générale](#introduction-générale)
2.  [Architecture du Système](#architecture-du-système)
3.  [Backend : API d'Analyse de Sentiment (Flask)](#backend--api-danalyse-de-sentiment-flask)
    *   [Fonctionnalités](#fonctionnalités-api)
    *   [Prérequis](#prérequis-api)
    *   [Installation et Démarrage (API)](#installation-et-démarrage-api)
    *   [Structure des Fichiers (API)](#structure-des-fichiers-api)
    *   [Configuration (API)](#configuration-api)
    *   [Endpoints de l'API](#endpoints-de-lapi)
    *   [Logique des Modèles](#logique-des-modèles-api)
4.  [Frontend : Interface Utilisateur (Vue.js)](#frontend--interface-utilisateur-vuejs)
    *   [Fonctionnalités](#fonctionnalités-frontend)
    *   [Prérequis](#prérequis-frontend)
    *   [Installation et Démarrage (Frontend)](#installation-et-démarrage-frontend)
    *   [Structure des Fichiers (Frontend)](#structure-des-fichiers-frontend)
    *   [Composants Clés](#composants-clés-frontend)
    *   [Interaction avec l'API](#interaction-avec-lapi-frontend)
5.  [Modèles de Machine Learning](#modèles-de-machine-learning)
6.  [Contributions et Améliorations Possibles](#contributions-et-améliorations-possibles)
7.  [Licence](#licence)

---

## 1. Introduction Générale

L'analyse de sentiments, ou "opinion mining", est un domaine du Traitement du Langage Naturel (NLP) qui vise à identifier et extraire des informations subjectives à partir de sources textuelles. Ce projet met en œuvre un pipeline complet, de l'entraînement de modèles de Machine Learning à leur déploiement via une API et leur utilisation dans une interface web conviviale. L'objectif est de fournir des prédictions de sentiment (positif, négatif, neutre) pour un texte donné, avec une indication de la confiance du modèle.

## 2. Architecture du Système

Le système est architecturé en deux composants principaux :

*   **Backend (API Flask) :**
    *   Sert de cerveau à l'application.
    *   Héberge les modèles de Machine Learning pré-entraînés.
    *   Expose des endpoints pour recevoir du texte (et potentiellement des métadonnées) et retourner des prédictions de sentiment.
    *   Gère la logique de chargement des modèles, la prédiction, et le formatage des résultats.
*   **Frontend (Application Vue.js) :**
    *   Fournit une interface utilisateur interactive.
    *   Permet aux utilisateurs de saisir du texte.
    *   Communique avec l'API backend pour obtenir les analyses de sentiment.
    *   Affiche les résultats de manière visuelle et intuitive (émojis, barre de confiance, distribution des probabilités).

Ces deux composants communiquent via des requêtes HTTP (typiquement RESTful).

---

## 3. Backend : API d'Analyse de Sentiment (Flask)

L'API backend est construite avec Flask, un micro-framework Python léger et flexible.

### Fonctionnalités (API)

*   Chargement de modèles de sentiment pré-entraînés (format `.joblib`).
*   Gestion de deux types de modèles :
    *   Modèle "texte seul" : prédit le sentiment basé uniquement sur le contenu textuel.
    *   Modèle "texte + métadonnées" : prend en compte le texte et des informations contextuelles (ex: plateforme, moment de la journée).
*   Endpoints pour :
    *   La prédiction de sentiment.
    *   La vérification de l'état de santé de l'API et des modèles (`/health`).
    *   L'obtention d'informations sur les modèles chargés (`/models/info`).
*   Formatage des réponses en JSON, incluant le sentiment prédit, la probabilité de confiance, et les probabilités brutes pour chaque classe.
*   Introduction d'un seuil de confiance pour identifier les prédictions "indéterminées".
*   Logging des requêtes et des erreurs.

### Prérequis (API)

*   Python 3.7+
*   Les bibliothèques listées dans `requirements.txt` (créé à partir de `pip freeze > requirements.txt` dans l'environnement du projet) :
    *   `Flask`
    *   `Flask-CORS` (pour permettre les requêtes cross-origin depuis le frontend)
    *   `scikit-learn`
    *   `joblib`
    *   `pandas`
    *   `numpy`
    *   `nltk` (si le prétraitement est effectué dans l'API)

### Installation et Démarrage (API)

1.  **Clonez le dépôt de l'API :**
    ```bash
    git clone https://github.com/INFO-DECISIO/Api-Test.git
    # ou le dépôt principal contenant le code de l'application
    git clone https://github.com/INFO-DECISIO/Projet-Info-Descisionelle.git
    cd Projet-Info-Descisionelle/application # ou le répertoire racine de l'API
    ```

2.  **Créez et activez un environnement virtuel :**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Installez les dépendances :**
    ```bash
    pip install -r requirements.txt
    # Si requirements.txt n'existe pas, installez manuellement :
    # pip install Flask Flask-CORS scikit-learn joblib pandas numpy nltk
    ```

4.  **Placez les modèles pré-entraînés :**
    Assurez-vous que les fichiers modèles (`sentiment_model_text_only.joblib` et `sentiment_model_text_metadata.joblib`) sont présents dans le répertoire configuré (par défaut `app/models/` relatif au script de démarrage de Flask, ou selon les chemins dans `config.py`).

5.  **Configurez les variables d'environnement (si nécessaire) :**
    L'application peut utiliser des variables d'environnement Flask pour la configuration (ex: `FLASK_APP`, `FLASK_ENV`, chemins des modèles). Reportez-vous au fichier de configuration (ex: `config.py`).

6.  **Démarrez le serveur Flask :**
    ```bash
    flask run
    # Ou, si vous utilisez un script de démarrage comme run.py :
    # python run.py
    ```
    Par défaut, l'API devrait être accessible à l'adresse `http://127.0.0.1:5000`.

### Structure des Fichiers (API) (Basée sur le code fourni)


application/
├── app/
│ ├── api/
│ │ ├── init.py # Initialisation du Blueprint API
│ │ └── routes.py # Définition des endpoints de l'API
│ ├── services/
│ │ ├── init.py
│ │ └── predictor.py # Classe SentimentPredictor pour la logique ML
│ ├── models/ # Répertoire pour les fichiers .joblib des modèles
│ │ ├── sentiment_model_text_only.joblib
│ │ └── sentiment_model_text_metadata.joblib
│ ├── init.py # Factory de l'application Flask
│ └── config.py # Configuration de l'application (chemins, mapping, etc.)
├── venv/ # Environnement virtuel
├── run.py # (Optionnel) Script pour démarrer l'application
└── requirements.txt # Dépendances Python

### Configuration (API)

La configuration de l'application Flask est gérée via un fichier `config.py` (ou des variables d'environnement). Les paramètres configurables incluent :

*   `MODEL_TEXT_PATH`: Chemin vers le modèle "texte seul".
*   `MODEL_FULL_PATH`: Chemin vers le modèle "texte + métadonnées".
*   `SENTIMENT_MAP`: Dictionnaire mappant les codes de sentiment prédits (0, 1, 2) à des labels, valeurs de sortie, et emojis.
*   `CONFIDENCE_THRESHOLD`: Seuil de probabilité en dessous duquel une prédiction est considérée comme "indéterminée".
*   `UNDETERMINED`: Définition du résultat pour les prédictions indéterminées.

### Endpoints de l'API

*   **`GET /health`**
    *   Description : Vérifie l'état de santé de l'API et l'état de chargement des modèles.
    *   Réponse :
        ```json
        {
          "status": "ok",
          "timestamp": 1678886400.000,
          "models": {
            "text_only": true,
            "text_metadata": true
          }
        }
        ```

*   **`POST /predict`** (ou `/test-predict` selon votre route)
    *   Description : Prédit le sentiment d'un texte.
    *   Corps de la requête (JSON) :
        *   Pour "texte seul" :
            ```json
            {
              "text": "C'est une journée magnifique !"
            }
            ```
        *   Pour "texte + métadonnées" :
            ```json
            {
              "use_metadata": true,
              "cleaned_text": "super film facebook soir", // Note: idéalement, l'API ferait le nettoyage
              "Platform": "Facebook",
              "Time of Tweet": "night"
            }
            ```
    *   Réponse (JSON) :
        ```json
        {
          "success": true,
          "result": {
            "label": "positive",
            "value": 1,
            "emoji": "😄"
          },
          "probability": 0.95, // Probabilité de la classe prédite
          "all_probabilities": [0.05, 0.95, 0.00], // Probabilités pour [neutral, positive, negative]
          "error": null
        }
        ```
        En cas d'erreur :
        ```json
        {
          "success": false,
          "result": null,
          "probability": null,
          "all_probabilities": null,
          "error": "Message d'erreur"
        }
        ```

*   **`GET /models/info`**
    *   Description : Fournit des informations sur les modèles disponibles et le mapping des sentiments.
    *   Réponse (JSON) :
        ```json
        {
          "text_only": {
            "name": "Multinomial NB (Text Only)",
            "loaded": true,
            "path": "app/models/sentiment_model_text_only.joblib"
          },
          "text_metadata": {
            "name": "SVC Linear (Text + Metadata)",
            "loaded": true,
            "path": "app/models/sentiment_model_text_metadata.joblib"
          },
          "sentiment_classes": {
            "neutral": {"label": "neutral", "value": 0, "emoji": "😐"},
            "positive": {"label": "positive", "value": 1, "emoji": "😄"},
            "negative": {"label": "negative", "value": -1, "emoji": "😔"},
            "undetermined": {"label": "undetermined", "value": null, "emoji": "🤔"}
          }
        }
        ```

### Logique des Modèles (API)

La classe `SentimentPredictor` (`app/services/predictor.py`) gère :

*   Le chargement paresseux des modèles `.joblib`.
*   La distinction entre les prédictions "texte seul" et "texte + métadonnées".
*   La préparation des données d'entrée pour les pipelines Scikit-learn.
*   L'appel aux méthodes `predict()` et `predict_proba()` des modèles.
*   Le formatage de la réponse finale en utilisant le `SENTIMENT_MAP` et le `CONFIDENCE_THRESHOLD` de la configuration.

---

## 4. Frontend : Interface Utilisateur (Vue.js)

L'application frontend est développée avec Vue.js (version 3) et Vite comme outil de build. Elle utilise Tailwind CSS pour le style.

### Fonctionnalités (Frontend)

*   Interface utilisateur responsive et esthétique pour l'analyse de sentiment.
*   Zone de saisie de texte avec compteur de caractères et feedback visuel.
*   Analyse de sentiment en temps réel (avec debounce) lors de la saisie.
*   Affichage clair du résultat :
    *   Emoji dynamique représentant le sentiment.
    *   Texte descriptif du sentiment.
    *   Barre de confiance animée indiquant la certitude du modèle.
    *   Graphique en anneau (Doughnut) affichant la distribution des probabilités pour chaque classe de sentiment.
*   Indicateur de chargement pendant l'analyse.
*   Boutons pour effacer le texte ou forcer une nouvelle analyse.

### Prérequis (Frontend)

*   Node.js (version compatible avec Vite et Vue 3, ex: LTS)
*   npm (généralement inclus avec Node.js) ou yarn

### Installation et Démarrage (Frontend)

1.  **Clonez le dépôt du frontend :**
    ```bash
    git clone https://github.com/INFO-DECISIO/Front-end-.git
    cd Front-end-
    ```

2.  **Installez les dépendances :**
    ```bash
    npm install
    ```

3.  **Configurez l'URL de l'API Backend :**
    Dans `src/components/SentimentAnalyzer.vue`, modifiez la variable `apiBaseUrl` pour qu'elle pointe vers votre API Flask (par défaut `http://127.0.0.1:5000` si l'API tourne localement).
    ```javascript
    // src/components/SentimentAnalyzer.vue
    // ...
    data() {
      return {
        // ...
        apiBaseUrl: 'http://127.0.0.1:5000', // Vérifiez cette URL
        // ...
      };
    },
    // ...
    ```

4.  **Démarrez le serveur de développement Vite :**
    ```bash
    npm run dev
    ```
    L'application devrait être accessible à l'adresse indiquée par Vite (souvent `http://localhost:5173`).

5.  **Pour la production :**
    ```bash
    npm run build
    ```
    Ceci générera les fichiers statiques dans le répertoire `dist/`, prêts à être déployés sur un serveur web.

### Structure des Fichiers (Frontend) (Basée sur le dump fourni)
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END

Front-end-/
├── public/ # Assets statiques
├── src/
│ ├── assets/ # CSS global, images
│ │ ├── base.css
│ │ └── main.css # Inclut Tailwind CSS
│ ├── components/ # Composants Vue
│ │ ├── SentimentAnalyzer.vue # Composant principal de l'application
│ │ ├── postcss.config.js # Config PostCSS (pour Tailwind)
│ │ ├── tailwind.config.js # Config Tailwind
│ │ └── icons/ # (Icônes SVG si utilisées)
│ ├── App.vue # Composant racine de l'application
│ └── main.js # Point d'entrée de l'application Vue
├── .gitignore
├── index.html # Template HTML principal
├── jsconfig.json # (ou tsconfig.json) Configuration JS/TS
├── package.json # Dépendances et scripts NPM
├── vite.config.js # Configuration de Vite
└── README.md

*Note : Les fichiers `postcss.config.js` et `tailwind.config.js` sont typiquement à la racine du projet, mais peuvent fonctionner dans `src/components/` si la configuration de build le prend en compte.*

### Composants Clés (Frontend)

*   **`App.vue` :** Composant racine qui charge le `SentimentAnalyzer`.
*   **`SentimentAnalyzer.vue` :** Cœur de l'interface utilisateur. Il gère :
    *   L'état local (message saisi, sentiment, confiance, etc.).
    *   La logique de saisie utilisateur et le debounce.
    *   Les appels API vers le backend Flask via `axios`.
    *   La mise à jour de l'UI en fonction des réponses de l'API.
    *   L'affichage des graphiques (barre de confiance, DoughnutChart avec `vue-chart-3`).
    *   Les animations (GSAP, transitions Vue).

### Interaction avec l'API (Frontend)

*   La communication avec le backend se fait via la bibliothèque `axios`.
*   L'endpoint `/test-predict` (ou `/predict`) de l'API est appelé pour obtenir les analyses.
*   La méthode `analyzeSentiment` dans `SentimentAnalyzer.vue` gère la requête POST et la mise à jour des données du composant avec la réponse.
*   Un `debounce` est utilisé sur la saisie pour limiter le nombre d'appels API.

---

## 5. Modèles de Machine Learning

Les modèles de Machine Learning utilisés par l'API sont entraînés séparément (probablement dans un notebook Jupyter ou un script Python dédié). Les étapes typiques de leur création incluent :

1.  **Collecte et Exploration des Données (EDA) :** Analyse d'un dataset de textes étiquetés par sentiment.
2.  **Prétraitement du Texte :** Nettoyage (minuscules, ponctuation), suppression des stopwords, stemming ou lemmatisation.
3.  **Vectorisation :** Transformation du texte en représentations numériques (ex: TF-IDF).
4.  **Entraînement :** Apprentissage de classifieurs (ex: Naive Bayes, Régression Logistique, SVM, XGBoost) sur les données vectorisées.
5.  **Évaluation :** Mesure des performances avec des métriques comme l'Accuracy, le F1-score, et analyse des matrices de confusion et des courbes d'apprentissage.
6.  **Sélection et Sauvegarde :** Choix du/des meilleur(s) modèle(s) et sauvegarde au format `.joblib` (pipeline complet incluant vectorizer et classifieur).

Les modèles actuellement configurés par défaut dans l'API sont (basé sur `predictor.py`) :
*   **Texte Seul :** `Multinomial NB (Text Only)`
*   **Texte + Métadonnées :** `SVC Linear (Text + Metadata)`

*Il est crucial que le prétraitement appliqué aux textes lors de la prédiction dans l'API soit identique à celui utilisé lors de l'entraînement des modèles.*

---

## 6. Contributions et Améliorations Possibles

*   **Backend :**
    *   Intégrer le prétraitement du texte directement dans l'API.
    *   Ajouter une validation plus stricte des données d'entrée.
    *   Optimiser les performances pour un grand nombre de requêtes (ex: Gunicorn, caching).
    *   Permettre le rechargement des modèles sans redémarrer l'API.
*   **Frontend :**
    *   Rendre l'URL de l'API configurable via des variables d'environnement.
    *   Améliorer la gestion des erreurs et le feedback utilisateur.
    *   Internationalisation (i18n) de l'interface.
    *   Tests unitaires et e2e.
*   **Modèles ML :**
    *   Entraîner sur des datasets plus volumineux et diversifiés.
    *   Expérimenter avec des techniques de prétraitement plus avancées (lemmatisation, gestion des négations).
    *   Tester des architectures de modèles plus complexes (embeddings de mots, Transformers comme BERT).
    *   Mettre en place une optimisation rigoureuse des hyperparamètres.



