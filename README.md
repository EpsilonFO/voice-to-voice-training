# Voice-to-Voice Streaming avec Gradium et Mistral

Ce projet implémente une interface vocale complète utilisant l'API Gradium pour la reconnaissance vocale (STT) et la synthèse vocale (TTS), combinée avec le LLM Mistral pour le traitement du langage naturel.

## Caractéristiques

- **Streaming bout-en-bout** : Minimise la latence entre la fin de votre phrase et la réponse de l'agent
- **Détection automatique de la fin de parole** : Utilise la détection de silence pour savoir quand vous avez fini de parler
- **Gestion intelligente des phrases** : Accumule le texte par phrases complètes avant de les vocaliser
- **Conversation continue** : Maintient l'historique de la conversation pour un contexte cohérent

## Prérequis

- Python 3.8+
- Un microphone fonctionnel
- Des haut-parleurs ou casque audio
- Une clé API Gradium valide
- Une clé API Mistral valide

## Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/votre-repo/gradium-mistral-v2v.git
cd gradium-mistral-v2v
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Installer le SDK Gradium

Suivez les instructions de Gradium pour installer leur SDK Python. Vous pouvez généralement faire :

```bash
pip install gradium
```

### 4. Configurer les clés API

Modifiez le fichier `mistralv2v.py` pour y insérer vos clés API :

```python
# Configuration API
VOICE_ID = "b35yykvVppLXyw_l"  # Elise - voix française
GRADIUM_API_KEY = "votre_cle_gradium_ici"

# Configuration Mistral
MISTRAL_API_KEY = "votre_cle_mistral_ici"
MISTRAL_MODEL = "mistral-large-latest"  # ou un autre modèle Mistral
```

## Utilisation

### Lancer l'application

```bash
python mistralv2v.py
```

### Interface

1. Appuyez sur Entrée pour commencer à parler
2. Parlez naturellement - le système détectera automatiquement quand vous avez fini
3. L'agent répondra vocalement en streaming
4. Appuyez sur 'q' puis Entrée pour quitter

### Visualisation

Pendant que vous parlez, vous verrez une barre de niveau audio :

```
[MIC] Parlez...
[########--##############--]
```

## Architecture Technique

### Flux de données

```
Microphone → [STT Gradium] → Texte → [LLM Mistral] → Texte → [TTS Gradium] → Haut-parleurs
```

### Optimisations de streaming

1. **STT en chunks** : L'audio est envoyé par morceaux de 80ms pour un traitement en temps réel
2. **LLM streaming** : Les tokens sont traités au fur et à mesure de leur génération
3. **TTS par phrases** : La synthèse vocale commence dès qu'une phrase complète est disponible
4. **WebSockets persistants** : Les connexions sont réutilisées pour minimiser la latence

### Gestion des erreurs

Le système inclut une gestion robuste des erreurs avec :
- Reconnexion automatique
- Gestion des timeouts
- Affichage des erreurs compréhensibles

## Configuration Avancée

### Paramètres audio

Vous pouvez ajuster les paramètres dans le code :

```python
# Seuil de détection de silence (RMS)
silence_threshold = 300

# Durée de silence pour considérer la fin de parole (secondes)
silence_duration = 1.2

# Durée minimale d'enregistrement (secondes)
min_recording_time = 1.5
```

### Modèles

Changez les modèles utilisés :

```python
# Modèle STT (par défaut: "default")
# Modèle TTS (par défaut: "default")
# Modèle Mistral (ex: "mistral-medium", "mistral-small")
```

## Dépannage

### Problèmes audio

- **Aucun son** : Vérifiez que vos haut-parleurs sont bien configurés
- **Microphone non détecté** : Vérifiez les permissions et la configuration du microphone
- **Latence élevée** : Assurez-vous d'avoir une bonne connexion internet

### Problèmes API

- **Erreur 401** : Vérifiez vos clés API
- **Erreur de connexion** : Vérifiez votre connexion internet
- **Quota dépassé** : Contactez le support Gradium ou Mistral

## Contribution

Les contributions sont bienvenues ! N'hésitez pas à ouvrir des issues ou des pull requests.

## Licence

MIT - Voir le fichier LICENSE pour plus de détails.

## Remerciements

- Gradium pour leur API audio de haute qualité
- Mistral AI pour leurs modèles de langage performants
- La communauté open-source pour les outils utilisés

---

*Made with ❤️ for voice interfaces*