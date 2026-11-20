# Genesis Studio — améliorations possibles

Classées par rapport « impact / effort ». Tout ce qui suit est **techniquement
faisable dans l'architecture actuelle** (le code est découpé en `core/` indépendant de
l'interface, donc chaque ajout est un module, pas une réécriture).

**État produit — v1.1.0 :** les étapes 1 et 2 ci-dessous sont livrées. Le canal GitHub
Releases a été ajouté au build Windows (bundle complet initial, patches différentiels,
validation SHA-256, sauvegarde/rollback et préservation des données utilisateur). Il reste
à exécuter le workflow et le build Windows réels après la première publication du dépôt ;
ce canal n'est pas encore validé pour distribution.

---

## 1. Le plus rentable : générer des corpus avec une IA gratuite

**MVP implémenté en v1.1** : l'onglet Corpus propose Ollama local, Groq et OpenRouter
(free uniquement), puis importe le résultat dans le projet. Cette première version ne
fabrique pas un grand corpus à elle seule : elle sert à produire quelques exemples ou
à enrichir un corpus. L'échelle de données requise reste dans le tableau du README.

### Offres à considérer (quotas et conditions à revérifier avant usage)

| Service | Ce que le gratuit donne | Limite pratique | Remarques |
|---|---|---|---|
| **Groq** | exemple officiel sur `openai/gpt-oss-120b` : 30 req/min, 1 000 req/jour, 8 000 tokens/min et 200 000 tokens/jour | les limites changent par modèle/organisation ; prévoir erreurs 429 et reprise | candidat pour de petits volumes de textes publics, après examen des conditions |
| **OpenRouter** (modèles `:free`) | routeur et variantes gratuites ; les quotas sont consultables par clé (les plafonds courants sont bas) | disponibilité / modèle / politique de données dépendent du fournisseur amont | pratique pour tester, mais pas de repli automatique vers un modèle payant |
| **Cerebras / Cloudflare Workers AI** | plans/quota gratuits selon service, modèle et région | vérifier le tableau de bord et les limites actuelles | candidats secondaires, après vérification de la licence et de l'usage des données |
| **Ollama en local** (Llama / Qwen / Mistral) | **illimité, gratuit, hors ligne** | limité par votre machine | le plus sûr pour la confidentialité ; idéal par défaut |

> **À ne pas brancher comme source gratuite de corpus pour ce projet : Google Gemini API.**
> Les conditions Gemini en vigueur interdisent d'utiliser le service pour développer des
> modèles concurrents ; elles réservent en outre l'accès des API clients distribués dans
> l'EEE aux services payants. Le quota gratuit réutilise aussi le contenu pour améliorer
> les produits Google. Vérifier les conditions à nouveau avant toute intégration.

### Ce que je brancherais concrètement

```
genesis/core/generators/
    client.py      # Groq, OpenRouter, Ollama local (pas Gemini)
    prompts.py     # modèles de prompts par catégorie (récits, dialogues, articles, FAQ…)
    dedup.py       # MinHash/SimHash : éliminer les textes quasi identiques
    quality.py     # filtre : longueur, langue, répétitions, taux de mots rares
```

- **Fait** : onglet Corpus → « Générer avec une IA », choix fournisseur, modèle, langue,
  catégorie, thème, style, tokens visés, max de requêtes et sortie par réponse.
- **Fait** : génération en worker Qt, arrêt borné à la fin de la requête, sortie partielle
  conservée après quota/erreur, import comme fichier corpus.
- **Fait** : déduplication approximative intra-session et filtre de réponses très courtes /
  bruitées. Le nettoyage de la totalité des fichiers importés reste à faire à l'étape 3.
- **Fait** : clé jamais dans les projets ; mémoire seule par défaut et mémorisation DPAPI
  facultative sous Windows. Les appels réels et DPAPI doivent encore être essayés sur
  Windows/avec des comptes fournisseur.
- **Garde-fous implémentés** : l'utilisateur choisit l'API explicitement et confirme
  l'envoi du prompt, jamais de clé dans les sauvegardes de projet, pas de Gemini pour
  générer les données d'entraînement du modèle, et mode local Ollama par défaut. Pour les API distantes : textes publics uniquement, contrôle
  de la licence et du droit d'utiliser les sorties, pas de données personnelles/secrètes.
- Le texte synthétique ne doit pas constituer automatiquement 100 % du corpus : il peut
  répéter des erreurs et appauvrir le style. Garder un jeu de validation de textes
  indépendants, dédupliquer, mélanger avec des sources autorisées et afficher la provenance.
- Estimation initiale : **3 à 5 jours**. Le MVP est déjà livré en v1.1 ; le travail restant
  concerne les essais sur fournisseurs réels, la provenance, la validation et la robustesse.

---

## 2. Performances & mémoire (grands gains, effort modéré)

| Amélioration | Effet | Effort |
|---|---|---|
| `scaled_dot_product_attention` (SDPA) / Flash-Attention | **Fait en v1.1** : SDPA automatique, PyTorch choisit le kernel adapté ; le chemin manuel sert de repli/test. Aucun facteur de gain n'est promis. | ✅ |
| `torch.compile` | **Fait en v1.1** : option avancée désactivée par défaut, compilation paresseuse, repli eager sans perte du pas si backend indisponible ; checkpoint portable. Gain à mesurer par machine. | ✅ |
| `cudnn.benchmark` | déjà activé pour CUDA via `gpu.configure_backends` ; peut aider ou être neutre selon forme des lots. | ✅ |
| Gradient checkpointing | entraîner un modèle 2× plus gros sur la même carte (un peu plus lent) | ~3 h |
| Optimiseur 8-bit / Adafactor | état d'optimiseur ÷4 en mémoire → lots plus gros | ~4 h |
| Quantification int8/int4 à la génération | charger un modèle 2–4× plus gros en VRAM pour discuter | ~1 j |
| Compilation des checkpoints en `safetensors` + mmap | chargement instantané, moins de RAM | ~4 h |

## 3. Qualité du corpus et de l'apprentissage

- **Déduplication** du corpus (MinHash) : supprime les répétitions qui favorisent le par-cœur.
- **Filtre de langue / de qualité** (longueur, ponctuation, taux de majuscules).
- **Packing** des documents courts en fenêtres pleines (moins de tokens perdus au padding).
- **Curriculum** : commencer par les documents longs et propres.
- **LoRA / fine-tuning** : réutiliser un modèle de base et n'entraîner qu'une petite
  partie des poids ; entraînement 10× plus rapide sur petite carte.
- **Tokenizer partagé** entre versions (aujourd'hui : un tokenizer par projet, ce qui
  est déjà cohérent, mais un tokenizer commun accélérerait les comparaisons).

## 4. Discussion et évaluation

- **Comparateur A/B** : deux versions côte à côte, même prompt, même graine.
- **RAG sur le corpus** : retrouver les passages proches (BM25 ou embeddings légers) et
  les injecter dans le prompt → le modèle « connaît » vos documents.
- **Détection de par-cœur** : mesurer le recouvrement entre les générations et le corpus
  (n-grammes communs) et afficher un avertissement « cette version récite ».
- **Perplexité + export CSV/PNG** des courbes pour suivre les progrès.
- **Contraintes de sortie** (JSON, listes) et pénalité de répétition dans l'interface.
- **Historique de conversation** exportable en Markdown, avec multi-tours et troncature
  intelligente du contexte.

## 5. Confort d'usage et produit

- **File d'attente d'entraînements** (lancer plusieurs versions à la suite, la nuit),
  avec arrêt programmé et notification Windows à la fin.
- **Modèles d'exemple inclus** (un petit modèle déjà entraîné sur les corpus fournis) pour
  comprendre l'application en 30 secondes.
- **Assistant de premier lancement** (3 écrans : corpus → gabarit → entraîner).
- **Aide contextuelle** (F1) et documentation intégrée.
- **Export Hugging Face** (`config.json`, `model.safetensors`, tokenizer) et import d'un
  GPT-2 from scratch → vos modèles deviennent utilisables ailleurs.
- **Sauvegarde/restauration** d'un projet en un fichier `.zip` (avec vérification).
- **Thème clair** + `i18n` (fichiers `.qm`) : l'interface est déjà 100 % en français,
  tout le texte est centralisé.
- **Plugins** : un dossier `plugins/` avec une API `register_tokenizer`,
  `register_preset`, `register_dataset_source` chargée au démarrage.
- **Installeur Inno Setup** + mise à jour automatique + signature de code (sinon
  Windows SmartScreen affiche un avertissement au premier lancement).

## 6. Ordre que je suivrais

1. **SDPA + `torch.compile` — terminé en v1.1.** Le gain de vitesse dépend du GPU et doit être mesuré sur une machine NVIDIA réelle.
2. **Générateur de corpus — MVP terminé en v1.1** (Ollama local, Groq, OpenRouter gratuit ; prompt, consentement, clé DPAPI facultative, sortie partielle, filtre léger et dédup intra-session). Reste à tester sur de vrais services/modèles et à raffiner les quotas/provenances.
3. **Déduplication / filtre qualité à l'échelle du corpus — prochain chantier.** Le premier filtre ne compare que les réponses générées dans une session ; il ne nettoie pas encore un corpus déjà importé.
4. Détection de par-cœur + comparateur A/B (3 j) → permet de juger objectivement.
5. Gradient checkpointing + quantification (3 j) → modèles plus gros sur la carte.
6. Confort : file d'attente, export Hugging Face, installeur (4 j).


### Vérification juridique et quotas (19 novembre 2026)

Les quotas API sont dynamiques : l'application devrait lire/afficher le plafond effectif
et jamais coder en dur des promesses de « milliers gratuits ». À titre concret, le quota
Groq officiel pour `openai/gpt-oss-120b` est 200 000 tokens/jour : même à débit constant,
produire 25 M tokens (un repère bas pour Nano) prendrait au moins 125 jours ; en pratique,
les plafonds par minute, la taille des réponses et les erreurs rallongent ce délai. Les
API gratuites sont donc un appoint pour un corpus modeste, pas une fabrique rapide de
centaines de millions de tokens. Google indique que les
limites Gemini dépendent du modèle/projet et sont consultables dans AI Studio. Ses
conditions Gemini actuelles excluent l'usage pour développer des modèles concurrents et
réservent aux services payants les API clients distribués dans l'EEE ; elles indiquent
aussi l'usage du contenu des services gratuits pour améliorer ses produits. Sources :
[limites officielles Gemini](https://ai.google.dev/gemini-api/docs/rate-limits) ;
[conditions Gemini (en vigueur depuis le 23 mars 2026)](https://ai.google.dev/gemini-api/terms).
Pour Groq, les plafonds sont par organisation/modèle et les valeurs effectives se lisent
dans les réglages de compte : [documentation des limites](https://console.groq.com/docs/rate-limits).
OpenRouter précise aussi ses variantes gratuites, leurs limites dynamiques et le risque
de bascule payante : [limites API](https://openrouter.ai/docs/api_reference/limits) ;
[variantes gratuites](https://openrouter.ai/docs/guides/routing/model-variants/free).
