# Genesis Studio

**Créez votre propre modèle de langage, de zéro.** Une application de bureau Windows
(PySide6/Qt 6, thème sombre) qui vous accompagne de l'import d'un corpus jusqu'à la
discussion avec **votre** modèle, entraîné sur **votre** machine.

Tout est fait maison : tokenizers (caractères, mots, BPE), GPT (attention, blocs
Transformer, échantillonnage), boucle d'entraînement avec checkpoints, reprise
d'entraînement et transfert de poids. Aucun appel réseau, aucune donnée envoyée.

![Écran des projets](captures/01_projets.png)

---

## 1. Démarrage rapide (Windows 11)

### Option A — lancer depuis les sources (recommandé pour commencer)

1. Installez **Python 3.10 ou plus récent** : <https://www.python.org/downloads/windows/>
   (cochez « Add python.exe to PATH » pendant l'installation).
2. Décompressez le dossier du projet où vous voulez.
3. Double-cliquez sur **`Installer et lancer.bat`**.
   Le script crée un environnement local `.venv`, installe les dépendances
   (PySide6, PyTorch, numpy) puis démarre l'application.
4. Les fois suivantes, utilisez **`Lancer.bat`** (démarrage immédiat).

> Installation manuelle équivalente :
> ```bat
> py -3 -m venv .venv
> .venv\Scripts\activate
> pip install -r requirements.txt
> python run.py
> ```

### GPU NVIDIA : installation spécifique

Double-cliquez sur **`Installer avec CUDA (GPU).bat`** (au lieu du script standard).
Il installe PyTorch compilé pour CUDA et affiche le résultat de
`torch.cuda.is_available()` à la fin. Vérifiez ensuite dans **Réglages** :
« Compilé avec CUDA : oui — CUDA 12.x » et le nom de votre carte.

### Option B — créer un exécutable autonome (.exe)

Double-cliquez sur **`Construire l'application (exe).bat`**.
À la fin, l'application se trouve dans `dist\Genesis Studio\Genesis Studio.exe`
(démarrage sans Python installé). Comptez 1,5 à 2,5 Go : c'est PyTorch.

Pour un exécutable plus léger, installez d'abord la version CPU de PyTorch :

```bat
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Mode portable

Créez un fichier vide nommé `portable.txt` à côté de `run.py` (ou de l'exe) :
les projets seront stockés dans un dossier `data\` voisin au lieu de
`%LOCALAPPDATA%\GenesisStudio`.

---

## 2. Mises à jour (build Windows)

Dans **Réglages → Mises à jour**, l'exécutable Windows vérifie les GitHub Releases du
projet, télécharge uniquement les fichiers modifiés et les valide par SHA-256. Après
confirmation, un petit updater indépendant de PyTorch ferme l'application, applique les
patches avec sauvegarde/rollback, puis relance Genesis Studio. Les projets, corpus,
réglages et checkpoints restent dans `%LOCALAPPDATA%\\GenesisStudio` (ou le dossier
portable) et ne sont jamais inclus dans les patches.

La toute première version diffusée doit être installée avec
`GenesisStudio-Windows-x64-full.zip`. Les versions ultérieures publient aussi un patch
incrémental. L'auto-update est réservé à l'exécutable Windows publié ; en lançant depuis
les sources, mettez le code à jour avec Git. Les releases sont publiées en poussant un tag
`vX.Y.Z` correspondant à `APP_VERSION` ; le workflow `.github/workflows/windows-release.yml`
construit et publie les assets. La première publication et le build Windows doivent être
validés avant de considérer ce mécanisme prêt pour les utilisateurs.

## 3. GPU / CUDA (support matériel)

Le GPU est pris en charge **de bout en bout**, pas seulement « détecté » :

| Aspect | Détail |
|---|---|
| Détection | `torch.cuda.is_available()`, nom de la carte, VRAM totale/libre, capacité de calcul, bfloat16, TF32 |
| Installation | `Installer avec CUDA (GPU).bat` (essaie cu126 → cu124 → cu121, vérifie `nvidia-smi`) ou `pip install -r requirements-cuda.txt` |
| Attention | PyTorch **SDPA** activé par défaut ; PyTorch choisit Flash/efficient/math selon la carte et la version. Le repli manuel est testé pour compatibilité. |
| Compilation | option avancée **torch.compile** (désactivée par défaut) ; première passe lente possible ; si le backend compile échoue, les mêmes fenêtres sont rejouées en eager avant tout pas d'optimiseur ; checkpoints restent portables |
| Entraînement sur GPU | tenseurs et modèle sur la carte, **précision mixte bfloat16/float16** (GradScaler si float16), **TF32** sur Ampere+ |
| Transferts | sur CUDA, lots préparés à l'avance dans un fil dédié + mémoire épinglée + `non_blocking` ; sur CPU, lecture synchrone (pas de fil inutile) |
| Mémoire | erreur « out of memory » interceptée : l'application réduit le batch de moitié et demande de relancer (elle ne reprend pas le pas échoué automatiquement) |
| Calibration | bouton **Réglages → Calibrer maintenant** : mesure le plus grand lot validé (test plafonné à 128) et le débit (tokens/s) d'un modèle Micro de référence ; l'onglet Entraînement propose ensuite un lot prudent redimensionné selon le gabarit/contexte courant |
| Chat | génération sur le même appareil que l'entraînement |

Sans GPU, tout fonctionne sur le CPU (aucune option à changer) ; l'option de
précision mixte est simplement ignorée et la carte est indiquée « indisponible ».

> Important : `pip install torch` **sans** `--index-url` installe une version **CPU**.
> C'est pour cela que le script CUDA existe. L'onglet Réglages affiche en clair
> « Compilé avec CUDA : oui/non » — si c'est « non », aucune accélération GPU n'est
> possible, quel que soit le matériel.

## 4. Dimensionnement : quelle taille de modèle pour quel corpus ?

L'application calcule en direct le ratio **tokens / paramètres** et fournit un filtre
d'alerte (carte « Volume de corpus / modèle » de l'onglet Entraînement, bouton
**Guide des tailles**). Il s'agit d'ordres de grandeur, pas d'une garantie : le ratio ne sait pas
si les documents se répètent, s'ils sont variés, ni si le modèle mémorise.

| Ratio tokens/paramètre | Interprétation indicative | Époques conseillées |
|---|---|---|
| < 1 | Critique : le modèle est plus gros que le corpus | 1 ; réduire le modèle/ajouter du texte |
| 1 – 20 | Risque élevé : volume de texte trop faible pour un GPT maison robuste | 1 ; ajouter des données |
| 20 – 100 | Plancher de départ ; surveiller la validation, pas de garantie | 1 |
| 100 – 300 | Repère prudent si les textes sont variés et dédupliqués | 1–2 |
| > 300 | Volume plus confortable ; la qualité et la validation restent décisives | 1 (ou tester un modèle plus grand) |

Le chiffre ~20 tokens/paramètre vient des lois d'échelle pour des entraînements de
grande taille (optimum calcul/données) ; il **ne signifie pas** « 20 suffit pour ne pas
mémoriser » sur un petit GPT. C'est pourquoi le tableau donne aussi un repère prudent
à ~100 tokens/paramètre. Le verdict pratique reste la courbe : arrêtez si la perte de
validation stagne/remonte alors que la perte d'entraînement continue de baisser.

Ordres de grandeur (repères indicatifs, BPE ~4 caractères/token, 1 Mo ≈ 250 000 tokens ; vocabulaire indicatif ~2 048) :

| Gabarit | Paramètres | Repère bas (~20 tok/param) | Repère prudent (~100 tok/param) |
|---|---|---|---|
| Nano | 238 k | 4,8 M tokens · ~19 Mo | 23,8 M tokens · ~95 Mo |
| Micro | 1,08 M | 21,6 M tokens · ~86 Mo | 108 M tokens · ~433 Mo |
| Mini | 5,31 M | 106 M tokens · ~424 Mo | 531 M tokens · ~2,1 Go |
| Small | 11,6 M | 232 M tokens · ~928 Mo | 1,16 Md tokens · ~4,6 Go |

Ces ordres de grandeur se calculent à partir des paramètres et dépendent du tokenizer ;
l'application les recalcule avec le modèle choisi. L'application compte les tokens totaux,
pas les passages uniques : la règle n'est pas une loi physique, et le volume en octets
n'est pas une mesure de diversité. Un corpus répété ne vaut pas un corpus varié. Le signe d'une mémorisation : la perte d'entraînement s'effondre alors
que la perte de validation stagne (les deux courbes sont affichées en direct).

## 5. Générer des textes pour le corpus

Dans **Corpus → Générer avec une IA**, vous pouvez créer des textes qui sont ajoutés
comme un fichier source normal, puis analyser le corpus et lancer l'entraînement.
Le générateur fonctionne hors du fil graphique, garde une sortie partielle en cas de
quota/arrêt, filtre les réponses trop courtes ou trop répétitives et déduplique les
textes générés au cours de la même session. Le volume est approximé à 4 caractères par
token : il ne remplace pas le tokenizer du projet.

Fournisseurs disponibles :

- **Ollama local** (choix par défaut) : gratuit, pas de clé ni d'envoi distant, à condition
d'avoir installé et démarré Ollama avec le modèle indiqué.
- **Groq** et **OpenRouter** : clé API requise, quotas/conditions du fournisseur à vérifier.
OpenRouter est verrouillé sur `openrouter/free` ou les modèles `:free` pour empêcher un
repli payant accidentel. Les clés ne sont jamais écrites dans un projet ; sous Windows,
la mémorisation est facultative et chiffrée par DPAPI. Aucun fournisseur distant ne reçoit
vos documents du corpus, seulement la consigne, le thème, le style et les réglages.

> Les sorties sont synthétiques, parfois incorrectes ou redondantes : inspectez le fichier,
> utilisez un jeu de validation indépendant et mélangez avec des sources que vous avez le
> droit d'utiliser. Les quotas gratuits ne sont pas une solution réaliste pour fabriquer
> rapidement des centaines de millions de tokens. Gemini n'est pas proposé : ses conditions
> actuelles ne conviennent pas à la génération d'un corpus pour entraîner un modèle
> concurrent. Voir [`AMELIORATIONS.md`](AMELIORATIONS.md) pour le détail.

## 6. Le cycle de travail

| Étape | Onglet | Ce qui se passe |
|---|---|---|
| 1 | **Corpus** | Vous glissez vos fichiers. Ils sont convertis en texte, copiés dans le projet, et analysés par le tokenizer. |
| 2 | **Corpus → Analyser** | Le vocabulaire est appris sur votre corpus, puis tout le texte est converti en un fichier de tokens lu en mémoire virtuelle. |
| 3 | **Entraînement** | Vous choisissez un gabarit (taille du réseau), les époques, et vous lancez. Chaque entraînement crée une **version**. |
| 4 | **Versions** | Vous comparez, vous discutez avec chaque version, vous en supprimez, vous continuez l'entraînement. |

### Formats acceptés (conversion automatique)

`TXT` · `Markdown` · `CSV` / `TSV` · `JSON` / `JSONL` · `HTML` / `XML` · `EPUB` ·
`LaTeX` · `SVG` · journaux — et, si les paquets facultatifs sont installés,
`DOCX` (`pip install python-docx`) et `PDF` (`pip install pypdf`).
Les fichiers sont lus par blocs de 1 Mio : un corpus de 10 Go ne sature pas la RAM.

Un **corpus d'exemple** (poèmes, contes, paires instruction/réponse) est inclus :
bouton « Corpus d'exemple » dans l'onglet Corpus.

---

## 7. Les fonctions demandées, en détail

### Projets et versions
* Chaque **projet** rassemble un corpus, un tokenizer et **toutes les versions** du modèle.
* Chaque **entraînement** crée une version nommée `v1`, `v2`, … avec sa courbe de perte,
  sa perte de validation, ses pas, ses tokens vus, sa durée et son dossier.
* **Suppression d'une version** : bouton `⋯ → Supprimer cette version…`, avec une boîte
  de confirmation qui exige de cocher « Je confirme la suppression définitive ».
  Le nombre et le poids des checkpoints concernés sont affichés.

### Ne jamais repartir de zéro
Deux mécanismes, dans l'onglet **Versions** :

* **Continuer** → ouvre l'onglet Entraînement avec la version choisie comme point de
  départ. La nouvelle version **hérite des poids** (transfert), garde la même
  architecture (verrouillée, car sinon les poids ne seraient plus compatibles) et
  **cumule** les tokens vus. Aucun checkpoint n'est dupliqué : la version pointe vers
  les poids du parent jusqu'à son premier enregistrement.
* **Reprendre** (versions « interrompue ») → restaure **exactement** le pas, l'état de
  l'optimiseur, la perte et l'historique depuis `last.pt`, puis poursuit.

L'onglet Entraînement présélectionne d'ailleurs la dernière version entraînée : par
défaut, on continue son modèle plutôt que d'en recréer un.

### Corpus volumineux sans saturer la RAM
1. Les fichiers sont lus **en flux** (blocs de 1 Mio, conversions incluses).
   L'apprentissage du tokenizer BPE dure de quelques secondes à ~1 minute selon la
   taille et la diversité du corpus (mesuré : 20 s pour 1 024 tokens de vocabulaire,
   40 s pour 2 048, 73 s pour 4 096 sur un corpus de 4,5 Mo) ; une barre de progression
   l'accompagne.
2. La tokenisation écrit directement `tokens.bin` (uint16/uint32) sur le disque.
3. L'entraînement lit ce fichier via `numpy.memmap` : seules les fenêtres tirées au
   hasard sont chargées en mémoire. La RAM utilisée ne dépend donc ni de la taille du
   corpus ni du nombre d'époques.

### Choix du nombre de paramètres, avec explications
Cinq gabarits — **Nano**, **Micro**, **Mini**, **Small**, **Personnalisé** — avec, pour
chacun : le nombre exact de paramètres pour *votre* vocabulaire, la mémoire disque
estimée, la vitesse, et une description de ce qu'on peut en attendre. Le bouton
**« Voir l'architecture »** affiche la configuration effective et le code réellement
instancié (`ModelConfig(...) → GPT(cfg)`, blocs, têtes, `head_dim`, etc.).

En mode **Personnalisé** : couches, têtes d'attention, largeur `d_model`, longueur de
contexte. La validation (`largeur % têtes == 0`) est vérifiée en direct.

### Discuter avec le modèle
Chaque version entraînée a son écran de discussion : réponses **en flux**, trois
gabarits de prompt (continuation libre, Instruction/Réponse, Dialogue), réglages
température / top-k / top-p / tokens max / anti-répétition, et l'option d'inclure ou non
les échanges précédents dans le contexte. Les réponses sont copiables.

> Un modèle peu entraîné produit des suites de caractères : c'est normal. Le panneau
> « À propos de ce modèle » rappelle le nombre de pas et de tokens vus.

---

## 8. Réglages d'entraînement

| Réglage | Effet |
|---|---|
| Époques | Nombre de passages sur le corpus (un petit corpus demande plusieurs époques). |
| Taille de lot | Fenêtres traitées ensemble : plus stable, plus gourmand. |
| Longueur de contexte | Tokens vus d'un coup (≤ contexte du gabarit). |
| Taux d'apprentissage | 1e-3 pour Nano/Micro, 3e-4 → 1e-4 pour Mini/Small. |
| Échauffement, accumulation de gradient, coupure de gradient, weight decay | Réglages fins dans « Paramètres avancés ». |
| Checkpoint tous les N pas | Sauvegarde fréquente : une coupure ne fait jamais perdre l'entraînement. |
| Fils CPU / appareil | 0 = automatique ; GPU (CUDA) si disponible. |

Le panneau de droite affiche les pas par époque, la durée estimée (puis mesurée), la
taille des poids, la courbe de perte en direct, et un journal détaillé.

---

## 9. Où sont mes données ?

```
%LOCALAPPDATA%\GenesisStudio\
├── projects\<projet>\project.json      # projet, corpus, versions
│   ├── corpus\                        # fichiers importés (copies)
│   ├── tokenizer\tokenizer.json       # vocabulaire appris
│   ├── tokens.bin                     # corpus encodé (lu en mémoire virtuelle)
│   └── versions\<version>\
│       ├── checkpoints\last.pt, best.pt
│       └── entrainement.log
├── corbeille\                         # projets supprimés (récupérables à la main)
└── settings.json
```

Sauvegarder = copier ce dossier. `GENESIS_HOME` permet de le déplacer.

---

## 10. Structure du code

```
genesis/
├── app.py                  # point d'entrée Qt
├── core/
│   ├── paths.py            # emplacements, tailles lisibles
│   ├── tokenizers.py       # Char / Word / BPE (entraînement + encodage maison)
│   ├── corpus.py           # import, conversion, lecture en flux, tokens.bin
│   ├── model.py            # GPT (attention causale, blocs, échantillonnage)
│   ├── gpu.py              # détection CUDA, précision mixte, calibration
│   ├── sizing.py           # ratio tokens/paramètres, risque de mémorisation
│   ├── generators/         # clients Groq/OpenRouter/Ollama, prompts, filtres, dédup
│   │   ├── client.py       # HTTP + limites de reprise
│   │   ├── service.py      # génération bornée, sortie partielle
│   │   ├── dedup.py        # similitude de n-grammes
│   │   ├── quality.py      # contrôle minimal des textes
│   │   └── secrets.py      # DPAPI (Windows), mémoire seule ailleurs
│   ├── trainer.py          # boucle d'entraînement, checkpoints, reprise
│   ├── session.py          # orchestration (tokenisation, versions, transfert)
│   ├── chat.py             # prompts + génération en flux
│   └── project.py          # modèle de données projet/version
└── ui/
    ├── theme.py            # palette, feuille de style, icônes SVG
    ├── widgets.py          # cartes, chips, drop zone, graphiques, toasts
    ├── dialogs.py          # nouvelles versions, confirmations, détails
    ├── workers.py          # threads Qt (import, tokenisation, entraînement, génération)
    ├── main_window.py      # fenêtre + barre latérale
    └── pages/              # projets, corpus, versions, entraînement, discussion
tests/
├── test_engine.py          # moteur complet (3 tokenizers, reprise, transfert, génération)
└── test_ui.py              # captures de chaque écran (plateforme offscreen)
tools/make_icon.py          # génère assets/genesis.ico
```

### Tests

```bat
.venv\Scripts\python tests\test_engine.py
.venv\Scripts\python tests\test_optimizations.py
.venv\Scripts\python tests\test_generator.py
.venv\Scripts\python tests\test_generator_ui.py
.venv\Scripts\python tests\test_ui.py          :: écrit captures\*.png
```

---

## 11. Ce qui a été vérifié

Tests exécutés (Linux, Qt en mode « offscreen », même code que sous Windows) :

| Test | Contenu | Résultat |
|---|---|---|
| `tests/test_optimizations.py` | SDPA vs attention manuelle (sorties + causalité), simulation d'échec Inductor/Triton et repli eager sans checkpoint wrapper | ✅ |
| `tests/test_generator.py` | prompts, filtre de qualité, déduplication, garde-fou contre OpenRouter payant, payload Ollama local, retry borné, sortie partielle après quota | ✅ |
| `tests/test_generator_ui.py` | UI du générateur, consentement distant, garde-fous OpenRouter et import de la sortie (sans réseau) | ✅ |
| `tests/test_engine.py` | 3 tokenizers, import/conversion, tokenisation, entraînement, transfert de poids vers une v2, reprise exacte d'une version interrompue, génération, suppression de version | ✅ |
| `tests/test_ui_flows.py` | création de projet, glisser-déposer simulé, analyse, entraînement lancé depuis l'onglet (threads Qt), « Continuer » avec héritage, discussion, suppression confirmée, rechargement depuis le disque | ✅ |
| `tests/test_ui.py` | captures de tous les écrans (`captures/*.png`) | ✅ |
| `tests/test_dialogs.py` | état vide, création de projet, confirmation de suppression, détails de version, **Réglages/GPU** (`11_reglages_gpu.png`), **générateur** (`13_generateur_corpus.png`), **guide des tailles** (`12_guide_tailles.png`) | ✅ |
| `tests/test_gpu_sizing.py` | rapport matériel, benchmark CPU borné, ratios et tableau de dimensionnement | ✅ |
| Vérification ponctuelle (module `gpu`) | détection matérielle sur CPU, `run_benchmark` de bout en bout (lot CPU validé, débit en tokens/s), calculs `sizing.advise` sur plusieurs volumes | ✅ |

Vérifications ponctuelles supplémentaires : ouverture du dialogue **Réglages** depuis
la fenêtre principale, sans et avec un projet ouvert (mise à jour du conseil de
calibration), calibration CPU exécutée depuis l'interface et sauvegardée dans les
réglages, dialogue et garde-fous du générateur, et démarrage de l'application complète
(`run.py`) sans erreur. Les appels réels à Groq/OpenRouter/Ollama et le chiffrement DPAPI
Windows n'ont pas été testés ici (pas de clé, pas de serveur Ollama ni de Windows).

**Non exécuté ici** (environnement Linux, pas de GPU NVIDIA) : l'entraînement réel sur
CUDA (précision mixte bfloat16/float16, TF32, transferts épinglés) — le chemin de code
est intégré mais doit être validé sur une carte NVIDIA ; le script `Installer avec CUDA
(GPU).bat` ; la barre de titre sombre Windows
(`DwmSetWindowAttribute`), les scripts `.bat` et la compilation PyInstaller
(`genesis.spec`). Ils sont fournis et écrits pour Windows 11, mais nécessitent une
exécution sur votre machine pour être validés — dites-moi s'il y a le moindre
accroc, je corrige.

Les suggestions de nouvelles fonctionnalités et l'analyse des générateurs de corpus IA
(quota gratuit, conditions d'utilisation, mode local) sont détaillées dans
[`AMELIORATIONS.md`](AMELIORATIONS.md).

## 12. Limites connues & feuille de route

* **Pas de cache KV** dans la génération : chaque token re-calcule le contexte. C'est
  simple et fiable ; pour de gros gabarits, l'ajout d'un cache accélérerait
  nettement la discussion.
* **Architecture figée** quand on hérite des poids (par construction : les matrices
  doivent correspondre). Pour changer de gabarit, on repart d'un modèle neuf.
* **Pas encore** : export/import d'un modèle hors de l'application, reprise d'un
  entraînement après changement de tokenizer (volontairement refusé), fine-tuning
  supervisé, quantification, évaluation automatique de la qualité.
* L'estimation de durée avant lancement est approximative ; la valeur mesurée la
  remplace dès les premiers pas.
* Formats `DOCX`/`PDF` : dépendances optionnelles (non incluses dans l'exe par défaut).

Les points d'extension naturels sont déjà en place : `core/session.py` centralise
l'orchestration, `core/model.py` isole l'architecture, et `ui/workers.py` sépare
proprement le calcul de l'interface.
