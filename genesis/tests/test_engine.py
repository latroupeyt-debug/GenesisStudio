"""Test bout-en-bout du moteur (sans interface) : corpus -> tokens -> entraînement
-> reprise -> génération. À lancer avec :  python tests/test_engine.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TMP = Path(tempfile.mkdtemp(prefix="genesis-test-"))
os.environ["GENESIS_HOME"] = str(TMP / "home")

from genesis.core import chat, corpus, session  # noqa: E402
from genesis.core.project import ProjectStore  # noqa: E402
from genesis.core.tokenizers import SPECIAL  # noqa: E402
from genesis.core.trainer import TrainConfig, TrainHooks  # noqa: E402

SAMPLE = """Le petit renard traverse la forêt silencieuse.
Il cherche un abri pour la nuit, sous les branches basses.
La lune éclaire les chemins couverts de feuilles mortes.
Demain, le soleil se lèvera sur les collines vertes.
Le petit renard court vers la rivière et boit lentement.
"""


def make_corpus(path: Path, repeats: int = 400) -> None:
    path.write_text("\n".join([SAMPLE] * repeats), encoding="utf-8")


def main() -> int:
    store = ProjectStore()
    for kind, opts in (("char", {}), ("word", {"max_vocab": 300}), ("bpe", {"vocab_size": 320})):
        print(f"\n=== tokenizer {kind} ===")
        project = store.create(f"Projet {kind}", "test", kind)
        project.data["tokenizer"]["options"] = opts
        project.save()

        src = TMP / f"corpus_{kind}.txt"
        make_corpus(src)
        imported, errors = corpus.import_paths(project.corpus_dir, [src])
        assert not errors, errors
        project.add_corpus_entries(imported)

        stats = session.tokenize_project(project, progress=lambda f, m: None)
        print("tokens:", stats["tokens"], "vocab:", stats["vocab_size"], "dtype:", stats["dtype"])
        assert stats["tokens"] > 100

        tok = project.tokenizer()
        sample = tok.decode(tok.encode("Le petit renard " + SPECIAL + " test"))
        assert "renard" in sample, sample

        cfg = TrainConfig(epochs=1, batch_size=8, seq_len=32, lr=6e-3,
                          eval_every=10, ckpt_every=20, max_steps=30, warmup=5)
        version, message, transferred = session.build_version(
            project, preset_key="nano", train_cfg=cfg, device_pref="cpu",
        )
        print("version:", version.name, message)

        logs: list[str] = []
        hooks = TrainHooks(on_log=lambda m: logs.append(m))
        trainer = session.make_trainer(project, version, hooks, cfg)
        summary = trainer.run()
        version = session.persist_run(project, version.id, summary, cfg.to_dict())
        print("après v1 :", version.steps, "pas, loss", version.last_loss, "statut", version.status)
        assert version.steps == 30

        # --- 2e version : transfert de poids, on ne repart pas de zéro ---
        cfg2 = TrainConfig(epochs=1, batch_size=8, seq_len=32, lr=3e-3,
                           eval_every=10, ckpt_every=20, max_steps=12, warmup=2)
        v2, msg2, transferred2 = session.build_version(
            project, preset_key="nano", train_cfg=cfg2,
            base_version_id=version.id, device_pref="cpu",
        )
        print("v2:", msg2, "| transfert:", transferred2)
        assert transferred2, msg2
        t2 = session.make_trainer(project, v2, hooks, cfg2)
        s2 = t2.run()
        v2 = session.persist_run(project, v2.id, s2, cfg2.to_dict())
        print("après v2 :", v2.steps, "pas, tokens cumulés", v2.tokens_seen)
        assert v2.tokens_seen > v2.steps * 8 * 32  # héritage des tokens vus

        # --- reprise exacte d'une version interrompue ---
        v3, _, _ = session.build_version(project, preset_key="nano", train_cfg=cfg,
                                         device_pref="cpu")
        t3 = session.make_trainer(project, v3, hooks, cfg)
        t3.cfg.max_steps = 6
        t3.total_steps = 6
        s3 = t3.run()
        v3 = session.persist_run(project, v3.id, s3, cfg.to_dict())
        t4 = session.make_trainer(project, v3, TrainHooks(), cfg, resume_exact=True)
        assert t4.start_step == 6, (t4.start_step, v3.steps)
        t4.cfg.max_steps = 10
        t4.total_steps = 10
        s4 = t4.run()
        v3 = session.persist_run(project, v3.id, s4, cfg.to_dict())
        print("reprise ok :", v3.steps, "pas au total")

        # --- discussion ---
        model, tokenizer, device, meta = session.load_version_model(project, v2)
        pieces: list[str] = []
        text = chat.stream_generate(
            model, tokenizer, chat.build_prompt("continuation", "Le petit renard"),
            chat.SamplingParams(temperature=0.9, max_new_tokens=40, top_k=20),
            on_text=pieces.append, device=device,
        )
        print("génération:", repr(text[:120]))
        assert isinstance(text, str) and len(text) > 0

        # --- suppression d'une version ---
        before = len(project.versions)
        project.delete_version(v3.id)
        assert len(project.versions) == before - 1
        assert not project.version_dir(v3.id).exists()

    print("\nTOUS LES TESTS DU MOTEUR SONT PASSÉS")
    print("Dossier de test :", TMP)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
