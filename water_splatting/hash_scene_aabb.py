"""Ablation-selected hash_aabb_scale per scene (Phase 1e protocol)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class SceneAabbLock:
    hash_aabb_scale: float
    eval_run_id: str
    eval_json: str
    image_cache: str = "cpu"


# Locked after full aabb ablations (2026-06-28 batch unless noted).
SCENE_AABB: Mapping[str, SceneAabbLock] = {
    "panama": SceneAabbLock(44.7, "2026-06-28_092302", "outputs/eval/panama-aabb-ablation-44.7.json"),
    "jg": SceneAabbLock(52.0, "2026-06-28_042127", "outputs/eval/jg-aabb-ablation-52.json"),
    "kwaj": SceneAabbLock(35.0, "2026-06-28_130758", "outputs/eval/kwaj-aabb-ablation-35.json"),
    "isro": SceneAabbLock(25.0, "2026-06-28_150638", "outputs/eval/isro-aabb-ablation-25.json"),
    "iui3": SceneAabbLock(45.0, "2026-06-27_072109", "outputs/eval/iui3-aabb45.0.json"),
    "curacao": SceneAabbLock(36.0, "2026-06-28_182532", "outputs/eval/curacao-aabb-ablation-36.json"),
}

# Dataset folder names under data/undistorted/ (may differ from scene keys).
SCENE_DATA_DIRS: Mapping[str, str] = {
    "panama": "Panama",
    "jg": "JapaneseGradens-RedSea",
    "kwaj": "Kwajalein",
    "isro": "Isro",
    "iui3": "IUI3",
    "curacao": "Curasao",
}


def get_aabb(scene: str) -> float:
    return SCENE_AABB[scene.lower()].hash_aabb_scale
