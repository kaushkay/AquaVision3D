from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig

from water_splatting.hash_water_splatting import HashWaterSplattingModelConfig

MAX_TRAIN_STEPS = 15000


def _exp_decay(lr_final: float, warmup_steps: int = 0) -> ExponentialDecaySchedulerConfig:
    return ExponentialDecaySchedulerConfig(
        lr_final=lr_final,
        max_steps=MAX_TRAIN_STEPS,
        warmup_steps=warmup_steps,
    )


def _default_optimizers() -> dict:
    return {
        "means": {"optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15), "scheduler": _exp_decay(lr_final=5e-05)},
        "features_dc": {"optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15), "scheduler": _exp_decay(lr_final=0.0025)},
        "features_rest": {"optimizer": AdamOptimizerConfig(lr=0.000125, eps=1e-15), "scheduler": _exp_decay(lr_final=0.000125)},
        "opacities": {"optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15), "scheduler": _exp_decay(lr_final=0.05)},
        "scales": {"optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15), "scheduler": _exp_decay(lr_final=0.005)},
        "quats": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": _exp_decay(lr_final=0.001)},
        "camera_opt": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": _exp_decay(lr_final=5e-05)},
        "hash_medium": {
            "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15, max_norm=0.001),
            "scheduler": _exp_decay(lr_final=0.00015, warmup_steps=500),
        },
    }


hash_water_splatting_method = MethodSpecification(
    config=TrainerConfig(
        method_name="hash-water-splatting",
        steps_per_eval_image=1000,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=MAX_TRAIN_STEPS,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=FullImageDatamanagerConfig(
                dataparser=NerfstudioDataParserConfig(load_3D_points=True),
            ),
            model=HashWaterSplattingModelConfig(
                num_steps=MAX_TRAIN_STEPS,
                main_loss="l1",
                ssim_loss="reg_ssim",
                zero_medium=False,
            ),
        ),
        optimizers=_default_optimizers(),
        viewer=ViewerConfig(num_rays_per_chunk=32768),
        vis="viewer",
    ),
    description="Hash-water-splatting: 3DGS with spatial hash-grid medium.",
)

hash_water_splatting_method_big = MethodSpecification(
    config=TrainerConfig(
        method_name="hash-water-splatting-big",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=MAX_TRAIN_STEPS,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=FullImageDatamanagerConfig(
                dataparser=NerfstudioDataParserConfig(load_3D_points=True),
            ),
            model=HashWaterSplattingModelConfig(
                num_steps=MAX_TRAIN_STEPS,
                continue_cull_post_densification=False,
            ),
        ),
        optimizers={
            **_default_optimizers(),
            **{
                "features_dc": {"optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15), "scheduler": None},
                "features_rest": {"optimizer": AdamOptimizerConfig(lr=0.000125, eps=1e-15), "scheduler": None},
                "opacities": {"optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15), "scheduler": None},
                "scales": {"optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15), "scheduler": None},
                "quats": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None},
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=32768),
        vis="viewer",
    ),
    description="Hash-water-splatting (larger capacity) with spatial hash-grid medium.",
)
