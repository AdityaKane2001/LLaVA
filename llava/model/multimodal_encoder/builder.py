import os
from torch.nn import ModuleList
from .clip_encoder import CLIPVisionTower
from .dino_encoder import DINOVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')


def build_multiple_vision_towers(vision_tower_cfg, **kwargs):
    # Returns a list of vision towers
    towers_pathlist = getattr(vision_tower_cfg, 'mm_multiple_vision_towers', getattr(vision_tower_cfg, 'multiple_vision_towers', None))
    
    vision_towers = list()
    for towerpath in towers_pathlist:
        is_absolute_path_exists = os.path.exists(towerpath)
        if (is_absolute_path_exists or towerpath.startswith("openai") or towerpath.startswith("laion") or "ShareGPT4V" in towerpath) and ("clip" in towerpath):
            vision_towers.append(CLIPVisionTower(towerpath, args=vision_tower_cfg, **kwargs))
        elif is_absolute_path_exists or ("dino" in towerpath):
            vision_towers.append(DINOVisionTower(towerpath, args=vision_tower_cfg, **kwargs))
        else:
            raise ValueError(f'Unknown vision tower: {towerpath}')
    return ModuleList(modules=vision_towers)

# def build_multiple_vision_towers_with_adapters(vision_tower_cfg, **kwargs):
#     # Returns a list of vision towers
#     towers_pathlist = getattr(vision_tower_cfg, 'mm_multiple_vision_towers', getattr(vision_tower_cfg, 'multiple_vision_towers', None))
    
#     vision_towers = list()
#     for towerpath in towers_pathlist:
#         is_absolute_path_exists = os.path.exists(towerpath)
#         if (is_absolute_path_exists or towerpath.startswith("openai") or towerpath.startswith("laion") or "ShareGPT4V" in towerpath) and ("clip" in towerpath):
#             vision_towers.append(
#                 nn.Sequential[CLIPVisionTower(towerpath, args=vision_tower_cfg, **kwargs)
#             )
#         elif is_absolute_path_exists or ("dino" in towerpath):
#             vision_towers.append(DINOVisionTower(towerpath, args=vision_tower_cfg, **kwargs))
#         else:
#             raise ValueError(f'Unknown vision tower: {towerpath}')
#     return ModuleList(modules=vision_towers)
    