"""Checkpoint identity and evidence-backed task bindings; no metric targets."""
from pathlib import Path
from inference.core import sha256

ROOT = Path(__file__).resolve().parents[2]
MODELS = {
    'kitti-flow': {
        'file': 'checkpoints/main-results/flow/kitti15-main-flow.pth.tar',
        'sha256': '65ee6137a0c0d5b733b5b17167498649e19bb46f7933fdaaf441668005c8c57b',
        'flow': ['kitti2015', 'sintel', 'flyingthings3d', 'spring'],
        'depth': ['flyingthings3d'],
    },
    'kitti-depth': {
        'file': 'checkpoints/main-results/depth/kitti15-main-depth.pth.tar',
        'sha256': '36fe80abdb531296fcc2150aadfe80c5f308a253f8b992bb0de1253981542866',
        'flow': [], 'depth': ['kitti2015', 'eigen'],
    },
    'kitti12-candidate': {
        'file': 'checkpoints/main-results/flow/kitti12-main-flow.pth.tar',
        'sha256': '3a54a91b92cc211e1313734c27bfc6c29fe5d82b39163ac532c699f76d64b3c0',
        'flow': ['kitti2012'], 'depth': [],
    },
    'adapted-candidate': {
        'file': 'checkpoints/main-results/flow/flyingthings3d-spring-transfer-flow.pth.tar',
        'sha256': '8f24a4206b8add43d74d72d087664d95e2ef97ac0ff5fc2ae030ec03fc3bd791',
        'flow': ['flyingthings3d', 'spring'], 'depth': ['flyingthings3d', 'spring'],
    },
}
DEFAULTS = {('flow', 'kitti2015'): 'kitti-flow', ('flow', 'sintel'): 'kitti-flow',
            ('depth', 'kitti2015'): 'kitti-depth', ('depth', 'eigen'): 'kitti-depth'}


def validate(checkpoint, task, dataset, allow_candidate=False):
    path = Path(checkpoint)
    if not path.is_file():
        raise ValueError(f'Checkpoint not found: {path}; no alternative will be substituted.')
    digest = sha256(path)
    matches = [(name, item) for name, item in MODELS.items() if item['sha256'] == digest]
    if not matches:
        raise ValueError('Checkpoint checksum is not in the release mapping. A compatible filename is insufficient.')
    name, model = matches[0]
    if dataset not in model[task]:
        raise ValueError(f'Checkpoint {name} has no supported {task} evaluation binding for {dataset}.')
    needs_candidate = name.endswith('-candidate') or dataset in ('flyingthings3d', 'spring')
    if needs_candidate and not allow_candidate:
        raise ValueError('This combination is for candidate evaluation, not a claimed complete manuscript reproduction. '
                         'Read the guide and pass --allow-candidate explicitly to evaluate it.')
    return name, digest
