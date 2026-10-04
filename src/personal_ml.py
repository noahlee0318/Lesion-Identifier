"""Local, per-person experimental lesion-mask CNN. No network or model downloads.

Train on manually reviewed discs, select threshold on validation days, evaluate
once on held-out test days. Calibration is scale measurement, not training.
This small whole-image baseline is not the planned production tiled detector.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from scipy.ndimage import label as components
from scipy.optimize import linear_sum_assignment

SIZE = 256
POSES = ('frontal', 'left60', 'right60')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def load_dataset(path):
    """Validate original bytes and annotations; reject duplicates across dates."""
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if data.get('schema') != 1 or not isinstance(data.get('profile_id'), str) or not data['profile_id']:
        raise ValueError('Use a version 1 personal-workspace export.')
    if not isinstance(data.get('images'), list) or not data['images']:
        raise ValueError('Add photos before exporting.')
    ids, hashes = set(), set()
    for item in data['images']:
        if item.get('profile_id') != data['profile_id']:
            raise ValueError('Photos from different profiles cannot be combined.')
        if item['id'] in ids or item['pose'] not in POSES or item['kind'] not in ('calibration', 'session'):
            raise ValueError('Invalid or duplicate photo record.')
        ids.add(item['id'])
        date.fromisoformat(item['date'])
        raw = base64.b64decode(item['bytes'], validate=True)
        sha = hashlib.sha256(raw).hexdigest()
        if sha != item['sha256'] or sha in hashes:
            raise ValueError('Photo bytes changed or the same photo was included twice.')
        hashes.add(sha)
        with Image.open(io.BytesIO(raw)) as original:
            image = ImageOps.exif_transpose(original)
            if list(image.size) != [item['width'], item['height']]:
                raise ValueError('Annotation dimensions do not match the oriented photo.')
        scale = item.get('px_per_mm')
        if scale is not None and (isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0):
            raise ValueError('Photo scale must be a positive finite number.')
        if type(item.get('reviewed')) is not bool or not isinstance(item.get('labels'), list):
            raise ValueError('Every photo needs an explicit annotation review state.')
        for spot in item['labels']:
            if not all(isinstance(spot.get(k), (float, int)) and math.isfinite(spot[k]) for k in ('x', 'y', 'radius')):
                raise ValueError('Spot coordinates must be finite numbers.')
            if not (0 <= spot['x'] < item['width'] and 0 <= spot['y'] < item['height'] and 0 < spot['radius'] <= max(image.size)):
                raise ValueError('Spot coordinates are outside the photo.')
    return data


def split_days(records, val_days=14, test_days=14, gap=3):
    """Fixed trailing calendar windows; gaps excluded. Never random image splits."""
    if min(val_days, test_days) < 1 or gap < 0:
        raise ValueError('Invalid split window.')
    days = sorted({date.fromisoformat(r['date']) for r in records})
    if not days:
        raise ValueError('No reviewed, scale-calibrated session photos for this pose.')
    test_start = days[-1] - timedelta(days=test_days - 1)
    val_end = test_start - timedelta(days=gap + 1)
    val_start = val_end - timedelta(days=val_days - 1)
    train_end = val_start - timedelta(days=gap + 1)
    groups = {'train': [], 'val': [], 'test': [], 'gap': []}
    for record in records:
        day = date.fromisoformat(record['date'])
        key = 'train' if day <= train_end else 'val' if val_start <= day <= val_end else 'test' if day >= test_start else 'gap'
        groups[key].append(record)
    if any(not groups[k] for k in ('train', 'val', 'test')):
        raise ValueError('Keep collecting and reviewing photos: the default split needs about 35 calendar days, including photos in train, validation, and test windows.')
    return groups


def eligible(data, pose):
    return [r for r in data['images'] if r['kind'] == 'session' and r['pose'] == pose and r['reviewed'] and r.get('px_per_mm')]


def tensors(record):
    with Image.open(io.BytesIO(base64.b64decode(record['bytes']))) as original:
        image = ImageOps.exif_transpose(original).convert('RGB').resize((SIZE, SIZE))
        x = torch.tensor(np.asarray(image).copy(), dtype=torch.float32).permute(2, 0, 1) / 255
    yy, xx = np.mgrid[:SIZE, :SIZE]
    xx = (xx + .5) * record['width'] / SIZE
    yy = (yy + .5) * record['height'] / SIZE
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    for spot in record['labels']:
        mask |= (xx - spot['x']) ** 2 + (yy - spot['y']) ** 2 <= spot['radius'] ** 2
    return x.unsqueeze(0), torch.tensor(mask, dtype=torch.float32)[None, None]


class LesionMaskNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(16, 1, 1))

    def forward(self, x):
        return self.layers(x)


def centers(probability, record, threshold):
    regions, count = components(probability >= threshold)
    points = []
    for index in range(1, count + 1):
        yy, xx = np.where(regions == index)
        if len(xx) >= 2:
            points.append([(float(xx.mean()) + .5) * record['width'] / SIZE,
                           (float(yy.mean()) + .5) * record['height'] / SIZE])
    return points


def counts(predicted, truth, scale, tolerance_mm=2):
    if not predicted or not truth:
        return 0, len(predicted), len(truth)
    distances = np.linalg.norm(np.asarray(predicted)[:, None] - np.asarray(truth)[None], axis=2) / scale
    valid = distances <= tolerance_mm
    # Invalid edges cost more than all valid distances combined: maximum matching.
    cost = np.where(valid, distances / (tolerance_mm * (min(distances.shape) + 1)), 2)
    rows, cols = linear_sum_assignment(cost)
    tp = int(valid[rows, cols].sum())
    return tp, len(predicted) - tp, len(truth) - tp


def metrics(rows):
    tp, fp, fn = np.asarray(rows, dtype=int).sum(axis=0).tolist()
    return {'tp': tp, 'fp': fp, 'fn': fn, 'precision': tp / (tp + fp) if tp + fp else None,
            'recall': tp / (tp + fn) if tp + fn else None,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0}


def probabilities(model, records, device):
    model.eval()
    with torch.no_grad():
        return [torch.sigmoid(model(tensors(r)[0].to(device)))[0, 0].cpu().numpy() for r in records]


def score(records, probs, threshold):
    rows = [counts(centers(p, r, threshold), [[s['x'], s['y']] for s in r['labels']], r['px_per_mm']) for r, p in zip(records, probs)]
    return metrics(rows), rows


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as file:
        json.dump(value, file, indent=2, allow_nan=False)


def train(data, output, pose='frontal', epochs=10, device='cuda'):
    if epochs < 1:
        raise ValueError('Train for at least one epoch.')
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable. Choose --device cpu explicitly for a slower run.')
    groups = split_days(eligible(data, pose))
    for part in ('train', 'val'):
        if not any(r['labels'] for r in groups[part]):
            raise ValueError(f'{part} needs reviewed positive examples as well as background.')
    output = Path(output)
    if output.resolve().is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError('Keep personal model runs outside the code repository, in a private local folder.')
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(17)
    model = LesionMaskNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    losses = []
    for _ in range(epochs):
        model.train()
        total = 0
        for record in groups['train']:
            x, y = tensors(record)
            if record['labels'] and not y.any():
                raise ValueError('A labeled lesion disappeared at baseline resolution. Use closer photos or the future tiled detector.')
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(model(x), y, pos_weight=torch.tensor(10., device=device))
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
        losses.append(total / len(groups['train']))
    probs = probabilities(model, groups['val'], device)
    candidates = [(t, score(groups['val'], probs, t)[0]) for t in (.3, .5, .7)]
    threshold, val = max(candidates, key=lambda pair: pair[1]['f1'])
    torch.save(model.cpu().state_dict(), output / 'weights.pt')
    card = {'schema': 1, 'architecture': 'lesion-mask-cnn-v1', 'profile_id': data['profile_id'],
            'pose': pose, 'dataset_hash': digest(data), 'weights_hash': hashlib.sha256((output / 'weights.pt').read_bytes()).hexdigest(),
            'threshold': threshold, 'seed': 17, 'epochs': epochs, 'device': device, 'losses': losses,
            'splits': {k: [r['id'] for r in records] for k, records in groups.items()},
            'day_counts': {k: len({r['date'] for r in records}) for k, records in groups.items()},
            'validation': val, 'status': 'trained_not_evaluated',
            'limitations': 'Experimental 256px mask baseline; small lesions may disappear and adjacent lesions may merge. No tracking, diagnosis, cross-pose totals, or browser inference.'}
    write_json(output / 'model.json', card)
    return card


def load_model(folder):
    folder = Path(folder)
    card = json.loads((folder / 'model.json').read_text())
    if card['architecture'] != 'lesion-mask-cnn-v1' or hashlib.sha256((folder / 'weights.pt').read_bytes()).hexdigest() != card['weights_hash']:
        raise ValueError('Model metadata or weights do not match.')
    model = LesionMaskNet()
    model.load_state_dict(torch.load(folder / 'weights.pt', map_location='cpu', weights_only=True))
    return model, card


def evaluate(data, folder):
    model, card = load_model(folder)
    if digest(data) != card['dataset_hash'] or data['profile_id'] != card['profile_id']:
        raise ValueError('Evaluation requires the exact frozen dataset used for training.')
    groups = split_days(eligible(data, card['pose']))
    if {k: [r['id'] for r in v] for k, v in groups.items()} != card['splits']:
        raise ValueError('Split metadata changed.')
    records = groups['test']
    result, rows = score(records, probabilities(model, records, 'cpu'), card['threshold'])
    enough = len({r['date'] for r in records}) >= 5 and result['tp'] + result['fn'] >= 20
    passed = enough and (result['precision'] or 0) > .8 and (result['recall'] or 0) > .7
    report = {'profile_id': data['profile_id'], 'pose': card['pose'], 'dataset_hash': card['dataset_hash'],
              'metrics': result, 'test_images': len(records), 'test_days': len({r['date'] for r in records}),
              'status': 'provisional_gate_passed' if passed else 'gate_failed' if enough else 'insufficient_evidence',
              'gate': 'Precision > 0.8 and recall > 0.7; at least 20 labeled test lesions over 5 days. Provisional engineering gate, not clinical validation.',
              'matching_tolerance_mm': 2, 'per_image': [{'id': r['id'], 'tp': c[0], 'fp': c[1], 'fn': c[2]} for r, c in zip(records, rows)],
              'warning': 'Do not tune on this report. Repeated test-set selection is optimistic; reserve new days for future model comparisons.'}
    write_json(Path(folder) / 'evaluation.json', report)
    return report


def predict(data, folder, image_id):
    """Explicitly experimental per-image inference; never adds to daily counts."""
    model, card = load_model(folder)
    if data['profile_id'] != card['profile_id']:
        raise ValueError('This model belongs to another profile.')
    record = next((r for r in data['images'] if r['id'] == image_id), None)
    if record is None or record['pose'] != card['pose'] or not record.get('px_per_mm'):
        raise ValueError('Choose a scale-calibrated photo in the model view.')
    points = centers(probabilities(model, [record], 'cpu')[0], record, card['threshold'])
    return {'experimental': True, 'profile_id': data['profile_id'], 'image_id': image_id,
            'candidate_count': len(points), 'centers_mm': [[x / record['px_per_mm'], y / record['px_per_mm']] for x, y in points],
            'warning': 'Unconfirmed model candidates, not a diagnosis or validated daily count. Coordinates are relative to this photo, not a registered atlas.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('inspect', 'train', 'evaluate', 'predict'):
        cmd = sub.add_parser(name)
        cmd.add_argument('dataset', type=Path)
        if name != 'inspect':
            cmd.add_argument('run', type=Path)
        if name == 'train':
            cmd.add_argument('--pose', choices=POSES, default='frontal')
            cmd.add_argument('--epochs', type=int, default=10)
            cmd.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
        if name == 'predict':
            cmd.add_argument('--image-id', required=True)
    args = parser.parse_args()
    try:
        data = load_dataset(args.dataset)
        if args.command == 'inspect':
            result = {'profile_id': data['profile_id'], 'images': len(data['images']), 'eligible_by_pose': {p: len(eligible(data, p)) for p in POSES}}
        elif args.command == 'train':
            result = train(data, args.run, args.pose, args.epochs, args.device)
        elif args.command == 'evaluate':
            result = evaluate(data, args.run)
        else:
            result = predict(data, args.run, args.image_id)
        print(json.dumps(result, indent=2))
    except (ValueError, KeyError, FileExistsError) as error:
        parser.exit(2, f'{error}\n')


if __name__ == '__main__':
    main()
