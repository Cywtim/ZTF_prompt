#!/usr/bin/env python3
"""Batch rebuild all PRF TDE sources with v2 adapter + re-classify."""
import subprocess, os, csv, json, numpy as np
from pathlib import Path

os.chdir(Path(__file__).parent)

# Load coordinates
coord_map = {}
with open('../ZTF_TDE/lasair/TNS+vVReview_TDEs_Copy_watchlist_results.csv') as f:
    for row in csv.DictReader(f):
        ztf = row.get('objectId', '').strip()
        if ztf:
            coord_map[ztf] = (float(row['ramean']), float(row['decmean']))

flux_dir = Path('../ZTF_TDE/data/TS/Flux/TDE/')
available = {f.stem.replace('_difference_photometry_flux', '') 
             for f in flux_dir.glob('*.npy') if 'irsa_full' not in f.name}

training = [
    'ZTF20achpcvt','ZTF20abnorit','ZTF19aapreis','ZTF19aakiwze',
    'ZTF19aakswrb','ZTF22abkfhua','ZTF20abjwvae',
    'ZTF21acafvhf','ZTF22aaaedas','ZTF20abowque',
    'ZTF22aaabovl','ZTF19abhhjcc','ZTF20abfcszi','ZTF19aabbnzo',
    'ZTF21abcgnqn','ZTF20acnznms','ZTF22aaahtqz',
    'ZTF18abxftqm','ZTF21abxngcz',
    'ZTF20aamqmfk','ZTF19aarioci',
    'ZTF19acspeuw','ZTF20abefeab','ZTF22aaabqko','ZTF22abajudi',
    'ZTF18actaqdw','ZTF18acaqdaa','ZTF19aatylnl','ZTF20acitpfz',
    'ZTF21abjrysr','ZTF22aagvrlq','ZTF22aagyuao','ZTF20acqoiyt',
    'ZTF18aakelin','ZTF19abzrhgq','ZTF19abhejal','ZTF19abidbya',
    'ZTF22abegjtx','ZTF22aacgcwv','ZTF20abisysx','ZTF21aauuybx',
    'ZTF20abgwfek','ZTF20aahmtso','ZTF22aaddwbo','ZTF20aabqihu',
    'ZTF21abhrchb','ZTF19accmaxo','ZTF21aanxhjv',
    'ZTF21aaaokyp',
]

todo = [(z, *coord_map.get(z, (None, None))) for z in training if z in available]

print(f'=== Batch Rebuild v2: {len(todo)} sources ===\n')

idx = json.loads(open('index.json').read())

for i, (ztf_name, ra, dec) in enumerate(todo):
    sid = f'{ztf_name}_PRF'
    
    # Step 1: rebuild adapter
    cmd = ['python3', 'ztf_adapter.py', ztf_name]
    if ra is not None:
        cmd += ['--ra', str(ra), '--dec', str(dec)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    bl_line = [l for l in r.stdout.split('\n') if 'Baseline:' in l]
    bl_info = bl_line[0].strip() if bl_line else '?'
    
    # Step 2: update index
    npy_path = Path(f'../ZTF_TDE/data/TS/Flux/TDE/{ztf_name}_difference_photometry_flux.npy')
    arr = np.load(npy_path)
    idx[sid] = {'label': 'TDE', 'n_points': len(arr), 'source': 'PRF paper v2'}
    
    # Step 3: classify
    r2 = subprocess.run(['python3', 'classify.py', sid, '--mode', 'text', '--n-shot', '1', '--force'],
                       capture_output=True, text=True, timeout=120)
    result_line = [l for l in r2.stdout.split('\n') if '[done]' in l]
    cls_info = result_line[0].strip().split('] ',1)[-1] if result_line else '?'
    
    print(f'[{i+1:2d}/{len(todo)}] {sid:35s} {bl_info:60s} → {cls_info}')

open('index.json', 'w').write(json.dumps(idx, indent=2, ensure_ascii=False))
print(f'\n=== Done ===')