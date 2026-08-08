#!/usr/bin/env python3
"""Batch process all PRF TDE sources."""
import subprocess, sys, os, csv, json
from pathlib import Path

os.chdir(Path(__file__).parent)

# Load coordinates
coord_map = {}
with open('../ZTF_TDE/lasair/TNS+vVReview_TDEs_Copy_watchlist_results.csv') as f:
    for row in csv.DictReader(f):
        ztf = row.get('objectId', '').strip()
        if ztf:
            coord_map[ztf] = (float(row['ramean']), float(row['decmean']))

# Available sources
flux_dir = Path('../ZTF_TDE/data/TS/Flux/TDE/')
available = {f.stem.replace('_difference_photometry_flux', '') 
             for f in flux_dir.glob('*.npy') if 'irsa_full' not in f.name}

# Training TDEs
training = [
    'ZTF21aabiipy','ZTF20achpcvt','ZTF21aapvvtb','ZTF21aaaokyp',
    'ZTF22aabimec','ZTF20abnorit','ZTF19aapreis','ZTF19aakiwze',
    'ZTF21abmwftm','ZTF19aakswrb','ZTF22abkfhua','ZTF20abjwvae',
    'ZTF21acafvhf','ZTF22aaaedas','ZTF20abowque','ZTF21abqtckk',
    'ZTF22aaabovl','ZTF19abhhjcc','ZTF20abfcszi','ZTF19aabbnzo',
    'ZTF21abcgnqn','ZTF20acyydkh','ZTF20acnznms','ZTF22aaahtqz',
    'ZTF21aaeoitd','ZTF18abxftqm','ZTF17aaazdba','ZTF21abxngcz',
    'ZTF20acvezvs','ZTF20aamqmfk','ZTF22aavvqyh','ZTF19aarioci',
    'ZTF19acspeuw','ZTF20abefeab','ZTF22aaabqko','ZTF22abajudi',
    'ZTF18actaqdw','ZTF18acaqdaa','ZTF19aatylnl','ZTF20acitpfz',
    'ZTF21abjrysr','ZTF22aagvrlq','ZTF22aagyuao','ZTF20acqoiyt',
    'ZTF18aakelin','ZTF19abzrhgq','ZTF19abhejal','ZTF19abidbya',
    'ZTF22abegjtx','ZTF22aacgcwv','ZTF20abisysx','ZTF21aauuybx',
    'ZTF20abgwfek','ZTF20aahmtso','ZTF22aaddwbo','ZTF20aabqihu',
    'ZTF21abhrchb','ZTF19accmaxo','ZTF21aanxhjv',
]

done = {'ZTF21aaaokyp', 'ZTF20achpcvt'}

todo = []
for z in training:
    if z in available and z not in done:
        ra, dec = coord_map.get(z, (None, None))
        todo.append((z, ra, dec))

print(f'=== Batch PRF TDE: {len(todo)} sources ===\n')

idx = json.loads(open('index.json').read())

ok = fail = 0
for i, (ztf_name, ra, dec) in enumerate(todo):
    sid = f'{ztf_name}_PRF'
    print(f'[{i+1}/{len(todo)}] {sid} ... ', end='', flush=True)
    
    # Step 1: ztf_adapter
    cmd = ['python3', 'ztf_adapter.py', ztf_name]
    if ra is not None:
        cmd += ['--ra', str(ra), '--dec', str(dec)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        print(f'FAIL (adapter: {r.stderr.strip()[-100:]})')
        fail += 1
        continue
    
    # Step 2: index
    import numpy as np
    npy_path = Path(f'../ZTF_TDE/data/TS/Flux/TDE/{ztf_name}_difference_photometry_flux.npy')
    arr = np.load(npy_path)
    idx[sid] = {'label': 'TDE', 'n_points': len(arr), 'source': 'PRF paper (Anilkumar et al. 2026)'}
    
    # Step 3: classify
    r2 = subprocess.run(['python3', 'classify.py', sid, '--mode', 'text', '--n-shot', '1', '--force'],
                       capture_output=True, text=True, timeout=120)
    
    # Parse result
    result_line = [l for l in r2.stdout.split('\n') if '[done]' in l]
    if result_line:
        print(result_line[0].strip().split('] ',1)[-1])
    else:
        err_line = [l for l in r2.stderr.split('\n') if l.strip()]
        print(f'classify: {err_line[-1][:80] if err_line else "unknown"}')
    
    ok += 1

# Save index
open('index.json', 'w').write(json.dumps(idx, indent=2, ensure_ascii=False))
print(f'\n=== Done: {ok} ok, {fail} fail ===')