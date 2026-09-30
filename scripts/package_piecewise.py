"""Create a portable eval/ deployment snapshot with the trained JSON prior."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/piecewise_polynomial/piecewise_wan_deployment.zip')
    args=parser.parse_args()
    paths=list(ROOT.glob('*.py'))
    for directory,pattern in (('model','*.py'),('conf','*.yaml'),('tests','test*.py')):
        paths.extend((ROOT/directory).glob(pattern))
    paths.extend(ROOT/name for name in ('requirements.txt','prompts.txt','PIECEWISE_DEPLOYMENT.txt'))
    artifact=ROOT/'artifacts/piecewise_polynomial'
    paths.extend(artifact/name for name in ('prior.json','summary.json','ablations.json','manifest.json',
        'fold_metrics.csv','per_trajectory.csv','fit_and_ablation.png'))
    validation=artifact/'deployment_validation.json'
    if validation.is_file():
        checked=json.loads(validation.read_text())
        for name,digest in checked['source_sha256'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
                raise ValueError(f'deployment validation is stale for {name}')
        paths.append(validation)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    records=[]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.output,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(set(paths)):
            name='eval/'+path.relative_to(ROOT).as_posix()
            data=path.read_bytes()
            z.writestr(name,data)
            records.append({'path':name,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)})
        z.writestr('deployment_manifest.json',json.dumps({'files':records,
            'scope':'Wan inference adapter + frozen polynomial prior; Wan code and checkpoint supplied by server',
            'reconstruction':'polynomial in cumulative predicted relative-change coordinate'},indent=2)+'\n')
    print(args.output.resolve())
    print(f'{len(records)} files; {args.output.stat().st_size} bytes')


if __name__=='__main__':
    main()
