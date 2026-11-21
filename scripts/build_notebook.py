"""Build the self-contained offline solution from the authoritative modules."""
import ast
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
MODULES = ['features', 'scoring', 'model']
INTRO = """# Memecoins: Pump or Dump? Predict Pump.fun Launches on Solana

An offline CPU baseline for the community competition. This notebook contains
all runtime code: no repository, network access, external wallet data, serialized
models, or test outcomes are required. Attach the competition input and run all.

## Payoff and decision
Each token is skipped (`tp=0`) or assigned a take-profit in `[3,100]`. A reached
take returns `tp - 1.3`; otherwise it returns `max(final_x - 1.3, -1)`.
For `max_x=10, final_x=0.5`, choices `0,5,10,20` return `0,3.7,8.7,-0.8`.
Entry is the last AMM price at migration +60 seconds. Outcomes describe the
following hour, not the observed first-minute tape.

## Method and safeguards
Projected per-day parquet reads preserve chain/file order. Curve, first-minute
AMM, launch, migration and point-in-time creator/wallet/regime features feed
three-seed classifiers for reaches of 3/5/7/10/20x. Temporal held-out predictions
fit Platt calibration and held-payoff bins. A fixed take grid maximizes expected
value. Margins/exclusions are tuned on training OOF only; their reported PnL is
**in-sample-tuned**, with `default_pnl` (margin zero, no exclusions) and a tp=5
break-even baseline alongside it. Default PnL still uses OOF-fitted calibration
and held-payoff estimates; it is not nested unbiased validation.

Labels enter history only at `mt+3600`; each walk-forward fold trains strictly
before its day minus two hours. No test-wide ranking, quantiles, caps or outcome
lookup is used. Test decisions use training-fixed parameters independently.
Missing labels/failed modelling produce zero-take submissions; missing tapes
produce NaNs; LightGBM falls back to sklearn HGB.

**Warnings:** untested on real competition data. Synthetic numbers are NOT a
competition score. This is a community-run competition; prizes depend on the
individual sponsor and are not an official Kaggle payout. Check current rules,
including Kaggle foundational rule 6b about sharing competition code on Kaggle
forums/notebooks, before relying on a public GitHub repository for eligibility.
Not financial advice. See the project README for submission and freeze details.
"""
DRIVER = """submission, walk_forward = run_pipeline()
print('Walk-forward: pnl is in-sample-tuned on training OOF; default_pnl has margin=0 and no exclusions.')
print(walk_forward.to_string(index=False))
if len(walk_forward):
    print('PnL totals:', walk_forward[['pnl', 'default_pnl', 'baseline_pnl']].sum().to_dict())
print(f'Submission: {len(submission)} tokens; n buys={int(submission.tp.gt(0).sum())}')
print('tp distribution:')
print(submission.tp.value_counts().sort_index().to_string())
print('Wrote ./submission.csv; attempted /kaggle/working/submission.csv.')
"""


def inline_module(path):
    """Strip entire local import statements, including continuation lines."""
    source = Path(path).read_text()
    lines = source.splitlines()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and (node.level or (node.module or '').startswith('src')):
            for i in range(node.lineno-1, node.end_lineno):
                lines[i] = ''
    return '\n'.join(lines).strip()+'\n'


def build_notebook(root=ROOT):
    """Return an unexecuted notebook; deterministic code cells mirror src."""
    root = Path(root)
    cells = [nbformat.v4.new_markdown_cell(INTRO)]
    cells += [nbformat.v4.new_code_cell(inline_module(root/'src'/f'{module}.py')) for module in MODULES]
    cells.append(nbformat.v4.new_code_cell(DRIVER))
    for i, cell in enumerate(cells):
        cell.id = f'solution-{i}'
    return nbformat.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.10.0'}})


def main():
    path = ROOT/'solution.ipynb'
    nbformat.write(build_notebook(), path)
    print(f'Built {path.name}: inline features, scoring, model + driver')


if __name__ == '__main__':
    main()
