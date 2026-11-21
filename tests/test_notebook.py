from pathlib import Path

import nbformat
from nbclient import NotebookClient

from scripts.build_notebook import build_notebook
from scripts.check_notebook import check_notebook, lint_code, main as lint_main
from src.scoring import validate_submission
from src.synthetic import generate_synthetic

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_notebook_in_sync():
    actual = nbformat.read(ROOT/'solution.ipynb', as_version=4)
    expected = build_notebook()
    assert [c.source for c in actual.cells if c.cell_type == 'code'] == [c.source for c in expected.cells if c.cell_type == 'code']
    assert actual.metadata.kernelspec.name == 'python3'
    assert all(c.execution_count is None and not c.outputs for c in actual.cells if c.cell_type == 'code')


def test_notebook_lint():
    assert check_notebook(ROOT/'solution.ipynb') == []


def test_negative_lint(tmp_path, capsys):
    bad = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell('import requests')])
    path = tmp_path/'bad.ipynb'
    nbformat.write(bad, path)
    assert lint_main([str(path)]) == 1
    assert 'banned import requests' in capsys.readouterr().out
    assert lint_code('import os as o\no.system("whoami")')
    assert lint_code('from urllib import request')
    assert lint_code('exec("pass")')
    assert lint_code('mint = "'+'A'*32+'"')
    assert lint_code('coin = "So11111111111111111111111111111111111111112"') == []


def test_standalone_notebook_execution(tmp_path, monkeypatch):
    data = tmp_path/'inputs'/'synthetic'
    generate_synthetic(data, days=7, tokens_per_day=45)
    work = tmp_path/'isolated'
    work.mkdir()
    monkeypatch.setenv('PUMP_DATA_DIR', str(data))
    notebook = nbformat.read(ROOT/'solution.ipynb', as_version=4)
    NotebookClient(notebook, timeout=300, kernel_name='python3',
                   resources={'metadata': {'path': str(work)}}).execute()
    submission = pd.read_csv(work/'submission.csv')
    assert validate_submission(submission, pd.read_csv(data/'test.csv'))
    output = '\n'.join(o.get('text', '') for c in notebook.cells if c.cell_type == 'code' for o in c.outputs)
    assert 'Zero-take fallback' not in output
    assert 'default_pnl' in output and 'tp distribution' in output
