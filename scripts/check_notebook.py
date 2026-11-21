"""Reject prohibited imports/calls and mint-like literals in notebook and src."""
import argparse
import ast
from pathlib import Path
import re

import nbformat

ROOT = Path(__file__).resolve().parents[1]
WSOL = 'So11111111111111111111111111111111111111112'
BANNED_MODULES = {'requests', 'urllib', 'socket', 'http.client', 'subprocess', 'pickle', 'joblib', 'base64'}
BANNED_CALLS = {'os.system', 'os.popen', 'shutil.rmtree', 'eval', 'exec', 'builtins.eval', 'builtins.exec'}
MINT = re.compile(r'[1-9A-HJ-NP-Za-km-z]{32,44}')


def _banned_import(name):
    return any(name == banned or name.startswith(banned+'.') for banned in BANNED_MODULES)


def lint_code(source, label='<code>'):
    """Return readable violations; this is a guardrail, not a security sandbox."""
    errors, aliases = [], {}
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        return [f'{label}:{error.lineno}: syntax error: {error.msg}']
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                aliases[alias.asname or alias.name.split('.')[0]] = alias.name if alias.asname else alias.name.split('.')[0]
                if _banned_import(alias.name):
                    errors.append(f'{label}:{node.lineno}: banned import {alias.name}')
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            for alias in node.names:
                name = module+'.'+alias.name
                aliases[alias.asname or alias.name] = name
                if _banned_import(module) or _banned_import(name) or name in BANNED_CALLS:
                    errors.append(f'{label}:{node.lineno}: banned import {name}')
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            parts, func = [], node.func
            while isinstance(func, ast.Attribute):
                parts.insert(0, func.attr)
                func = func.value
            if isinstance(func, ast.Name):
                name = '.'.join([aliases.get(func.id, func.id)]+parts)
                if name in BANNED_CALLS:
                    errors.append(f'{label}:{node.lineno}: banned call {name}')
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value != WSOL and MINT.fullmatch(node.value):
                errors.append(f'{label}:{node.lineno}: hardcoded mint-like base58 literal')
    return errors


def check_notebook(path, source_dir=ROOT/'src'):
    notebook = nbformat.read(path, as_version=4)
    errors, count = [], 0
    for i, cell in enumerate(notebook.cells):
        if cell.cell_type == 'code':
            count += 1
            errors.extend(lint_code(cell.source, f'{Path(path).name}:cell{i}'))
    files = sorted(Path(source_dir).glob('*.py'))
    for file in files:
        errors.extend(lint_code(file.read_text(), str(file.relative_to(ROOT)) if file.is_relative_to(ROOT) else str(file)))
    print(f'Checked {Path(path).name}: {count} code cells; {len(files)} src/*.py files ({", ".join(f.name for f in files)})')
    return errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('notebook', nargs='?', default=str(ROOT/'solution.ipynb'))
    args = parser.parse_args(argv)
    errors = check_notebook(args.notebook)
    for error in errors:
        print(error)
    print('Notebook/source lint FAILED' if errors else 'Notebook/source lint passed')
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())
