"""Reconstruye el dataset YOLO del manifiesto sin dividir ni entrenar de nuevo.

Uso desde la raíz del repositorio: python src/prepare_dataset.py
"""

import csv
import hashlib
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / 'data' / 'split_manifest.csv'
DATASET = ROOT / 'dataset_yolo26_mate_termo'
SPLITS = ('train', 'val', 'test')


def digest(path: Path) -> str:
    """Calcula el SHA-256 de un archivo.

    Args:
        path: Archivo a comprobar.

    Returns:
        Hash hexadecimal.
    """
    with path.open('rb') as handle:
        checksum = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            checksum.update(chunk)
        return checksum.hexdigest()


def prepare_dataset() -> None:
    """Copia las imágenes y etiquetas del split fijo y escribe su YAML.

    Verifica hashes, nombres y aislamiento de grupos antes de copiar. No
    borra directorios ni sobrescribe archivos existentes diferentes.

    Raises:
        ValueError: Si el manifiesto o los archivos no son coherentes.
    """
    with MANIFEST.open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows or len({r['image_name'] for r in rows}) != len(rows):
        raise ValueError('Manifiesto vacío o con imágenes duplicadas.')
    grouped = {}
    copies = []
    expected = {}
    for row in rows:
        name, split, group = (row[k] for k in ('image_name', 'split', 'group'))
        if Path(name).name != name or split not in SPLITS:
            raise ValueError('Nombre o split no permitido en el manifiesto.')
        if group in grouped and grouped[group] != split:
            raise ValueError('Una sesión está repartida entre dos splits.')
        grouped[group] = split
        for kind, source, checksum in [
            ('images', ROOT / 'dataset' / name, row['image_sha256']),
            ('labels', ROOT / 'data/unified/labels' / (Path(name).stem
                                                     + '.txt'),
             row['label_sha256']),
        ]:
            if digest(source) != checksum:
                raise ValueError(f'Hash diferente del manifiesto: {source}')
            destination = DATASET / kind / split / source.name
            if not destination.resolve().is_relative_to(ROOT.resolve()):
                raise ValueError('Destino fuera del proyecto.')
            if destination.exists() and digest(destination) != checksum:
                raise ValueError(f'Archivo existente diferente: {destination}')
            expected.setdefault(destination.parent, set()).add(source.name)
            copies.append((source, destination))
    for folder, names in expected.items():
        if folder.exists():
            extra = {p.name for p in folder.iterdir() if p.is_file()} - names
            if extra:
                raise ValueError(f'Archivos ajenos al split en {folder}: '
                                 f'{sorted(extra)}')
    for source, destination in copies:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(source, destination)
    yaml = ('# Rutas relativas a este archivo; no depende de /content.\n'
            'train: images/train\nval: images/val\ntest: images/test\n'
            'names:\n  0: mate\n  1: termo\n')
    (DATASET / 'data.yaml').write_text(yaml, encoding='utf-8')
    shutil.copy2(MANIFEST, DATASET / 'split_manifest.csv')
    for split in SPLITS:
        print(f'{split}: {sum(r["split"] == split for r in rows)} imágenes')
    print('Dataset preparado sin reentrenar:', DATASET)
    print('Configuración portable del repositorio:', ROOT / 'data.yaml')


if __name__ == '__main__':
    prepare_dataset()
