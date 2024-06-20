from pathlib import Path

home_dir = Path.cwd()

Path('data').mkdir(parents=True,exist_ok=True)
Path('data/raw').mkdir(parents=True,exist_ok=True)
Path('data/processed').mkdir(parents=True,exist_ok=True)
Path('data/raw/test.txt').touch(exist_ok=True)
Path('data/.gitignore').touch(exist_ok=True)
Path('models').mkdir(parents=True,exist_ok=True)
Path('notebooks').mkdir(parents=True,exist_ok=True)
Path('src').mkdir(parents=True,exist_ok=True)
Path('src/make_dataset.py').touch(exist_ok=True)
Path('src/train_model.py').touch(exist_ok=True)
Path('src/test_model.py').touch(exist_ok=True)
Path('dvc.yaml').touch(exist_ok=True)
Path('params.yaml').touch(exist_ok=True)
Path('.gitignore').touch(exist_ok=True)
Path('requirements.txt').touch(exist_ok=True)
