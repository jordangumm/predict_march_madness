mkdir dependencies

git clone https://github.com/jordangumm/automaxout.git dependencies/automaxout
git clone https://github.com/jordangumm/cbbstats.git dependencies/cbbstats

pip install -r requirements.txt

pipenv run bash dependencies/cbbstats/build.sh
pipenv run python build_examples.py
