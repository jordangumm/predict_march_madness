wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p miniconda
rm miniconda.sh

source miniconda/bin/activate

pip install -r https://raw.githubusercontent.com/Lasagne/Lasagne/v0.1/requirements.txt
pip install -r requirements.txt
