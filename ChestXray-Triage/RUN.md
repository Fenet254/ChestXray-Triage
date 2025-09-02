conda env create -f environment.yml && conda activate cxr
make prep && make train && make eval && make demo