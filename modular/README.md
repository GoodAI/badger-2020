# modular

experiments with task modularization


# Basic Requirements

All the requirements should be contained in the `requirements.txt`, furthermore,
 the python 3.7 (and 3.6 on AWS) and [PyTorch 1.5](https://pytorch.org/) (no GPU required) was used.

    pip install -r requirements.txt
   
Some dependencies need to be installed manually. For these, follow instruction how to install their dependencies the `requirements.txt`.

 
The experiment results are logged into the [Sacred](https://github.com/IDSIA/sacred) and observed using [Omniboard](https://github.com/vivekratnavel/omniboard) tool. 
These can be installed as a docker image:

1. Install [Docker](https://www.docker.com/) and [Docker Compose](https://docs.docker.com/compose/install/)
2. `cd sacred`
3. `docker-compose up -d`
4. Open `localhost:9000` in browser (note port 9000 can be changed in `sacred/docker-compose.yml`)
5. Uncheck any filters applied in the top row.
6. See the "Setting up the Sacred server" below for switching to local Sacred storage 

# Running the experiments

Note, **the project root** here is the folder `badger-2020/modular/`.


The experiments are ran using the Sacred tool and the main configuration file is under `coral/coral_config.py`. An example how to run the experiments is the following: 

```
python coral/baseline_experiment.py with lr=0.0001 policy=RnnCommNet matrix=robot p_branching=0.5
```
Other examples can be found in the `0.sh`.

# Rendering the results

Learned polices can be rendered using pygame. To render a learned policy, run e.g.:

```
python coral/render.py 34
```

which download from Sacred and render the last serialized model parameters from the experiment ID 34.


# Results comparison

Graphs in the blogpost were rendered using the `coral/compare.py` script. To compare multiple runs stored in Sacred, run e.g.:

```
python coral/compare.py 39 41
```


