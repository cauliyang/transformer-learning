# Transformer-Learning

This repository serves as a curated codebase to accompany my exploration of the exceptional blog: http://nlp.seas.harvard.edu/annotated-transformer/.
The intent is to restructure the code for seamless interaction in a command-line environment, while judiciously maintaining minimal dependencies to enrich the learning experience.
To this end, console output will still utilize the `print` function, albeit enhanced via the `rich` library, as opposed to employing specialized logging libraries.

## How to

This project is orchestrated through `Poetry`, thereby facilitating dependency management and package distribution.
The architecture accommodates relative imports, permitting a modular and organized code structure.

```python
from .transformer import make_model

test_model = make_model()
```

- Install dependencies via poetry

```sh
poetry install
```

- run examples

```sh
examples
```

## Acknowledge

- http://nlp.seas.harvard.edu/annotated-transformer/
