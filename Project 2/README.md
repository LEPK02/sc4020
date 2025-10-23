# Disease & Cancer Pattern Analysis

This project analyses symptom co-occurrence patterns in disease profiles using the Apriori algorithm and identifies feature patterns in cancer diagnosis using sequential pattern mining (GSP).

---

## Installation

Requirements: [Python 11](https://www.python.org/downloads/release/python-3110/), [pip](https://pip.pypa.io/en/stable/installation/), [pipenv](https://pipenv.pypa.io/en/latest/installation.html)

Supported OS: Windows

1. **Clone the repository:**
```bash
git clone "https://github.com/LEPK02/sc4020.git"
cd "./Project 2/src"
```

2. **Install dependencies:**
```bash
pipenv install
```

---

## Running

Run the program from `/src`:

```bash
pipenv run python __main__.py
```

Or:

```bash
pipenv shell
python __main__.py
```

---

## Development

To modify or update dependencies:

1. Activate the development environment from `/src`:

```bash
pipenv shell
```

2. Add or remove packages:

```bash
pipenv install <package_name>       # Add a new dependency
pipenv uninstall <package_name>     # Remove a dependency
```

3. Update the `Pipfile.lock`:

```bash
pipenv lock
```

4. Regenerate `requirements.txt`:

```bash
pipenv lock -r > requirements.txt
```
