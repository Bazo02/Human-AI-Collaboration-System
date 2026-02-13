# Human–AI Collaborative Loan Decision Support System

## Overview

This project implements a controlled experimental platform for evaluating Human–AI collaboration in loan decision making.

Participants complete 24 loan decisions.

* **12 decisions without AI support**
* **12 decisions with AI support** (recommendation + confidence + explanation)

The system measures:

* Decision accuracy
* Decision time
* Trust in AI
* Perceived usability
* AI reliance behavior

---

## System

Participants review synthetic loan applications and decide to **Approve** or **Reject** each case.

All decisions, interaction events, and survey responses are stored in a SQLite database.

An admin interface provides:

* Aggregated metrics and visualizations
* Condition comparison (baseline vs AI-assisted)
* Participant data management

---

## Tech Stack

* Python (Flask)
* scikit-learn (Logistic Regression)
* pandas
* matplotlib (server-side backend)
* SQLite
* HTML / CSS / JavaScript

---

## Installation

```bash
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate # Mac
pip install -r requirements.txt
```

---

## Running from main branch

Prepare data and train model:

```bash
python app/data_prep.py
python app/model_train.py
```

Start the server:

```bash
python app/main.py
```

Open:

```
http://127.0.0.1:5000
```

Admin panel:

```
http://127.0.0.1:5000/admin
```

---


## Run localy from deploy branch (meant for renderer) 

From the project root directory:

```bash
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Mac/Linux

pip install -r requirements.txt

python -m app.data_prep
python -m app.model_train
python -m app.main
```

Then open:

```
http://127.0.0.1:5000
```

**Important:** Run the commands from the project root folder, not inside `/app`.

```

```

## Data

All study data is stored in:

```
/outputs/study.db
```

Tables:

* `decisions`
* `events`
* `surveys`

---

## Research Purpose

The system enables controlled evaluation of:

* Human-only vs AI-assisted performance
* Trust and reliance in AI systems
* Effects of AI explanations on decision making


