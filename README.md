# Human–AI Collaborative Loan Decision Support System

## Overview

This project implements a controlled experimental platform for evaluating Human–AI collaboration in loan decision making.

Participants complete 24 loan decisions:

* **12 decisions without AI support**
* **12 decisions with AI support** (recommendation + confidence + explanation)

The system measures:

* Decision accuracy
* Decision time
* Trust in AIA
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
source venv/bin/activate # Mac/Linux

pip install -r requirements.txt
```

---

## Run Locally From Deploy Branch (Meant for Render)

Run the following commands from the project root directory:

```bash
python -m venv venv

venv\Scripts\activate   # Windows
source venv/bin/activate  # Mac/Linux

pip install -r requirements.txt

python -m app.data_prep
python -m app.model_train
python -m app.main

```

Then open:

```text
http://127.0.0.1:5000
```

**Important:** Run the commands from the project root folder, not inside `/app`.

---

## Data

All study data is stored in:

```text
/outputs/study.db
```

### Tables

* `decisions`
* `events`
* `surveys`

---

## Research Purpose

The system enables controlled evaluation of:

* Human-only vs AI-assisted performance
* Trust and reliance in AI systems
* Effects of AI explanations on decision making