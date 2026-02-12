# Human–AI Collaborative Loan Decision Support System

## Overview

This project implements a controlled experimental platform for evaluating Human–AI collaboration in loan decision making.

Participants complete 40 loan decisions in a within-subject design:

* **20 decisions without AI support**
* **20 decisions with AI support** (recommendation + confidence + explanation)

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
pip install -r requirements.txt
```

---

## Running

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


