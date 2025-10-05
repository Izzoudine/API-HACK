# NASA Climate API

This project is a FastAPI-based backend that predicts extreme weather probabilities (such as heatwaves, heavy rain, or cold conditions) for a given date and location. It is part of the **NASA Space Challenge 2025** initiative to use AI and data to anticipate climate extremes that impact human life and ecosystems.

---

## 📁 Project Structure

```
├── .gitignore
├── main.py               # Entry point that runs the FastAPI app
├── requirements.txt      # Dependencies for the API
├── api/
│   └── forecast.py       # Contains the core logic for forecasting
└── README.md             # Project documentation
```

---

## 📍 Repository and Online Endpoint

- **GitHub Repository**: The source code is hosted at [https://github.com/Izzoudine/API-HACK/](https://github.com/Izzoudine/API-HACK/)  
  *Note: The repository may currently be empty or in early development. Ensure all files are pushed to make the project accessible.*

- **Online API Endpoint**: The API is deployed at [http://3.22.102.204/api3](http://3.22.102.204/api3)  

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/Izzoudine/API-HACK.git
cd API-HACK
```

### 2. Create and activate a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Run the API Locally

Start the FastAPI server:

```bash
uvicorn main:app --reload
```

By default, the API runs at:
```
http://127.0.0.1:8000
```

---

## 🌐 NASA Space Challenge 2025

This project contributes to the NASA Space Challenge 2025 by leveraging AI and climate data to predict extreme weather events, helping communities prepare for and mitigate the impacts of climate change.
