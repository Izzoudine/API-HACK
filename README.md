NASA Climate API
This project is a FastAPI-based backend that predicts extreme weather probabilities (such as heatwaves, heavy rain, or cold conditions) for a given date and location. It is part of the NASA Space Challenge 2025 initiative to use AI and data to anticipate climate extremes that impact human life and ecosystems.
📁 Project Structure
├── .gitignore
├── main.py               # Entry point that runs the FastAPI app
├── requirements.txt      # Dependencies for the API
├── api/
│   └── forecast.py       # Contains the core logic for forecasting
└── README.md             # Project documentation

📍 Repository and Online Endpoint

GitHub Repository: The source code is hosted at https://github.com/Izzoudine/API-HACK/. Note: The repository may currently be empty or in early development. Ensure all files are pushed to make the project accessible.
Online API Endpoint: The API is deployed at http://3.22.102.204/api3. If you encounter a "404 Not Found" error, verify the route in main.py or try accessing the root http://3.22.102.204/ or Swagger docs at http://3.22.102.204/docs.

⚙️ Installation & Setup

Clone the repository
git clone https://github.com/Izzoudine/API-HACK.git
cd API-HACK


Create and activate a virtual environment
Windows (PowerShell):
python -m venv venv
venv\Scripts\activate

Linux / macOS:
python3 -m venv venv
source venv/bin/activate


Install dependencies
pip install -r requirements.txt



🚀 Run the API Locally
Start the FastAPI server:
uvicorn main:app --reload

By default, the API runs at:
http://127.0.0.1:8000
