# 🚀 BHL2025-MLovers - Quick Start Guide


## 🎯 Setup 

### Step 1: Install Dependencies

Open terminal and run:

```bash
cd BHL2025-MLovers
pip install -r requirements.txt
```



## 🚀 Run the Application


### Terminal 1: Run Streamlit App

Open **NEW terminal** and run:

```bash
cd frontend
streamlit run Hardware_Recommender.py
```

**You should see:**
```
Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

---

## 🌐 Access the App

### ✅ Open in Browser

Click the link or go to:
```
http://localhost:8501
```

**You should see:**
- Logo and title
- Navigation menu on left
- Streamlit interface

---

## 📊 Project Pages

1. **Hardware Recommender** - Hardware recommendations
2. **Device Management System** - Device data
3. **Demand Prediction** - Employee growth forecast (SARIMA model)


---

## 🛑 Stop the App

### In Terminal 2 (Streamlit):
```
Press Ctrl+C
```

### In Terminal 1 (Backend):
```
Press Ctrl+C (if still running)
```


## Troubleshooting

### Port 8501 Already in Use

If you see: `Error: Address already in use`

**Option 1 - Use different port:**
```bash
streamlit run Hardware_Recommender.py --server.port 8502
```
Then access: `http://localhost:8502`

**Option 2 - Kill process using port:**
```bash
# Windows
netstat -ano | findstr :8501
taskkill /PID <PID> /F

# Mac/Linux
lsof -i :8501
kill -9 <PID>
```

### Models Not Found

If you see: `❌ SARIMA model not found`

**Solution:**
```bash
cd backend
python model_training.py
```

Wait for completion, then try again.

### Import Errors

If you see: `ModuleNotFoundError`

**Solution:**
```bash
pip install -r requirements.txt
```

Then try again.

---

## 📊 Data Files

Make sure these files exist in `data/` folder:
- `company_growth_detailed.csv` - Detailed employee growth metrics
- `devices_with_prices.csv` - Hardware data

---

## ✅ You're All Set!

Everything is ready to go. Just:

1. **Terminal 1:** `cd frontend && streamlit run Hardware_Recommender.py`
2. **Open:** http://localhost:8501

Enjoy! 🎉

