# ASL Hand Gesture Translator (AIN3002 Final Project)

This project implements a real-time American Sign Language (ASL) hand gesture recognition system using a webcam and a dual-input CNN model (image + landmarks). It speaks each word aloud when completed.

---

## 📁 Project Structure

```
📂 .idea/               # IDE configuration files
📂 Training/            # Training scripts for all models
│   ├── Dual_CNN_Training.py
│   ├── ResNet50_Training.py
│   └── Simple_CNN_Training.py
📂 venv/                # Python virtual environment (optional)
├── Dual_CNN.keras      # Trained dual-input model
├── main.py             # Real-time ASL translator script
├── requirements.txt    # Required libraries
├── README.md           # Instruction and documentation file
```

---

## 🧠 How It Works

- Tracks hand with MediaPipe  
- Extracts 21 3D landmarks  
- Captures cropped grayscale hand image  
- Passes both inputs into a trained CNN model  
- Confirms a letter if same prediction is stable for 1.5s  
- When hand is removed, adds a space and speaks the full sentence aloud

---

## 📚 Dataset Used

- **Custom dataset** (collected and processed manually)
- [Kaggle: Synthetic ASL Alphabet Dataset](https://www.kaggle.com/datasets/lexset/synthetic-asl-alphabet)

---

## 🚀 How to Use

### 1. ✅ Install Dependencies

Open your terminal and run:

```bash
pip install -r requirements.txt
```

### 2. 📁 Prepare Files

Ensure the following files are present in the project folder:

- `Dual_CNN.keras` (your trained model)
- `label_names2.txt` (one label per line, e.g., A-Z and blank)

### 3. ▶️ Run the Application

```bash
python main.py
```

### 4. ✋ Use Gestures

- Show an ASL sign with one hand  
- Keep your hand steady for ~1.5 seconds  
- The letter is added automatically  
- Remove hand from screen to insert a space and trigger text-to-speech

### 5. ⌨️ Keyboard Shortcuts

| Key | Action                  |
|-----|-------------------------|
| R   | Reset the full sentence |
| B   | Delete last character   |
| Q   | Quit the application    |

---

## 📂 Training Scripts

If you wish to retrain or explore model training, check the `Training/` folder:

- `Simple_CNN_Training.py` – Basic 3-layer CNN for image-only input  
- `ResNet50_Training.py` – Transfer learning using ResNet50  
- `Dual_CNN_Training.py` – Combined model using both image and landmarks
