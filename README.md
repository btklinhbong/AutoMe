# AutoMe — Continuous Wearable Patch for Parkinson's Tremor Tracking & DBS Evaluation

## Project Overview & Clinical Problem
AutoMe is a biomedical engineering project under **Aggies Create (Medical Technology Incubator)** focused on continuous tremor monitoring for Parkinson's disease care.

Current DBS and medication adjustments often rely on short, subjective clinic assessments (~30 minutes). AutoMe aims to replace that with **continuous, quantitative tremor tracking** to help neurologists evaluate:
- Tremor intensity over time
- Frequency distribution of hand/finger tremors
- Response to DBS parameter changes and medication schedules

The target tremor frequency range is **3-7 Hz**, while the processing pipeline captures broader movement behavior in **1-12 Hz**.

---

## Hardware Architecture (MPU6050 + ESP32)
AutoMe uses a wearable patch architecture built around low-power sensing and local logging:

- **ESP32 Microcontroller**: main processing and control unit
- **MPU6050 6D IMU**:
  - 3-axis accelerometer
  - 3-axis gyroscope
- **microSD Card Module**: local data logging
- **RTC Module**: accurate timestamps for long-duration tracking
- **LiPo Battery**: portable power source for continuous wear

High-level data flow:
1. IMU captures motion data continuously.
2. ESP32 timestamps and writes samples to microSD.
3. CSV logs are transferred for offline signal processing and analysis.

---

## Signal Processing Pipeline
Starter analysis workflow in Python:

1. **Load CSV sensor data** with `pandas`.
2. **Apply Butterworth bandpass filter (1-12 Hz)** to isolate tremor-relevant motion.
3. **Compute FFT** to transform time-domain data into frequency-domain spectrum.
4. **Focus on 3-7 Hz tremor band** for Parkinsonian tremor evaluation.
5. **Plot frequency spectrum** for inspection and comparison across sessions.

---

## Installation & Setup
### 1) Clone and enter repository
```bash
git clone <your-repo-url>
cd AutoMe
```

### 2) Create and activate Python environment
```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows PowerShell
```

### 3) Install dependencies
```bash
pip install -r requirements.txt
```

### 4) Run starter processing script
```bash
python src/main.py --input data/sample_tremor.csv --column accel_x --sampling-rate 50
```

> Replace `data/sample_tremor.csv` with your exported sensor CSV.

---

## Deliverables & Team Structure
### Core Deliverables
- Wearable hardware prototype (ESP32 + MPU6050 patch)
- Continuous logging firmware with timestamped CSV output
- Python signal processing toolkit for tremor quantification
- Visual frequency-domain reports supporting DBS and medication review

### Suggested Team Structure
- **Hardware Lead**: sensor integration, power, enclosure
- **Firmware Lead**: ESP32 acquisition/logging and data integrity
- **Signal Processing Lead**: filtering, FFT, feature extraction
- **Clinical/Data Lead**: protocol design, labeling, interpretation support

---

## Repository Starter Structure
```text
AutoMe/
├── data/
├── docs/
├── src/
│   └── main.py
├── .gitignore
├── README.md
└── requirements.txt
```
