# Brain Tumor Detection

An application that uses deep learning to detect and segment brain tumors in MRI scans.

## Features

- Load and analyze brain MRI scans
- Detect and segment tumors using YOLO
- Display detection results with visual annotations
- Export findings as PDF reports
- User-friendly GUI interface

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended for faster inference)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/Tumor-Detection.git
   cd Tumor-Detection
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```

3. Activate the virtual environment:
   - Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Download the model file:
   - The application requires a trained YOLO model file named `best.pt`
   - Place it in the root directory of the project

## Usage

Run the application:
```bash
python TumorDetection.py
```

### Instructions:
1. Click "Insert MRI brain scan" to load an MRI image
2. The application will automatically detect and segment any tumors
3. Results will be displayed on the left panel
4. Click "Extract data as PDF" to save the analysis as a PDF report
5. Press ESC or click "Close Application" to exit

## Model Information

The application uses YOLOv8 for tumor detection and segmentation. The model was trained on a dataset of brain MRI scans with annotated tumor regions.

## License

[Your chosen license]

## Contact

[Your contact information]
