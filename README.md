# Unblurr - AI Image Restoration Suite

A powerful AI-powered application for enhancing and unblurring images using Real-ESRGAN models. Upload blurry images and make them crystal clear with advanced AI processing.

## ✨ Features

- 🎨 **Batch Processing**: Upload multiple images at once
- 🤖 **AI Models**: Choose between different models for detail or text optimization
- 📐 **Automatic Upscaling**: Increase the resolution of your images
- 🔄 **Real-time Progress**: View the progress of your processing
- 💾 **Preserved Metadata**: Original filenames and dimensions are preserved

## 🛠️ Requirements

### General
- **Node.js** 18.x or higher
- **Python** 3.8 or higher
- **npm** or **yarn** package manager

### Python Dependencies
See `requirements.txt` for the complete list. Main packages:
- PyTorch 2.5.1
- Real-ESRGAN 0.3.0
- Flask 3.0.0+
- OpenCV, NumPy, Pillow

### Models
The application uses Real-ESRGAN models that are automatically downloaded on first use. You can also manually place models in the `models/` directory.

## 📦 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd unblurr
```

### 2. Install Node.js Dependencies

```bash
npm install
```

### 3. Create Python Virtual Environment

**On Mac/Linux:**
```bash
python3 -m venv .venv
```

**On Windows:**
```bash
python -m venv .venv
```

### 4. Activate Virtual Environment

**On Mac/Linux:**
```bash
source .venv/bin/activate
```

**On Windows (Command Prompt):**
```cmd
.venv\Scripts\activate
```

**On Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

### 5. Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Note:** PyTorch installation may take some time. On some systems, you may need to install specific PyTorch versions for your platform (CPU vs GPU support).

### 6. (Optional) Download Models

Models are automatically downloaded on first use, but you can also manually download and place them in the `models/` directory:
- `realesr-general-x4v3.pth`
- `RealESRGAN_x2plus.pth`

## 🚀 Usage

### Start the Application

The application can be automatically started with the start scripts that perform all necessary checks.

**On Mac/Linux:**
```bash
chmod +x start.sh
./start.sh
```

**On Windows (Command Prompt):**
```cmd
start.bat
```

**On Windows (PowerShell):**
```powershell
.\start.ps1
```

**Note:** If you're using PowerShell scripts for the first time, you may need to adjust the execution policy:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Manual Start

If you want to start the application manually:

**Terminal 1 - Python Backend:**
```bash
source .venv/bin/activate  # Mac/Linux
# or
.venv\Scripts\activate      # Windows

python unblur_server.py
```

**Terminal 2 - Next.js Frontend:**
```bash
npm run dev
```

### Access the Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:5000

## 🐛 Troubleshooting

### Python Virtual Environment not found
```bash
python3 -m venv .venv
source .venv/bin/activate  # Mac/Linux
pip install -r requirements.txt
```

### Node.js not installed
Download and install Node.js from [nodejs.org](https://nodejs.org/)

### npm packages not installed
```bash
npm install
```

### Port already in use
If port 3000 or 5000 is already in use:
- Change the port in `package.json` (for Next.js)
- Change the port in `unblur_server.py` (for Flask backend)

### PyTorch installation issues
For CPU-only version:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

For GPU support (CUDA):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Models not found
Check if the `models/` directory exists and contains the required `.pth` files. Models are automatically downloaded on first use.

## 📁 Project Structure

```
unblurr/
├── app/                    # Next.js application
│   ├── api/               # API routes
│   ├── components/        # React components
│   └── page.tsx          # Main page
├── models/                # AI models
├── unblur_server.py      # Python Flask backend
├── start.sh              # Start script (Mac/Linux)
├── start.bat             # Start script (Windows CMD)
├── start.ps1             # Start script (Windows PowerShell)
├── requirements.txt      # Python dependencies
└── package.json          # Node.js dependencies
```

## 🔧 Development

### Frontend Development
```bash
npm run dev
```

### Backend Development
```bash
source .venv/bin/activate
python unblur_server.py
```

### Build for Production
```bash
npm run build
npm start
```

## 📝 License

[Add your license here]

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## 📧 Contact

[Add contact information here]
