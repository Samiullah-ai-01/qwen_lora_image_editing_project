# SignForge Local

**Local-First Signage Mockup Generator with Multi-LoRA Composition**

SignForge Local is a powerful, privacy-focused tool for generating professional signage mockups. It leverages the Qwen-Image model and a sophisticated Multi-LoRA composition system to create high-quality images tailored to specific sign types, materials, and environments—all running locally on your hardware.

## 🚀 Key Features

-   **Privacy-First & Local:** Runs entirely on your machine. No data leaves your network.
-   **Multi-LoRA Composition:** Combine multiple adapters (e.g., "Channel Letters" + "Brick Wall" + "Night") for precise control.
-   **Single GPU Optimization:** Optimized for consumer GPUs (24GB VRAM recommended, 16GB min) with `bf16`, `xformers`, and CPU offloading.
-   **Interactive UI:** React-based frontend for easy prompting, adapter selection, and gallery viewing.
-   **Comprehensive API:** Full REST API for integration and automation.
-   **Developer Friendly:** Modular architecture, extensive testing, and clear documentation.

## 🛠️ Quick Start

### Prerequisites

-   **OS:** Linux (Ubuntu 20.04+) or Windows 10/11 (WSL2 recommended)
-   **GPU:** NVIDIA GPU with 16GB+ VRAM (24GB recommended for best performance)
-   **Software:**
    -   Python 3.10+
    -   Node.js 18+ (for UI)
    -   CUDA 12.1+

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/signforge-local.git
    cd signforge-local
    ```

2.  **Run the setup script:**
    ```bash
    # Linux/Mac
    ./scripts/setup_dev.sh
    
    # Windows (PowerShell)
    ./scripts/setup_dev.ps1
    ```
    This will create a virtual environment, install dependencies, and build the frontend.

3.  **Download models:**
    ```bash
    source .venv/bin/activate
    python scripts/download_models.py
    ```
    *Note: You may need a Hugging Face token for some models.*

4.  **Start the server:**
    ```bash
    # Linux/Mac
    ./scripts/server.sh
    
    # Windows
    ./scripts/server.ps1
    ```

5.  **Open the UI:**
    Navigate to `http://localhost:8000` in your browser.

## 🏗️ Project Structure

```
signforge-local/
├── configs/               # Configuration files (YAML)
│   ├── app.yaml           # Main application config
│   └── training/          # LoRA training configs
├── data/                  # Data directory
│   ├── raw/               # Raw training images
│   └── processed/         # Preprocessed datasets
├── models/                # Model storage
│   ├── base/              # Base model (Qwen-Image)
│   └── loras/             # LoRA adapters
├── outputs/               # Generated outputs & logs
├── scripts/               # Utility scripts
├── src/                   # Source code
│   └── signforge/
│       ├── core/          # Core utilities (config, logging, device)
│       ├── ml/            # ML pipeline & LoRA management
│       ├── server/        # Flask backend
│       ├── ui/            # React frontend
│       └── ...
└── tests/                 # Unit & integration tests
```

## 🧠 LoRA System

SignForge uses a categorized LoRA system:

1.  **Sign Type:** `channel_letters`, `box_sign`, `neon`, `monument`
2.  **Mounting:** `flush`, `raceway`, `standoff`, `hanging`
3.  **Perspective:** `front`, `angle`, `street_level`
4.  **Environment:** `urban`, `mall`, `night`, `suburban`
5.  **Lighting:** `front_lit`, `halo_lit`, `backlit`

Combine these in the UI to create complex scenes:
> "A neon sign reading 'OPEN' on a brick wall at night"
> -> Activates: `sign_type/neon` + `environment/night` + `material/brick`

## 💻 Development

### Running Tests
```bash
pytest
```

### Building Frontend
```bash
python -m signforge.ui.build
```

### Training a New LoRA
1.  Place images and captions in `data/raw/<domain>/<concept>/`.
2.  Preprocess:
    ```bash
    python -m signforge.data.preprocess
    ```
3.  Train:
    ```bash
    ./scripts/train.sh <domain>_<concept>
    ```

## 📊 Monitoring

SignForge includes built-in Prometheus metrics.
1.  Start the monitoring stack:
    ```bash
    docker-compose -f src/signforge/monitoring/dashboard/docker-compose.monitoring.yaml up -d
    ```
2.  View Grafana dashboard at `http://localhost:3000`.

## 🤝 Contributing

Contributions are welcome! Please read `CONTRIBUTING.md` (if available) and follow the code of conduct.
1.  Fork the repo
2.  Create a feature branch
3.  Submit a Pull Request

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.
