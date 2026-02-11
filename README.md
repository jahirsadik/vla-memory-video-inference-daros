# Video Inference Automation with SGLang

Automated, scalable pipeline for running video inference on multiple Vision Language Models using SGLang.

## 📁 Project Structure

```
video-inference-automation/
├── config/
│   └── models.yaml              # Model configurations and endpoints
├── src/
│   ├── __init__.py
│   ├── api_client.py            # SGLang API communication
│   ├── csv_handler.py           # CSV output handling
│   ├── logger.py                # Logging utilities
│   └── video_processor.py       # Main inference orchestration
├── videos/                      # Place input videos here
│   └── .gitkeep
├── results/                     # Output CSVs saved here
│   └── .gitkeep
├── main.py                      # Entry point
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🚀 Quick Start

### 1. Clone/Download the Project

```bash
git clone <your-repo> && cd video-inference-automation
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Your Models

Edit `config/models.yaml` to add your models:

```yaml
models:
  - name: "Qwen3-VL-8B"
    model_path: "Qwen/Qwen3-VL-8B-Instruct"
    port: 30000
    host: "localhost"
    attention_backend: "flashinfer"
    mm_attention_backend: "flashinfer"
    enabled: true

  - name: "AnotherModel"
    model_path: "model/path"
    port: 30001
    host: "localhost"
    enabled: false  # Set to true to enable
```

### 4. Prepare Your Videos

Place video files in the `videos/` directory:

```bash
cp /path/to/your/video.mp4 videos/
```

### 5. Start SGLang Servers

For each enabled model, start an SGLang server:

```bash
# Set optimizations for your GPU
export SGLANG_USE_CUDA_IPC_TRANSPORT=1
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

# Launch server for model 1 (port 30000)

python3 -m sglang.launch_server \
   --model-path Qwen/Qwen3-VL-8B-Instruct \
   --port 30000 \
   --host 0.0.0.0 \
   --attention-backend flashinfer \
   --keep-mm-feature-on-device \
   --trust-remote-code

# In a separate terminal, launch server for model 2 (port 30001) if needed
# python3 -m sglang.launch_server --model-path ... --port 30001
```

### 6. Start Video Hosting Server

Your videos need to be accessible via HTTP. Start a simple HTTP server in the videos directory:

```bash
cd videos
python3 -m http.server 8000
# Server runs at http://localhost:8000
```

### 7. Run the Pipeline

```bash
python3 main.py
```

Or with custom options:

```bash
python3 main.py \
  --config config/models.yaml \
  --video-url-base http://localhost:8000 \
  --log-level INFO \
  --log-file inference.log
```

## 📋 Configuration

### models.yaml

**Models Section:**
- `name`: Display name for the model
- `model_path`: Model identifier from HuggingFace or local path
- `port`: SGLang server port for this model
- `host`: Server hostname
- `attention_backend`: Attention implementation (flashinfer, fa3, etc.)
- `mm_attention_backend`: Multimodal attention backend
- `enabled`: Set to `true` to include in pipeline, `false` to skip

**Inference Section:**
- `max_tokens`: Maximum tokens in model response (default: 1024)
- `temperature`: Sampling temperature, 0.0 for deterministic (default: 0.0)
- `timeout`: Request timeout in seconds (default: 60)

**Directories Section:**
- `videos`: Input videos directory (default: ./videos)
- `results`: Output CSVs directory (default: ./results)

### Custom System Prompt

By default, the pipeline uses a corridor cube-counting prompt. Customize it in two ways:

**1. Edit Default in Code:**
Modify `DEFAULT_SYSTEM_PROMPT` in `video_processor.py`

**2. Use Command Line:**
```bash
python3 main.py --custom-prompt "Your custom prompt here"
```

## 🎯 Usage Examples

### Basic Run
```bash
python3 main.py
```

### Debug Mode with Verbose Logging
```bash
python3 main.py --log-level DEBUG
```

### Custom Prompt
```bash
python3 main.py --custom-prompt "Analyze this video and describe what you see"
```

### Multiple Models
1. Update `config/models.yaml` to add multiple models
2. Start all SGLang servers on different ports
3. Run: `python3 main.py`

### Remote Video Server
```bash
python3 main.py --video-url-base http://192.168.1.100:8000
```

## 📊 Output Format

Results are saved in `results/` directory as CSV files, one per video.

**Example CSV:** `long_corridor_random_6_cubes_results.csv`

```
timestamp,video_name,model_name,model_path,status,response,error_message
2024-01-15T10:30:45.123456,video.mp4,Qwen3-VL-8B,Qwen/Qwen3-VL-8B-Instruct,success,"Reasoning: ...",
2024-01-15T10:35:12.654321,video.mp4,LLaVA,llavav1.6,success,"Reasoning: ...",
```

**Columns:**
- `timestamp`: When the inference was run
- `video_name`: Input video filename
- `model_name`: Model display name
- `model_path`: Model identifier
- `status`: `success` or `error`
- `response`: Model's response text
- `error_message`: Error details if status is `error`

## 🔧 Troubleshooting

### "Cannot reach server at http://localhost:30000"

✓ Check if SGLang server is running on port 30000
✓ Verify port number in `config/models.yaml`
✓ Check firewall settings

### "Video not found" or 404 errors

✓ Verify videos are in the `videos/` directory
✓ Confirm HTTP server is running: `python3 -m http.server 8000`
✓ Check `--video-url-base` matches your server

### "Request timeout after 60s"

✓ Increase timeout in `config/models.yaml`:
```yaml
inference:
  timeout: 120  # Increase to 120 seconds
```
✓ Try smaller videos first to test
✓ Check GPU memory usage

### Out of Memory errors

✓ Process fewer videos at once
✓ Reduce `max_tokens` in config
✓ Check GPU has enough VRAM for model

### CSV appears empty or not being written

✓ Check write permissions in `results/` directory
✓ Verify disk space is available
✓ Check `inference.log` for detailed error messages

## 📝 Customization

### Adding a Custom Task

Edit `video_processor.py` and modify `DEFAULT_SYSTEM_PROMPT`:

```python
DEFAULT_SYSTEM_PROMPT = """
Your custom instruction here.

Format your response as:
Analysis: [your analysis]
Summary: [summary]
"""
```

### Adding More Models

Edit `config/models.yaml`:

```yaml
models:
  - name: "MyNewModel"
    model_path: "path/to/model"
    port: 30002
    host: "localhost"
    attention_backend: "flashinfer"
    mm_attention_backend: "flashinfer"
    enabled: true
```

Then start SGLang server on port 30002 and run the pipeline.

### Processing Specific Videos

Temporarily move other videos out of the `videos/` directory, or modify the code to accept a video filter.

## 🛠️ Advanced Features

### Logging

- Console logs: Always displayed
- File logs: Saved to `inference.log`
- Log level: Control with `--log-level`

Access logs at any time:
```bash
tail -f inference.log  # Real-time monitoring
```

### Health Checks

The pipeline automatically checks server health before processing:
```
✓ Server at http://localhost:30000 is healthy
✗ Server at http://localhost:30001 is not responding
```

### Summary Statistics

After processing, view results summary:
```
📊 Results saved:
  Total CSV files: 3
  Total results: 9 (3 models × 3 videos)
  Output directory: ./results
```

## 📦 Dependencies

- `requests`: HTTP client for API communication
- `pyyaml`: YAML configuration file parsing

Install with: `pip install -r requirements.txt`

## 🤝 Extending the Pipeline

The modular design allows easy extension:

1. **Custom API clients**: Modify `api_client.py`
2. **Different output formats**: Extend `csv_handler.py`
3. **Pre/post processing**: Modify `video_processor.py`
4. **Parallel processing**: Wrap processing loops with `concurrent.futures`

## ⚠️ Important Notes

1. **Video Hosting**: Videos must be accessible via HTTP by the SGLang server
2. **Timeouts**: Longer videos may need longer timeouts
3. **Temperature**: Set to 0.0 for deterministic outputs (counting, exact tasks)
4. **GPU Memory**: Run multiple models sequentially or on separate GPUs

## 📄 License

[Your License Here]

## 💡 Tips

- Start with one video and one model to verify everything works
- Check `inference.log` for detailed error messages
- Use `--log-level DEBUG` for troubleshooting
- Ensure adequate disk space for results CSVs
- Monitor GPU usage during processing

---

For issues or questions, check the logs and configuration first!
