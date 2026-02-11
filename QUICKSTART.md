# Quick Start Guide

Get up and running in 5 minutes!

## Step 1: Install Dependencies (1 min)

```bash
pip install -r requirements.txt
```

## Step 2: Create Project Structure (1 min)

```bash
# Create necessary directories
mkdir -p config videos results logs

# Move config file to right location
mv config_models.yaml config/models.yaml
```

## Step 3: Place Your Videos (1 min)

```bash
# Copy videos to the videos directory
cp /path/to/your/videos/*.mp4 videos/

# Verify
ls -la videos/
```

## Step 4: Start SGLang Server (1 min)

**Terminal 1:** Start the model server

```bash
# Set GPU optimizations (adjust for your hardware)
export SGLANG_USE_CUDA_IPC_TRANSPORT=1
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

# Launch server
python3 -m sglang.launch_server \
   --model-path Qwen/Qwen3-VL-8B-Instruct \
   --port 30000 \
   --host 0.0.0.0 \
   --attention-backend flashinfer \
   --keep-mm-feature-on-device \
   --trust-remote-code
```

Wait for message: `Server started at http://0.0.0.0:30000`

## Step 5: Start Video Server (1 min)

**Terminal 2:** Host your videos

```bash
cd videos
python3 -m http.server 8000
```

You should see: `Serving HTTP on 0.0.0.0 port 8000`

## Step 6: Run the Pipeline (1 min)

**Terminal 3:** Start inference

```bash
# Run with default settings
python3 main.py

# Or with custom options
python3 main.py --log-level DEBUG
```

## That's it! 🎉

Check the `results/` directory for CSV outputs.

---

## What's Happening?

1. **SGLang Server** (Terminal 1): Runs the Vision Language Model
2. **HTTP Server** (Terminal 2): Serves videos to the model
3. **Pipeline** (Terminal 3): Orchestrates inference and saves results

## Verify It's Working

Watch the output:

```
[INFO] Found 3 video file(s)
[INFO] Checking server health...
[INFO] ✓ Server at http://localhost:30000 is healthy

============================================================
Processing video: my_video.mp4
============================================================

▶ Running inference with Qwen3-VL-8B...
✓ Result saved to ./results/my_video_results.csv
✓ Qwen3-VL-8B completed successfully
```

## Next Steps

- **Multiple models?** Add more to `config/models.yaml` and start additional servers
- **Custom prompt?** Use `--custom-prompt "Your prompt here"` flag
- **Troubleshooting?** Check `README.md` for detailed troubleshooting guide

## Common Issues

### Port already in use
```bash
# Find process on port 30000
lsof -i :30000
# Kill it
kill -9 <PID>
```

### "Cannot reach server"
```bash
# Test connectivity
curl http://localhost:30000/health
```

### Video not found
```bash
# Verify videos are accessible
curl http://localhost:8000/video.mp4
```

---

Happy inferencing! 🚀
