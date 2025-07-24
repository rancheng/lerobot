# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Installation and Setup
```bash
# Clone and install
git clone https://github.com/huggingface/lerobot.git
cd lerobot

# Create conda environment (Python 3.10 required)
conda create -y -n lerobot python=3.10
conda activate lerobot
conda install ffmpeg -c conda-forge

# Install LeRobot
pip install -e .

# Install with specific environments (optional)
pip install -e ".[aloha, pusht]"  # for simulation environments

# WandB setup (optional)
wandb login
```

### Testing
```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/datasets/
pytest tests/policies/
pytest tests/robots/

# End-to-end testing (CPU/GPU)
make test-end-to-end DEVICE=cpu
make test-end-to-end DEVICE=cuda

# Specific policy testing
make test-act-ete-train DEVICE=cpu
make test-diffusion-ete-train DEVICE=cpu
make test-tdmpc-ete-train DEVICE=cpu
```

### Code Quality
```bash
# Linting (using ruff)
ruff check .
ruff format .

# Security checking
bandit -r lerobot/
```

### Core Scripts
```bash
# Train a policy
python lerobot/scripts/train.py --policy.type=act --env.type=aloha

# Evaluate a policy
python lerobot/scripts/eval.py --policy.path=path/to/model

# Visualize datasets
python lerobot/scripts/visualize_dataset.py --repo-id lerobot/pusht --episode-index 0

# Control real robot
python lerobot/scripts/control_robot.py

# Configure motors
python lerobot/scripts/configure_motor.py
```

## Architecture Overview

### Core Structure
- **lerobot/common/policies/**: Policy implementations (ACT, Diffusion, TDMPC, VQ-BeT, Pi0, SmolVLA, IDP3)
- **lerobot/common/datasets/**: Dataset handling and transformations with LeRobotDataset format
- **lerobot/common/envs/**: Simulation environments (ALOHA, PushT, XArm)
- **lerobot/common/robot_devices/**: Real hardware interfaces (Dynamixel/Feetech motors, cameras)
- **lerobot/scripts/**: Main executable scripts for training, evaluation, and robot control
- **lerobot/configs/**: Configuration classes with command-line overrides

### Policy Architecture
Each policy follows a consistent pattern:
- `configuration_*.py`: Config dataclass defining hyperparameters
- `modeling_*.py`: PyTorch model implementation
- Integration through `lerobot/common/policies/factory.py`

### Dataset Format
LeRobotDataset uses:
- HuggingFace datasets (Arrow/Parquet) for metadata
- MP4 videos for camera data (space-efficient)
- Safetensors for tensor data
- JSON for episode information and statistics

### Key Features
- **Multi-modal**: Supports cameras, robot states, actions
- **Temporal**: `delta_timestamps` for retrieving frames across time
- **Hub Integration**: Seamless upload/download from HuggingFace Hub
- **Video Compression**: Efficient storage with ffmpeg encoding

### Configuration System
Uses dataclasses with command-line override capability:
```bash
# Override config parameters
--policy.learning_rate=1e-4 --dataset.episodes="[0,1,2]"
```

### Robot Hardware Support
- **Motors**: Dynamixel and Feetech servos with calibration
- **Cameras**: OpenCV and Intel RealSense support
- **Robots**: ALOHA, Koch, SO-100/101, LeKiwi, Stretch

### Training and Evaluation
- Automatic checkpointing with configurable frequency
- WandB integration for experiment tracking
- Multi-environment parallel evaluation
- Resume training from checkpoints

## Important Notes

### Dependencies
- Python 3.10+ required
- PyTorch 2.2.1+ with CUDA support recommended
- ffmpeg with libsvtav1 encoder support
- Platform-specific dependencies in pyproject.toml

### Optional Hardware Extensions
Install additional dependencies based on hardware:
- `pip install -e ".[dynamixel]"` for Dynamixel motors
- `pip install -e ".[intelrealsense]"` for RealSense cameras
- `pip install -e ".[stretch]"` for Stretch robot

### Testing
208 test functions across 25 test files ensure code quality. Use `make test-end-to-end` for comprehensive validation of training and evaluation pipelines.