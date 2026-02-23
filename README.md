# Navigation — Deep Reinforcement Learning

**Author**: Tim Wilcoxson

A Double DQN agent with Dueling network architecture that learns to collect yellow bananas (+1 reward) while avoiding blue bananas (-1 reward) in a Unity ML-Agents environment.

## Environment

- **State space**: 37 dimensions (velocity + ray-based perception of objects)
- **Action space**: 4 discrete actions — forward, backward, turn left, turn right
- **Solve condition**: Average score >= 13 over 100 consecutive episodes

## Project Structure

| File | Description |
|---|---|
| `Navigation.ipynb` | Main training notebook |
| `model.py` | Dueling DQN network architecture |
| `dqn_agent.py` | Double DQN agent with experience replay |
| `Report.md` | Detailed report: algorithm, results, future work |
| `checkpoint.pth` | Trained model weights |
| `scores_plot.png` | Training rewards plot |
| `python/` | Bundled Unity ML-Agents Python package (v0.4) |
| `Banana.app/` | macOS Unity environment |

## Setup

### Prerequisites

- [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- macOS (this project uses the macOS Banana.app environment)

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/trwilcoxson/dqn-banana-navigation.git
   cd dqn-banana-navigation
   ```

2. Create and activate the conda environment:
   ```bash
   conda create -n drlnd-nav python=3.10 -y
   conda activate drlnd-nav
   ```

3. Install dependencies (includes PyTorch, NumPy, Jupyter, and all other required packages):
   ```bash
   cd python
   pip install .
   cd ..
   ```

4. Install the Jupyter kernel:
   ```bash
   python -m ipykernel install --user --name drlnd-nav --display-name "Python (drlnd-nav)"
   ```

5. Download the Banana environment for your OS and place it in the project root:
   - [macOS](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
   - [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
   - [Windows 64-bit](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

   **macOS users**: After unzipping, remove the quarantine attribute so the app can launch:
   ```bash
   xattr -cr Banana.app
   ```

   **Linux/Windows users**: Update the `file_name` path in the notebook's environment initialization cell to match your downloaded binary (e.g., `"Banana_Linux/Banana.x86_64"`).

## Training

```bash
conda activate drlnd-nav
jupyter notebook Navigation.ipynb
```

Select the **"Python (drlnd-nav)"** kernel and run all cells. The agent typically solves the environment in 400–600 episodes.

## Results

See [Report.md](Report.md) for the full learning algorithm description, architecture details, training plot, and ideas for future work.
